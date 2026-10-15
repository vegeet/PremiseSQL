import argparse
import ast
import collections
import hashlib
import json
import pathlib
import random
import re
import time

from .build_prompt import (LABELS, OUTPUT_SCHEMA, PROMPT_VERSION, TAXONOMY_VERSION,
                           build_failure_pattern_prompt, error_type)

ROOT = pathlib.Path(__file__).resolve().parent.parent

ATTRIBUTION_CONTEXT_FIELDS = (
    "schema", "schema_provenance", "schema_exposure_status", "original_schema_linker",
    "reference_schema", "reference_schema_provenance", "original_evidence_provenance",
    "paired_valid_question", "paired_valid_sql", "paired_valid_evidence",
    "paired_valid_provenance", "data_coverage", "flaw_description", "context_evidence",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def exact_quote(original, quote):
    """Recover only unique spans differing in whitespace, never paraphrases."""
    if not isinstance(original, str) or not isinstance(quote, str) or not quote.strip():
        return None
    if quote in original:
        return quote
    parts = quote.split()
    pattern = r"\s+".join(re.escape(part) for part in parts)
    matches = list(re.finditer(pattern, original))
    return matches[0].group() if len(matches) == 1 else None


def get_key(item):
    # Source contains dataset, tested model, stage and question category.
    return digest([item.get("source", "").replace("\\", "/"), item.get("sample_id"),
                   item.get("db_id"), item.get("question"), item.get("sql"), item.get("reason")])


def read_list(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or any(not isinstance(x, dict) for x in data):
        raise ValueError(f"Expected list of objects: {path}")
    return data


def escape_inner_quotes(text):
    escaped = []
    backslashes = 0
    for char in text:
        if char == '"' and backslashes % 2 == 0:
            escaped.append("\\")
        escaped.append(char)
        backslashes = backslashes + 1 if char == "\\" else 0
    return "".join(escaped)


def repair_evidence_quoting(text, sources):
    """Escape a malformed quote only when it matches supplied source text."""
    pattern = re.compile(
        r'(?P<prefix>"(?:field|source)"\s*:\s*"(?P<field>question|reason|sql)"'
        r'\s*,\s*"quote"\s*:\s*)"(?P<quote>.*?)"(?=\s*[,}])',
        re.DOTALL,
    )

    def repair(match):
        quoted = match.group("quote")
        try:
            decoded = json.loads('"' + escape_inner_quotes(quoted) + '"')
        except json.JSONDecodeError:
            return match.group()
        original = sources.get(match.group("field"))
        anchored = exact_quote(original, decoded)
        # Claude can render typographic source quotes as straight quotes while
        # also forgetting JSON escaping. This comparison permits syntactic
        # recovery only: retain its decoded text so the exact-evidence gate
        # will still discard a non-verbatim quotation.
        typography = str.maketrans({"“": '"', "”": '"'})
        same_text = isinstance(original, str) and exact_quote(
            original.translate(typography), decoded.translate(typography)
        ) is not None
        if anchored is None and not same_text:
            return match.group()
        return match.group("prefix") + json.dumps(decoded, ensure_ascii=False)

    return pattern.sub(repair, text)


def repair_explanation_quoting(text):
    """Escape inner quotes in a known free-text field occupying a whole line."""
    pattern = re.compile(
        r'(?P<prefix>^[ \t]*"(?:explanation|context_evidence|evidence_limitations|'
        r'description|why_existing_labels_do_not_fit)"[ \t]*:[ \t]*)'
        r'"(?P<text>[^\r\n]*)"(?P<suffix>[ \t]*,?[ \t]*$)',
        re.MULTILINE,
    )

    def repair(match):
        value = match.group("text")
        try:
            json.loads('"' + value + '"')
            return match.group()
        except json.JSONDecodeError:
            return (match.group("prefix") + '"' + escape_inner_quotes(value)
                    + '"' + match.group("suffix"))

    return pattern.sub(repair, text)


def read_json_object(value, sources=None):
    """Parse one JSON object, tolerating provider prose or Markdown fences."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError("Structured response must be a JSON object or string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as direct_error:
        # Read the root object after prose/fences. Never scan into a malformed
        # root and accidentally accept an inner assessment or evidence object.
        start = value.find("{")
        if start == -1:
            raise direct_error
        candidate = value[start:]
        try:
            parsed, _ = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            repaired = repair_evidence_quoting(candidate, sources or {})
            repaired = repair_explanation_quoting(repaired)
            parsed, _ = json.JSONDecoder().raw_decode(repaired)
    if not isinstance(parsed, dict):
        raise ValueError("Structured response must be a JSON object")
    return parsed


CLAUDE_FLAT_ASSESSMENTS = {
    "M01_assumption_injection": "M01",
    "M02_contradiction_reinterpretation": "M02",
    "M03_context_evidence_misuse": "M03",
    "M04_explicit_constraint_omission": "M04",
}


def normalize_response_shape(response):
    """Normalize a deterministic Claude rendering of the requested assessment array."""
    if "label_assessments" in response:
        return response
    present = [name for name in CLAUDE_FLAT_ASSESSMENTS if name in response]
    if not present:
        return response
    normalized = dict(response)
    assessments = []
    for field, label_id in CLAUDE_FLAT_ASSESSMENTS.items():
        value = normalized.pop(field, None)
        if not isinstance(value, dict):
            # Leave an incomplete array for the ordinary schema validator to
            # reject with an exact path instead of guessing semantic content.
            continue
        assessment = dict(value)
        assessment.setdefault("label_id", label_id)
        assessments.append(assessment)
    normalized["label_assessments"] = assessments
    return normalized


def normalize_assessment_controls(response):
    """Canonicalize provider spellings for fields whose meaning is deterministic."""
    assessments = response.get("label_assessments")
    if not isinstance(assessments, list):
        return response
    verdict_aliases = {
        "supported": "supported",
        "not_supported": "not_supported",
        "notsupported": "not_supported",
        "uncertain": "uncertain",
    }
    context_aliases = {
        "bounded_examples": "bounded_examples",
        "bounded_example": "bounded_examples",
        "boundedexamples": "bounded_examples",
        "original_schema": "original_schema",
        "originalschema": "original_schema",
        "none": "none",
    }
    transformations = {
        "M01": "addition_or_substitution",
        "M02": "reinterpretation",
        "M03": "evidence_misuse",
        "M04": "deletion",
    }
    for assessment in assessments:
        if not isinstance(assessment, dict):
            continue
        # Claude occasionally emits this redundant non-schema alias alongside
        # replacement_status.  It carries no additional semantic information and
        # should not invalidate an otherwise complete four-label response.
        assessment.pop("replacement_scope", None)
        verdict_key = re.sub(r"[^a-z]", "_", str(assessment.get("verdict", "")).casefold())
        verdict_key = re.sub(r"_+", "_", verdict_key).strip("_")
        compact_verdict = verdict_key.replace("_", "")
        verdict = verdict_aliases.get(verdict_key, verdict_aliases.get(compact_verdict))
        if verdict:
            assessment["verdict"] = verdict
        label_id = assessment.get("label_id")
        if label_id in transformations and verdict:
            assessment["transformation"] = (
                transformations[label_id] if verdict == "supported" else "none"
            )
        if label_id != "M03" or verdict != "supported":
            assessment["context_basis"] = "none"
        else:
            context_key = re.sub(
                r"_+", "_",
                re.sub(r"[^a-z]", "_", str(assessment.get("context_basis", "")).casefold()),
            ).strip("_")
            canonical_context = context_aliases.get(
                context_key, context_aliases.get(context_key.replace("_", ""))
            )
            if canonical_context:
                assessment["context_basis"] = canonical_context
        if assessment.get("target_requirement") is None:
            assessment["target_requirement"] = ""
        if assessment.get("context_evidence") is None:
            assessment["context_evidence"] = ""
        if assessment.get("evidence") is None:
            assessment["evidence"] = []
        normalize_evidence_fields(assessment.get("evidence"))
    novel = response.get("novel_candidate")
    if isinstance(novel, dict):
        normalize_evidence_fields(novel.get("evidence"))
    return response


def normalize_evidence_fields(entries):
    """Accept the exact source/field alias without changing quote content."""
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict) or "source" not in entry:
            continue
        source = entry["source"]
        if isinstance(source, str) and source in {"question", "reason", "sql"}:
            if "field" not in entry or entry["field"] == source:
                entry["field"] = entry.pop("source")


def merge_repeated_m01(response):
    """One M01 assessment may describe several independently supported instances."""
    assessments = response.get("label_assessments")
    if not isinstance(assessments, list):
        return response, []
    instances = [a for a in assessments if isinstance(a, dict) and a.get("label_id") == "M01"]
    if (len(instances) < 2
            or any(a.get("verdict") != "supported"
                   or not isinstance(a.get("evidence"), list)
                   or not isinstance(a.get("explanation"), str)
                   for a in instances)):
        return response, []
    merged = dict(instances[0])
    merged["explanation"] = " ".join(a["explanation"] for a in instances)
    merged["evidence"] = []
    targets = []
    for instance in instances:
        target = instance.get("target_requirement")
        if isinstance(target, str) and target:
            targets.append(target)
        evidence = list(instance["evidence"])
        if target:
            evidence.append({"field": "question", "quote": target})
        for entry in evidence:
            if entry not in merged["evidence"]:
                merged["evidence"].append(entry)
    result = []
    for assessment in assessments:
        if isinstance(assessment, dict) and assessment.get("label_id") == "M01":
            if merged not in result:
                result.append(merged)
        else:
            result.append(assessment)
    return {**response, "label_assessments": result}, targets


def request_prompt(args, sample, model_output, prompt_builder=build_failure_pattern_prompt):
    prompt = prompt_builder(sample, model_output)
    if str(args.language_model).casefold().startswith("claude/"):
        prompt += (
            "\n\nCLAUDE OUTPUT SHAPE REQUIREMENT:\n"
            "Return exactly these five top-level fields: recognition, "
            "response_behavior, label_assessments, novel_candidate, and "
            "evidence_limitations. label_assessments must be one JSON array "
            "containing exactly four assessment objects whose label_id values "
            "are M01, M02, M03, and M04. Do not create top-level fields named "
            "M01_assumption_injection, M02_contradiction_reinterpretation, "
            "M03_context_evidence_misuse, or M04_explicit_constraint_omission."
            "\nEvery assessment object must contain all fields specified below. "
            "Every evidence entry must use exactly field and quote; field must "
            "be question, reason, or sql. Do not use source as a field name. "
            "Use exactly the enum strings in the schema. This schema describes "
            "the required output structure; return the populated result object, "
            "not the schema itself.\nOUTPUT JSON SCHEMA:\n"
            + json.dumps(OUTPUT_SCHEMA, ensure_ascii=False)
        )
    return prompt


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def input_issue(item):
    for field in ("reason", "sql"):
        value = item.get(field)
        if value is not None and not isinstance(value, str):
            return f"{field}: non-string output"
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, dict) and "type" in parsed:
                return f"{field}: JSON-schema placeholder"
    if not item.get("reason") and not item.get("sql"):
        return "reason and sql are both empty"
    return None


def analysis_disposition(record):
    """Separate a taxonomy gap from an intentional zero-label decision."""
    status = record.get("analysis_status")
    if status != "success":
        return status or "unknown"
    if record.get("supported_label_ids"):
        return "attributed"
    if record.get("novel_candidate") is not None:
        return "taxonomy_candidate"
    assessments = record.get("label_assessments") or []
    if any(item.get("verdict") == "uncertain" for item in assessments):
        return "insufficient_evidence"
    return "no_observable_core_mechanism"


def check_shape(value, schema, path="$"):
    """Validate the small schema subset used here, without extra dependencies."""
    if "anyOf" in schema:
        for alternative in schema["anyOf"]:
            try:
                check_shape(value, alternative, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path}: invalid nullable object")
    kind = schema["type"]
    if kind == "null":
        if value is not None:
            raise ValueError(f"{path}: expected null")
    elif kind == "string":
        if not isinstance(value, str) or ("enum" in schema and value not in schema["enum"]):
            raise ValueError(f"{path}: invalid string/enum")
    elif kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array")
        for index, entry in enumerate(value):
            check_shape(entry, schema["items"], f"{path}[{index}]")
    elif kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object")
        expected = set(schema["properties"])
        actual = set(value)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(
                f"{path}: output fields differ; missing={missing}, unknown={unknown}"
            )
        for name, subschema in schema["properties"].items():
            check_shape(value[name], subschema, f"{path}.{name}")


def safe_validation_detail(exc):
    """Return local validation diagnostics without persisting provider payloads."""
    if isinstance(exc, json.JSONDecodeError):
        return f"invalid JSON at line {exc.lineno}, column {exc.colno}"
    if isinstance(exc, ValueError):
        return str(exc)[:500]
    return None


def validate_response(response, model_output, sample=None):
    sample = sample or {}
    response = normalize_assessment_controls(
        normalize_response_shape(read_json_object(
            response, {**model_output, "question": sample.get("question")}
        ))
    )
    response, additional_m01_targets = merge_repeated_m01(response)
    detection = sample.get("detection")
    type_identification = sample.get("type_identification")
    sql_behavior = sample.get("sql_behavior")
    # These two fields are behavioral annotations already fixed by the validated
    # evaluation. Normalize them before schema validation so provider wording
    # cannot reject an otherwise usable mechanism assessment.
    if detection == 0:
        response["recognition"] = "not_explicit"
    elif detection == 1:
        response["recognition"] = "explicit_flaw"
    else:
        response["recognition"] = "uncertain"
    response["response_behavior"] = {
        0: "sql_generated", 1: "conditional_sql", 2: "abstained"
    }.get(sql_behavior, "uncertain")
    check_shape(response, OUTPUT_SCHEMA)
    # Validated metrics determine behavior fields only. Mechanism labels require
    # independent semantic/contextual evidence and are never derived from metrics.
    m03 = next((a for a in response["label_assessments"] if a["label_id"] == "M03"), None)
    provenance = str(sample.get("schema_provenance", ""))
    original_exposure = provenance.startswith("original_schema supplied") or (
        sample.get("schema") is not None and sample.get("schema_exposure_status") == "reconstructed")
    trusted_m03_context = bool(m03) and (
        (m03["context_basis"] == "original_schema" and original_exposure)
        or (m03["context_basis"] == "bounded_examples" and bool(sample.get("context_evidence"))))
    if m03 and m03["verdict"] == "supported" and not trusted_m03_context:
        m03["verdict"] = "uncertain"
        m03["target_requirement"] = ""
        m03["transformation"] = "none"
        m03["context_evidence"] = ""
        m03["context_basis"] = "none"
        m03["explanation"] = ("The selected context path lacks verified original-schema exposure "
                              "or bounded-example construction; complete reference schema alone "
                              "cannot establish original visibility. "
                              + m03["explanation"])
    assessments = response["label_assessments"]
    if collections.Counter(a["label_id"] for a in assessments) != collections.Counter(LABELS.keys()):
        raise ValueError("require each fixed label exactly once")

    expected_transformations = {
        "M01": "addition_or_substitution",
        "M02": "reinterpretation",
        "M03": "evidence_misuse",
        "M04": "deletion",
    }

    def incomplete_omission_target(text):
        normalized = " ".join(text.casefold().strip().split())
        if normalized in {
            "all", "all the", "overall", "situation", "the situation",
            "the order id", "the order id?", "the competition", "the authors",
            "at least", "more", "onward", "other details", "the inspection type",
            "a certain inspection type", "a single shipment", "single root beer",
            "high-priority", "high-performing", "high-impact", "high-potential",
            "veteran", "veteran players", "premium room",
        }:
            return True
        if re.fullmatch(
            r"(?:in|during|on|at|for|among)\s*(?:[?.!,，。…]+|providers\s+in)?",
            normalized,
        ):
            return True
        if re.fullmatch(
            r"(?:(?:in|from|of|for|by|during|on)\s+)?that\s+"
            r"(?:area|city|country|image|movie|team|person|direction|day)",
            normalized,
        ):
            return True
        # A deictic phrase without its referent is not a semantically complete
        # constraint.  The paired valid query may reveal what construction
        # removed, but it was not part of the flawed question seen by the model.
        if re.fullmatch(
            r"(?:(?:in|during|on|at|for)\s+)?"
            r"(?:(?:that|this|specified)|the(?:\s+specified)?|a\s+given)\s+"
            r"(?:period|time|day|year|season|regular\s+season|state|place|location)",
            normalized,
        ):
            return True
        return bool(re.search(
            r"\b(?:more|less|greater|lower|higher)\s+than\s*(?:\.|\?|…)?$",
            normalized,
        ))

    question = sample.get("question")
    for assessment in assessments:
        supported = assessment["verdict"] == "supported"
        label_id = assessment["label_id"]
        if label_id != "M03" or not supported:
            assessment["context_basis"] = "none"
        # transformation and target_requirement are control metadata. Normalize
        # them instead of failing an otherwise usable semantic assessment.
        assessment["transformation"] = (expected_transformations[label_id]
                                        if supported else "none")
        if not supported or label_id == "M03":
            assessment["target_requirement"] = ""
            continue

        target = exact_quote(question, assessment["target_requirement"].strip())
        if target is None:
            exact_question_quotes = [
                anchored for e in assessment["evidence"] if e["field"] == "question"
                if (anchored := exact_quote(question, e["quote"])) is not None
            ]
            if exact_question_quotes:
                assessment["target_requirement"] = min(exact_question_quotes, key=len)
            else:
                assessment.update(
                    verdict="uncertain", target_requirement="", transformation="none",
                    explanation=("Target requirement could not be anchored to an exact "
                                 "question span. " + assessment["explanation"]))
        else:
            assessment["target_requirement"] = target

        # The target itself is an exact question quote. Add it as control
        # evidence when the model supplied a valid span but forgot to duplicate
        # it in the evidence array. This prevents formatting-only false negatives.
        if (assessment["verdict"] == "supported"
                and assessment["target_requirement"]
                and not any(e["field"] == "question" for e in assessment["evidence"])):
            assessment["evidence"].append({
                "field": "question", "quote": assessment["target_requirement"]
            })

    m04 = next(a for a in assessments if a["label_id"] == "M04")
    m01 = next(a for a in assessments if a["label_id"] == "M01")

    def reject_control(assessment, detail):
        assessment.update(
            verdict="not_supported", target_requirement="", transformation="none",
            evidence=[], context_evidence="",
            explanation=f"Decision-control gate: {detail} " + assessment["explanation"],
        )

    if (m01["verdict"] == "supported"
            and m01["replacement_status"] != "concrete_replacement"):
        reject_control(m01, "M01 requires an observable concrete replacement.")

    if m04["verdict"] == "supported":
        if m04["target_status"] != "complete_explicit":
            reject_control(m04, "M04 requires a complete explicit current-question target.")
        elif sql_behavior == 2 or m04["implementation_scope"] == "complete_abstention":
            reject_control(m04, "M04 cannot be assigned to complete abstention.")
        elif m04["implementation_scope"] != "remaining_implementation":
            reject_control(m04, "M04 requires a remaining partial implementation.")
        elif m04["replacement_status"] != "no_replacement":
            reject_control(m04, "M04 requires deletion without a replacement.")

    # Catch an internally contradictory structured verdict before it becomes a
    # supported label. Keep this label-explicit so ordinary uses of "not"
    # elsewhere in an explanation are not semantic self-rejections.
    for assessment in (m01, m04):
        label_id = assessment["label_id"]
        self_rejection = re.search(
            rf"(?:\b{label_id}\s+(?:(?:is|was)\s+not\s+supported|"
            rf"is\s+unsupported|does\s+not\s+apply)\b|"
            rf"\bno\s+{label_id}\s+mechanism\s+(?:is\s+)?present\b|"
            rf"\brather\s+than\s+{label_id}\b)",
            assessment["explanation"], re.IGNORECASE,
        )
        if assessment["verdict"] == "supported" and self_rejection:
            reject_control(
                assessment,
                f"The explanation explicitly rejects {label_id}.",
            )

    # A model sometimes marks M04 as deletion while its own explanation says
    # that the same requirement received a concrete replacement. Prefer the
    # explicit semantic conclusion over the inconsistent control enum.
    if (m04["verdict"] == "supported"
            and re.search(
                r"(?:\b(?:does|did)\s+(?:implement|supply|use|choose)\s+"
                r"(?:a|the)\s+(?:concrete\s+)?(?:replacement|substitute|proxy)\b|"
                r"\b(?:maps?|mapped)\s+(?:this|the\s+(?:requirement|phrase|modifier|"
                r"concept|term))\s+to\b|"
                r"\breplaces?\s+(?:this|the\s+(?:requirement|phrase|modifier|"
                r"concept|term))\s+with\b)",
                m04["explanation"], re.IGNORECASE,
            )):
        reject_control(m04, "The explanation explicitly identifies a replacement.")

    m04 = next(a for a in assessments if a["label_id"] == "M04")
    if (m04["verdict"] == "supported"
            and incomplete_omission_target(m04["target_requirement"])):
        m04.update(
            verdict="not_supported", target_requirement="", transformation="none",
            evidence=[], context_evidence="",
            explanation=("Code-level boundary: M04 requires an explicit restrictive "
                         "requirement, not a deictic or syntactically incomplete phrase."))

    # Paired references can locate a perturbation but cannot turn a constraint
    # absent from the current question into an M04 target. Reject only explicit
    # admissions of reference-only support; ordinary mentions of the reference
    # remain available as diagnostic context.
    if (m04["verdict"] == "supported"
            and re.search(
                r"\bpaired[_ -]?valid[_ -]+(?:question|reference)\s+"
                r"(?:contains|specifies|shows|supplies|reveals|confirms)\b.{0,1500}\b"
                r"(?:remove|removes|removed|removing|deleted|disappears?|omitted|originally)\b",
                m04["explanation"], re.IGNORECASE | re.DOTALL,
            )):
        reject_control(
            m04,
            "M04 cannot be supported by a constraint found only in the paired reference.",
        )

    def valid_evidence(entries):
        valid = []
        for entry in entries:
            original = sample.get("question") if entry["field"] == "question" else model_output.get(entry["field"])
            anchored = exact_quote(original, entry["quote"])
            if anchored is not None:
                valid.append({**entry, "quote": anchored})
        return valid

    for a in assessments:
        a["evidence"] = valid_evidence(a["evidence"])
        if not a["explanation"].strip():
            a["explanation"] = "The response did not provide a usable mechanism explanation."
        if a["verdict"] != "supported":
            continue
        evidence_fields = {e["field"] for e in a["evidence"]}
        invalid_support = (
            not a["evidence"]
            or not evidence_fields.intersection({"reason", "sql"})
            or (a["label_id"] == "M03" and not a["context_evidence"].strip())
            or (a["label_id"] in {"M01", "M02", "M04"}
                and "question" not in evidence_fields)
        )
        if invalid_support:
            a.update(
                verdict="uncertain", target_requirement="", transformation="none",
                explanation=("Evidence gate: exact question/output/context evidence was "
                             "insufficient for supported status. " + a["explanation"]))

    # Reject only literal duplicate M01/M03 assessments. The same short output
    # quote can legitimately expose two causes: a positive semantic commitment
    # (M01) and treating bounded examples as exhaustive (M03). Distinct mechanism
    # explanations therefore survive even when their minimal evidence overlaps.
    m01 = next(a for a in assessments if a["label_id"] == "M01")
    m03 = next(a for a in assessments if a["label_id"] == "M03")
    if m01["verdict"] == m03["verdict"] == "supported":
        quotes1 = {(e["field"], e["quote"]) for e in m01["evidence"]
                   if e["field"] in {"reason", "sql"}}
        quotes3 = {(e["field"], e["quote"]) for e in m03["evidence"]
                   if e["field"] in {"reason", "sql"}}
        explanation1 = " ".join(m01["explanation"].casefold().split())
        explanation3 = " ".join(m03["explanation"].casefold().split())
        if quotes1 and quotes1 <= quotes3 and explanation1 == explanation3:
            m01.update(verdict="uncertain", transformation="none",
                       explanation="Independence gate: M01 duplicates the M03 assessment. "
                                   + m01["explanation"])

    # Apply M01/M02 versus M04 exclusivity only after every evidence and
    # independence gate has finalized the competing verdicts. Otherwise an M01
    # that is later rejected can leave behind a stale suppression of valid M04.
    m04 = next(a for a in assessments if a["label_id"] == "M04")
    if m04["verdict"] == "supported":
        deletion_target = m04["target_requirement"].strip().casefold()
        for other_id in ("M01", "M02"):
            other = next(a for a in assessments if a["label_id"] == other_id)
            targets = [other["target_requirement"]]
            if other_id == "M01":
                targets.extend(t for t in additional_m01_targets if exact_quote(question, t))
            overlaps = other["verdict"] == "supported" and any(
                target.strip() and deletion_target == target.strip().casefold()
                for target in targets
            )
            if overlaps:
                m04.update(
                    verdict="not_supported", target_requirement="", transformation="none",
                    evidence=[], context_evidence="",
                    explanation=(f"Code-level exclusivity: {other_id} supplies a replacement "
                                 "or reinterpretation for the same question requirement, so "
                                 "the strict deletion-only condition for M04 is not met."))
                break

    novel = response["novel_candidate"]
    if novel is not None:
        novel["evidence"] = valid_evidence(novel["evidence"])
        fields = {e["field"] for e in novel["evidence"]}
        hedge = re.search(r"\b(?:may|might|possible|possibly|appears|could|likely)\b",
                          novel["description"], re.IGNORECASE)
        excluded_error = re.search(
            r"\b(?:reason[- ]sql|sql (?:implementation|syntax)|join (?:error|defect)|"
            r"wrong (?:join|column|alias|predicate|grouping|aggregation)|"
            r"non[- ]existent (?:column|alias)|output column order)\b",
            novel["description"], re.IGNORECASE)
        if (not novel["evidence"] or not fields.intersection({"reason", "sql"})
                or "question" not in fields
                or not novel["description"].strip()
                or not novel["why_existing_labels_do_not_fit"].strip()
                or hedge or excluded_error):
            response["novel_candidate"] = None
    ids = [i for i in LABELS if any(a["label_id"] == i and a["verdict"] == "supported" for a in assessments)]
    return {**response, "supported_label_ids": ids, "supported_labels": [LABELS[i] for i in ids]}


class FailurePatternEvaluator:
    def __init__(self, args, prompt_builder=build_failure_pattern_prompt, client=None, **kwargs):
        self.args, self.prompt_builder, self.client = args, prompt_builder, client
        self.last_usage = None
        self.last_response = None
        self.last_finish_reason = None
        self.last_error_stage = "request"

    def evaluate(self, sample, model_output):
        prompt = request_prompt(self.args, sample, model_output, self.prompt_builder)
        self.last_usage = self.last_response = self.last_finish_reason = None
        self.last_error_stage = "request"
        if self.client is None:
            # Use the same environment and model mapping as the project, but no
            # shared wrapper retry/fallback that might silently increase cost.
            from dotenv import load_dotenv
            from openai import OpenAI
            from language_model.utils import model_name_dict
            import os
            load_dotenv(ROOT / ".env")
            self.model_name = model_name_dict.get(self.args.language_model, self.args.language_model)
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                                 base_url=os.environ.get("OPENAI_API_BASE_URL"),
                                 max_retries=0, timeout=self.args.timeout)
        completion = self.client.chat.completions.create(
            model=getattr(self, "model_name", self.args.language_model),
            messages=[{"role": "user", "content": prompt}], temperature=0,
            max_tokens=self.args.max_output_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "failure_attribution_v2", "strict": True, "schema": OUTPUT_SCHEMA}},
        )
        self.last_usage = completion.usage.model_dump() if completion.usage else None
        self.last_error_stage = "response_validation"
        if not completion.choices:
            raise ValueError("model response has no choices")
        self.last_finish_reason = completion.choices[0].finish_reason
        self.last_response = completion.choices[0].message.content
        if self.last_finish_reason != "stop":
            raise ValueError("incomplete model response")
        return validate_response(self.last_response, model_output, sample)

    __call__ = evaluate


def prepare_sample(item, args):
    sample = dict(item)
    # Targeted reruns reuse the exact prior exposure, including unknown/reduced
    # schemas. Never regenerate random schema context during an agreement rerun.
    if "attribution_context_snapshot" in item:
        snapshot = item["attribution_context_snapshot"]
        if not isinstance(snapshot, dict) or set(snapshot) != set(ATTRIBUTION_CONTEXT_FIELDS):
            raise ValueError("Invalid attribution_context_snapshot fields")
        sample.update(snapshot)
        return sample
    linker_name = getattr(args, "original_schema_linker", "auto")
    if linker_name == "auto":
        linker_name = item.get("original_schema_linker") or (
            "NoGoldSchemaInfo" if error_type(item)["code"] == "3_2" else "VanillaLinker")
    sample.setdefault("original_schema_linker", linker_name)
    if item.get("original_schema") is not None:
        sample["schema"] = item["original_schema"]
        sample["schema_provenance"] = "original_schema supplied with sample"
        sample["schema_exposure_status"] = "recorded"
        sample["data_coverage"] = item.get(
            "data_coverage",
            "The recorded original schema contains bounded SchemaLinker examples "
            "(normally at most 5 rows per table), not complete database contents.",
        )
        sample["context_evidence"] = item.get(
            "context_evidence",
            "Verified generation record: original_schema is the actual SchemaLinker "
            "exposure, whose displayed INSERT rows are bounded examples and are not "
            "exhaustive database contents.",
        )
    elif item.get("schema") is not None and linker_name != "NoGoldSchemaInfo":
        sample["schema_provenance"] = item.get("schema_provenance", "supplied schema; original exposure unverified")
        sample["schema_exposure_status"] = item.get("schema_exposure_status", "unverified")
    else:
        # A random NoGoldSchemaInfo rerun cannot reproduce historical exposure.
        # Keep full database truth separately from what the tested model saw.
        import schema_linking
        if getattr(args, "database_path", None):
            candidates = [pathlib.Path(args.database_path)]
        elif args.dataset == "spider":
            candidates = [ROOT / "data/spider/database"]
        elif error_type(item)["code"] == "3_2":
            candidates = [ROOT / "data/bird/dev_databases/dev_databases"]
        else:
            candidates = [ROOT / "data/bird/train_databases/train_databases"]
        db_id = item.get("db_id")
        if not isinstance(db_id, str) or pathlib.Path(db_id).name != db_id or "/" in db_id or "\\" in db_id:
            raise ValueError("invalid database ID")
        paths = [base / db_id / f"{db_id}.sqlite" for base in candidates]
        found = [p.resolve() for p in paths if p.is_file()]
        if len(found) == 1:
            reconstruction_linker = "VanillaLinker" if linker_name == "NoGoldSchemaInfo" else linker_name
            linker = schema_linking.load_schema_linker(
                reconstruction_linker,
                gold_schema_path=str(ROOT / "data" / args.dataset / "dev_gold_schema.json"))
            reconstructed = linker.get_schema(str(found[0].parent.parent), db_id,
                                              item.get("question", ""), item.get("evidence"))
            if linker_name == "NoGoldSchemaInfo":
                sample["schema"] = None
                sample["schema_exposure_status"] = "unknown_original"
                sample["schema_provenance"] = (
                    "Original NoGoldSchemaInfo input was not recorded; random removed columns "
                    "cannot be reconstructed from the full database.")
                sample["reference_schema"] = reconstructed
                sample["reference_schema_provenance"] = "complete database reference only; " + str(found[0])
            else:
                sample["schema"] = reconstructed
                sample["schema_exposure_status"] = "reconstructed"
                sample["schema_provenance"] = (
                    f"reconstructed with the original {linker_name} configuration; "
                    "DDL plus partial example rows; " + str(found[0]))
            sample["data_coverage"] = (
                "The displayed INSERT rows/column samples are partial exemplars only: "
                "the linker exposes at most 5 examples and may omit long rows. "
                "Absence from them does not establish absence from the complete database.")
            if not sample.get("context_evidence"):
                sample["context_evidence"] = (
                    "Verified prompt-construction fact: SchemaLinker uses num_insert_rows=5 "
                    "and insert_length_limitation=100, so displayed values are not exhaustive.")
        else:
            sample["schema"] = None
            sample["schema_exposure_status"] = "unknown_original"
            sample["schema_provenance"] = "unavailable" if not found else "ambiguous train/dev database; supply original_schema or --database-path"
    sample.setdefault("data_coverage", "unknown; no full database content evidence")
    return sample


def stratified(rows, seed):
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for item in rows:
        parts = item.get("source", "").replace("\\", "/").split("/")
        stage = next((p for p in parts if p in {"base", "explicit", "implicit"}), "unknown")
        groups[(stage, error_type(item)["code"] or "unknown")].append(item)
    buckets = [groups[k] for k in sorted(groups)]
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    return [bucket[i] for i in range(max(map(len, buckets), default=0)) for bucket in buckets if i < len(bucket)]


def failure_main(args, budget=None):
    budget = budget if budget is not None else {"calls": 0, "visited": 0}
    path = pathlib.Path(args.result_path)
    output = pathlib.Path(args.eval_output_path)
    rows = read_list(path)
    input_keys = {get_key(item) for item in rows}
    if len(input_keys) != len(rows):
        raise ValueError(f"Duplicate attribution identities in input: {path}")
    existing = read_list(output) if output.exists() else []
    if any(r.get("taxonomy_version") != TAXONOMY_VERSION for r in existing):
        raise ValueError("Output contains old taxonomy; choose a new output directory")
    # A rebuilt deterministic pool can replace some sampled rows.  Do not retain
    # stale attributions that are no longer present in the current input file.
    existing = [r for r in existing if r.get("attribution_key") in input_keys]
    normalized_existing = []
    for source in existing:
        record = dict(source)
        record["analysis_disposition"] = analysis_disposition(record)
        normalized_existing.append(record)
    results = {r["attribution_key"]: r for r in normalized_existing}
    evaluator = FailurePatternEvaluator(args)
    for item in stratified(rows, args.seed):
        if args.max_samples and budget["visited"] >= args.max_samples:
            break
        key = get_key(item)
        issue = input_issue(item)
        sample = prepare_sample(item, args) if not issue else item
        model_output = {"reason": item.get("reason"), "sql": item.get("sql")}
        prompt = request_prompt(args, sample, model_output)
        fingerprint = digest([prompt, OUTPUT_SCHEMA, PROMPT_VERSION, args.language_model, args.max_output_tokens])
        semantic_fingerprint = digest([prompt, OUTPUT_SCHEMA, PROMPT_VERSION, args.language_model])
        old = results.get(key, {})
        # A larger output ceiling repairs truncation without invalidating complete
        # results. Recognize legacy records made with the original 2400-token cap.
        matching_request = (
            old.get("attribution_semantic_hash") == semantic_fingerprint
            or old.get("prompt_hash") == fingerprint
            or old.get("prompt_hash") == digest(
                [prompt, OUTPUT_SCHEMA, PROMPT_VERSION, args.language_model, 2400]
            )
        )
        if matching_request and old.get("analysis_status") in {"success", "input_invalid"}:
            continue
        cached_response = (
            old.get("analysis_status") == "error"
            and matching_request
            and isinstance(old.get("raw_attribution_response"), str)
            and old.get("response_finish_reason") == "stop"
            and not getattr(args, "retry_invalid_responses", False)
        )
        if not issue and not cached_response and args.max_calls and budget["calls"] >= args.max_calls:
            break
        budget["visited"] += 1
        record = {**item, "attribution_key": key, "taxonomy_version": TAXONOMY_VERSION,
                  "prompt_version": PROMPT_VERSION, "prompt_hash": fingerprint,
                  "attribution_semantic_hash": semantic_fingerprint,
                  "max_output_tokens": args.max_output_tokens,
                  "attribution_model": args.language_model,
                  "validated_evaluation_used": {k: item.get(k) for k in
                                                ("sql_behavior", "detection", "type_identification")},
                  "attribution_context": {k: sample.get(k) for k in ATTRIBUTION_CONTEXT_FIELDS},
                  "supported_label_ids": [], "supported_labels": [], "novel_candidate": None}
        if issue:
            record.update(analysis_status="input_invalid", evidence_limitations=issue)
        elif args.dry_run:
            record.update(analysis_status="dry_run", attribution_prompt=prompt)
        elif cached_response:
            # Formatting repairs can reuse paid-for content. Keep a failed body
            # locally, never provider headers, credentials, or HTTP diagnostics.
            record.update(usage=old.get("usage"), attempts=old.get("attempts", 1),
                          cached_response_reused=True)
            try:
                record.update(validate_response(old["raw_attribution_response"], model_output, sample),
                              analysis_status="success")
            except Exception as exc:
                record.update(analysis_status="error", error_type=type(exc).__name__,
                              error_stage="response_validation",
                              raw_attribution_response=old["raw_attribution_response"],
                              response_finish_reason=old["response_finish_reason"])
                detail = safe_validation_detail(exc)
                if detail:
                    record["error_detail"] = detail
        else:
            budget["calls"] += 1
            evaluator.last_usage = None
            attempts = 0
            while True:
                attempts += 1
                try:
                    record.update(evaluator(sample, model_output), analysis_status="success")
                    break
                except Exception as exc:
                    transient = type(exc).__name__ in {"APIConnectionError", "APITimeoutError",
                                                       "RateLimitError", "InternalServerError"}
                    if transient and attempts <= args.connection_retries:
                        # Short exponential backoff with jitter; retries count against the API budget.
                        if args.max_calls and budget["calls"] >= args.max_calls:
                            record.update(analysis_status="error", error_type=type(exc).__name__,
                                          error_stage=evaluator.last_error_stage, attempts=attempts)
                            detail = safe_validation_detail(exc)
                            if detail:
                                record["error_detail"] = detail
                            break
                        time.sleep(min(8.0, 1.5 * (2 ** (attempts - 1))) + random.random())
                        budget["calls"] += 1
                        continue
                    # Persist only model content for a local validation failure;
                    # provider error bodies and headers can contain credentials.
                    record.update(analysis_status="error", error_type=type(exc).__name__,
                                  error_stage=evaluator.last_error_stage, attempts=attempts)
                    if evaluator.last_error_stage == "response_validation":
                        record.update(raw_attribution_response=evaluator.last_response,
                                      response_finish_reason=evaluator.last_finish_reason)
                    detail = safe_validation_detail(exc)
                    if detail:
                        record["error_detail"] = detail
                    break
            record["usage"] = evaluator.last_usage
        record["analysis_disposition"] = analysis_disposition(record)
        results[key] = record
        save(output, list(results.values()))
        print(f"[{budget['visited']}] {record['analysis_status']} {record['supported_labels']} candidate={record['novel_candidate'] is not None}")
        if record.get("error_detail"):
            print(json.dumps({"stage": record.get("error_stage"),
                              "error_type": record.get("error_type"),
                              "error_detail": record["error_detail"],
                              "cached_response_reused": record.get("cached_response_reused", False)},
                             ensure_ascii=False))
    successful = [r for r in results.values() if r.get("analysis_status") == "success"]
    status_counts = collections.Counter(r.get("analysis_status", "unknown") for r in results.values())
    disposition_counts = collections.Counter(analysis_disposition(r) for r in results.values())
    unresolved = (disposition_counts["taxonomy_candidate"]
                  + disposition_counts["insufficient_evidence"])
    eligible = disposition_counts["attributed"] + unresolved
    print(json.dumps({"output": str(output), "successful": len(successful),
        "status_counts": dict(status_counts),
        "disposition_counts": dict(disposition_counts),
        "label_counts": dict(collections.Counter(v for r in successful for v in r["supported_labels"])),
        "unlabeled_count": sum(not r["supported_label_ids"] for r in successful),
        "unlabeled_rate": (sum(not r["supported_label_ids"] for r in successful) / len(successful)
                           if successful else None),
        "unresolved_count": unresolved,
        "unresolved_rate": unresolved / eligible if eligible else None,
        "coverage_target_met": unresolved / eligible <= 0.07 if eligible else None,
        "candidates": sum(r["novel_candidate"] is not None for r in successful),
        "note": "Multi-label counts may exceed sample count", "calls_this_run": budget["calls"]}, ensure_ascii=False))
    return list(results.values())


def failure_all(args, target_models=None):
    root = pathlib.Path(args.input_dir) if args.input_dir else ROOT / "failure_analysis/failure" / args.dataset
    paths = [pathlib.Path(args.input)] if args.input else sorted(root.glob("*.json"))
    selected = target_models or args.target_models
    if selected:
        paths = [p for p in paths if p.stem in selected]
    if not paths:
        raise ValueError("No input files matched")
    budget = {"calls": 0, "visited": 0}
    version_tag = PROMPT_VERSION.replace(".", "_")
    output_root = pathlib.Path(args.output_dir) if args.output_dir else ROOT / "failure_analysis" / (
        f"failure_pattern_preview_v{version_tag}" if args.dry_run else f"failure_pattern_result_v{version_tag}")
    print(json.dumps({"stage": "request_policy", "attribution_model": args.language_model,
                      "timeout_seconds": args.timeout,
                      "max_output_tokens": args.max_output_tokens,
                      "connection_retries": args.connection_retries}, ensure_ascii=False))
    for path in paths:
        args.result_path = str(path)
        args.eval_output_path = str(output_root / args.language_model.replace("/", "_") / args.dataset / path.name)
        if pathlib.Path(args.eval_output_path).resolve() == path.resolve():
            raise ValueError("Input and output must differ")
        failure_main(args, budget)
        if args.max_samples and budget["visited"] >= args.max_samples:
            break
        if not args.dry_run and args.max_calls and budget["calls"] >= args.max_calls:
            break


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dataset", choices=["bird", "spider"], default="bird")
    parser.add_argument("-l", "--language_model", default="openai/gpt-5.4")
    parser.add_argument("--input", help="Single JSON input file")
    parser.add_argument("--input-dir", help="Directory of model JSON files")
    parser.add_argument("--output-dir")
    parser.add_argument("--target-models", nargs="+")
    parser.add_argument("--database-path", help="Explicit database root; otherwise resolve uniquely")
    parser.add_argument("--original-schema-linker", default="auto",
                        help="Original linker: auto uses NoGoldSchemaInfo for 3_2, VanillaLinker otherwise; "
                             "missing random 3_2 exposure is kept unknown, never rerandomized")
    parser.add_argument("--max-calls", type=int, default=0,
                        help="Optional global API-attempt cap including retries; 0=unlimited (default)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Optional global newly processed sample cap including invalid inputs; 0=all samples (default)")
    parser.add_argument("--max-output-tokens", type=int,
                        help="Output ceiling; default is 4096 for Claude and 2400 otherwise")
    parser.add_argument("--timeout", type=float,
                        help="Per-request timeout in seconds; default is 600 for gpt-6.1-sol and 120 otherwise")
    parser.add_argument("--connection-retries", type=int,
                        help="Transient retries after the first attempt (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Save prompts without loading credentials or calling API")
    parser.add_argument("--retry-invalid-responses", action="store_true",
                        help="Request fresh model content instead of locally replaying saved invalid JSON")
    args = parser.parse_args()
    slow_attribution_model = args.language_model.casefold().endswith("gpt-6.1-sol")
    if args.max_output_tokens is None:
        args.max_output_tokens = 4096 if args.language_model.casefold().startswith("claude/") else 2400
    if args.timeout is None:
        args.timeout = 600.0 if slow_attribution_model else 120.0
    if args.connection_retries is None:
        args.connection_retries = 2
    if (min(args.max_calls, args.max_samples, args.connection_retries) < 0
            or args.max_output_tokens < 1 or args.timeout <= 0):
        parser.error("Invalid sample/call/token/timeout limits")
    if args.dry_run and args.max_samples == 0:
        parser.error("Set --max-samples for a bounded preview")
    return args
