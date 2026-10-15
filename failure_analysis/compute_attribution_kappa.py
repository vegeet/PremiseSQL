"""Compute agreement on paired, successful multi-label attribution records."""

import argparse
import json
from collections import Counter
from pathlib import Path


LABELS = {
    "M01": "assumption_injection",
    "M02": "contradiction_reinterpretation",
    "M03": "context_evidence_misuse",
    "M04": "explicit_constraint_omission",
}
ROOT = Path(__file__).resolve().parent


def collect(directory):
    records = {}
    for path in sorted(directory.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for record in data:
            key = record["attribution_key"]
            if key in records:
                raise ValueError(f"Duplicate attribution key in {path}: {key}")
            records[key] = (record, path.relative_to(directory).parts[0])
    return records


def agreement(pairs):
    n = len(pairs)
    if not n:
        return {"n": 0, "kappa": None}
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    matches = sum(a == b for a, b in pairs)
    observed = matches / n
    expected = sum(left[x] * right[x] for x in left) / n**2
    return {
        "n": n,
        "matches": matches,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": (observed - expected) / (1 - expected) if expected != 1 else None,
        "judge1_counts": dict(sorted(left.items())),
        "judge2_counts": dict(sorted(right.items())),
    }


def summarize(pairs):
    per_label = {}
    pooled = []
    for label, name in LABELS.items():
        binary = [(int(label in a["supported_label_ids"]),
                   int(label in b["supported_label_ids"])) for a, b, _ in pairs]
        per_label[label] = {"name": name, **agreement(binary)}
        pooled.extend(binary)
    label_sets = [("+".join(sorted(a["supported_label_ids"])) or "NONE",
                   "+".join(sorted(b["supported_label_ids"])) or "NONE")
                  for a, b, _ in pairs]
    defined_kappas = [v["kappa"] for v in per_label.values() if v["kappa"] is not None]
    return {
        "paired_samples": len(pairs),
        "per_label_binary": per_label,
        "pooled_binary": agreement(pooled),
        "macro_mean_binary_kappa": sum(defined_kappas) / len(defined_kappas) if defined_kappas else None,
        "macro_defined_label_count": len(defined_kappas),
        "label_set_as_nominal_category": agreement(label_sets),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "failure_pattern_result_full_v3_8_20261007")
    parser.add_argument("--judges", nargs=2, default=["claude_claude-opus-5-5", "openai_gpt-6.1-sol"])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    left, right = [collect(args.root / judge) for judge in args.judges]
    shared = set(left) & set(right)
    pairs = []
    excluded = Counter()
    for key in sorted(shared):
        a, dataset_a = left[key]
        b, dataset_b = right[key]
        if a.get("analysis_status") != "success" or b.get("analysis_status") != "success":
            excluded[f'{a.get("analysis_status")} / {b.get("analysis_status")}'] += 1
            continue
        identity = ("source", "sample_id", "db_id", "question", "sql", "reason",
                    "taxonomy_version", "prompt_version", "evidence", "attribution_context")
        if dataset_a != dataset_b or any(a.get(field) != b.get(field) for field in identity):
            raise ValueError(f"Paired inputs differ: {key}")
        for record in (a, b):
            assessments = record["label_assessments"]
            if Counter(x["label_id"] for x in assessments) != Counter(LABELS.keys()):
                raise ValueError(f"Invalid assessment labels: {key}")
            supported = {x["label_id"] for x in assessments if x["verdict"] == "supported"}
            if supported != set(record["supported_label_ids"]):
                raise ValueError(f"Supported label mismatch: {key}")
        pairs.append((a, b, dataset_a))
    report = {
        "input_root": str(args.root.resolve()),
        "judges": args.judges,
        "method": "Unweighted Cohen kappa; each M01-M04 is binary supported vs not supported (including uncertain). Only both-success records are used. Pooled binary concatenates all four decisions; macro is the mean of per-label kappas. Label-set kappa treats each exact combination as one nominal category.",
        "judge_record_counts": [len(left), len(right)],
        "common_keys": len(shared),
        "unmatched_counts": [len(set(left) - shared), len(set(right) - shared)],
        "excluded_status_pairs": dict(excluded),
        "overall": summarize(pairs),
        "by_dataset": {dataset: summarize([p for p in pairs if p[2] == dataset])
                       for dataset in sorted({p[2] for p in pairs})},
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
