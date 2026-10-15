"""Build a deterministic, compact attribution pool from washed failures."""
import argparse
import difflib
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def stable_rank(item, seed):
    identity = [
        item.get("source"), item.get("sample_id"), item.get("db_id"),
        item.get("question"), item.get("sql"), item.get("reason"), seed,
    ]
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def stratum(item):
    parts = item.get("source", "").replace("\\", "/").split("/")
    stage = next((part for part in parts if part in {"base", "explicit", "implicit"}),
                 "unknown")
    error_type = Path(item.get("source", "")).stem.removeprefix("final_")
    return stage, error_type


def original_index(data_root, dataset):
    """Index the flawed-question inputs that supplied the tested model context."""
    folder = data_root / dataset / f"{dataset}_e"
    index = {}
    for path in sorted(folder.glob("*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            key = (path.name, item.get("db_id"), item.get("question"))
            if key in index:
                raise ValueError(f"Duplicate original sample key: {key}")
            index[key] = item
    return index


def valid_reference_index(data_root, dataset, originals):
    """Recover the paired valid query used before premise perturbation.

    The gold SQL and database ID are preserved during benchmark construction.  A
    similarity tie is left unresolved instead of attaching a possibly wrong pair.
    """
    candidates = defaultdict(list)
    folder = data_root / dataset / "data_valid"
    for path in sorted(folder.glob("*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            candidates[(item.get("db_id"), item.get("sql"))].append((path.name, item))

    references = {}
    for original_key, original in originals.items():
        options = candidates.get((original.get("db_id"), original.get("sql")), [])
        if not options:
            continue
        ranked = list(
            (
                difflib.SequenceMatcher(
                    None, original.get("question", ""), option.get("question", "")
                ).ratio(),
                path_name,
                option,
            )
            for path_name, option in options
        )
        ranked.sort(
            key=lambda entry: (
                entry[0], entry[1], entry[2].get("question", "")
            ),
            reverse=True,
        )
        best_score, path_name, best = ranked[0]
        if (len(ranked) > 1 and ranked[1][0] == best_score
                and ranked[1][2].get("question") != best.get("question")):
            continue
        references[original_key] = {
            "paired_valid_question": best.get("question"),
            "paired_valid_sql": best.get("sql"),
            "paired_valid_evidence": best.get("evidence", ""),
            "paired_valid_provenance": f"{dataset}/data_valid/{path_name}",
            "paired_valid_similarity": round(best_score, 6),
        }
    return references


def enrich_and_filter(rows, originals, references):
    """Restore original evidence and remove cases outside mechanism attribution."""
    kept = []
    excluded = defaultdict(int)
    for source in rows:
        item = dict(source)
        source_name = Path(item.get("source", "")).name
        key = (source_name, item.get("db_id"), item.get("question"))
        original = originals.get(key)
        if original is None:
            excluded["original_context_not_recoverable"] += 1
            continue
        error_code = Path(source_name).stem.removeprefix("final_")
        # Missing-Schema-Premise examples used a random NoGoldSchemaInfo view.
        # Without the recorded historical exposure, a reference database cannot
        # establish whether a generated field or relation was actually available.
        if error_code == "3_2" and item.get("original_schema") is None:
            excluded["missing_original_schema_exposure"] += 1
            continue
        reference = references.get(key)
        if reference is None:
            excluded["paired_valid_reference_not_recoverable"] += 1
            continue
        # Empty string is known absence of extra knowledge, not missing context.
        item["evidence"] = source.get("evidence", original.get("evidence", ""))
        item["original_evidence_provenance"] = source.get(
            "original_evidence_provenance",
            "saved generation input" if "evidence" in source else str(source_name))
        item.update(reference)

        behavior = (item.get("detection"), item.get("type_identification"),
                    item.get("sql_behavior"))
        if behavior in {(1, 1, 0), (1, 1, 1)}:
            excluded["correctly_identified_but_answered"] += 1
            continue
        kept.append(item)
    return kept, dict(excluded)


def allocate(group_sizes, quota):
    """Proportional allocation with one item for every non-empty stratum."""
    total = sum(group_sizes.values())
    if quota >= total:
        return dict(group_sizes)
    if quota < len(group_sizes):
        raise ValueError(
            f"Quota {quota} is smaller than {len(group_sizes)} non-empty strata"
        )

    allocation = {key: 1 for key in group_sizes}
    remaining = quota - len(allocation)
    capacities = {key: size - 1 for key, size in group_sizes.items()}
    capacity_total = sum(capacities.values())
    if remaining == 0 or capacity_total == 0:
        return allocation

    ideals = {key: remaining * capacity / capacity_total
              for key, capacity in capacities.items()}
    for key, ideal in ideals.items():
        allocation[key] += min(capacities[key], math.floor(ideal))

    left = quota - sum(allocation.values())
    order = sorted(
        group_sizes,
        key=lambda key: (ideals[key] - math.floor(ideals[key]), group_sizes[key], key),
        reverse=True,
    )
    for key in order:
        if left == 0:
            break
        if allocation[key] < group_sizes[key]:
            allocation[key] += 1
            left -= 1
    if left:
        raise RuntimeError("Could not allocate the requested sample quota")
    return allocation


def sample_rows(rows, quota, seed=42):
    groups = defaultdict(list)
    for item in rows:
        groups[stratum(item)].append(item)
    allocation = allocate({key: len(items) for key, items in groups.items()}, quota)
    selected = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda item: stable_rank(item, seed))
        selected.extend(ranked[:allocation[key]])
    return sorted(selected, key=lambda item: stable_rank(item, seed)), allocation


def build_sampled_pool(input_root, output_root, per_model=250, seed=42,
                       data_root=None):
    data_root = data_root or ROOT.parent / "data"
    manifest = {"seed": seed, "per_model": per_model, "datasets": {}}
    for dataset_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        output_dir = output_root / dataset_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_manifest = {}
        originals = original_index(data_root, dataset_dir.name)
        references = valid_reference_index(data_root, dataset_dir.name, originals)
        for source_file in sorted(dataset_dir.glob("*.json")):
            rows = json.loads(source_file.read_text(encoding="utf-8"))
            eligible, excluded = enrich_and_filter(rows, originals, references)
            selected, allocation = sample_rows(
                eligible, min(per_model, len(eligible)), seed
            )
            (output_dir / source_file.name).write_text(
                json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            dataset_manifest[source_file.stem] = {
                "source_count": len(rows),
                "eligible_count": len(eligible),
                "sample_count": len(selected),
                "excluded": excluded,
                "strata": {f"{stage}/{kind}": count
                            for (stage, kind), count in sorted(allocation.items())},
            }
        manifest["datasets"][dataset_dir.name] = dataset_manifest
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "failure")
    parser.add_argument("--output-root", type=Path, default=ROOT / "failure_sampled_250")
    parser.add_argument("--per-model", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    args = parser.parse_args()
    if args.per_model < 1:
        parser.error("--per-model must be positive")
    manifest = build_sampled_pool(args.input_root, args.output_root,
                                  args.per_model, args.seed, args.data_root)
    counts = {
        dataset: sum(model["sample_count"] for model in models.values())
        for dataset, models in manifest["datasets"].items()
    }
    print(json.dumps({"output": str(args.output_root), "sample_counts": counts,
                      "total": sum(counts.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
