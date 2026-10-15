"""Recompute premise-error metrics from the current evaluator outputs.

Each generation record contributes once. For judge-dependent metrics, the two
evaluator labels are averaged on records aligned with the current raw output.
SQL Behavior is normalized from {0, 1, 2} to [0, 1].
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from utils import get_invalid_evaluation_metrics


ROOT = Path(__file__).resolve().parent
DATASETS = ("bird", "spider")
MODES = ("base", "implicit", "explicit")
JUDGES = ("claude_claude-opus-4-6", "openai_gpt-5.4")
CATEGORIES = {
    "1_1": "Ambiguity",
    "1_2": "Unclear Intent",
    "2_1": "Text Contradiction",
    "2_2": "Schema Contradiction",
    "3_1": "Missing Constraint",
    "3_2": "Missing Schema",
    "3_3": "Semantic Gap",
    "4_1": "Inexpressible SQL",
    "4_2": "External Knowledge",
}
CATEGORY_GROUPS = {
    "E1": "Expression Error",
    "E2": "Premise Contradiction",
    "E3": "Missing Premise",
    "E4": "Beyond Ability",
}
METRICS = ("sql_behavior", "detection", "type_identification")


def load_json(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


def index_records(records, path):
    """Align duplicate db/question pairs by their occurrence within a file."""
    occurrences = Counter()
    indexed = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Non-object record at {path}:{position}")
        db_id, question = record.get("db_id"), record.get("question")
        if not isinstance(db_id, str) or not isinstance(question, str):
            raise ValueError(f"Missing record identity at {path}:{position}")
        identity = (db_id, question)
        key = (*identity, occurrences[identity])
        occurrences[identity] += 1
        if key in indexed:
            raise ValueError(f"Duplicate indexed identity at {path}:{position}")
        indexed[key] = record
    return indexed


def empty_bucket():
    return {"records": 0, **{f"sum_{metric}": 0.0 for metric in METRICS}}


def add_record(bucket, labels):
    bucket["records"] += 1
    for metric in METRICS:
        scale = 2 if metric == "sql_behavior" else 1
        bucket[f"sum_{metric}"] += sum(label[metric] / scale for label in labels) / len(labels)


def export_bucket(bucket):
    denominator = bucket["records"]
    return {
        "records": denominator,
        "SQL": bucket["sum_sql_behavior"] / denominator if denominator else None,
        "DET": bucket["sum_detection"] / denominator if denominator else None,
        "TYPE": bucket["sum_type_identification"] / denominator if denominator else None,
    }


def analyze(root=ROOT, judges=JUDGES):
    category_buckets = defaultdict(empty_bucket)
    category_group_buckets = defaultdict(empty_bucket)
    mode_buckets = defaultdict(empty_bucket)
    coverage = Counter()
    issues = []
    models_by_dataset = {}

    for dataset in DATASETS:
        dataset_root = root / "results" / dataset
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Missing result directory: {dataset_root}")
        models = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
        models_by_dataset[dataset] = models
        for model in models:
            for mode in MODES:
                mode_root = dataset_root / model / mode
                for category_id, category_name in CATEGORIES.items():
                    raw_path = mode_root / f"final_{category_id}.json"
                    if not raw_path.is_file():
                        issues.append({"path": str(raw_path), "problem": "missing raw file"})
                        continue
                    coverage["raw_files"] += 1
                    raw = index_records(load_json(raw_path), raw_path)
                    coverage["raw_records"] += len(raw)
                    judge_records = {}
                    for judge in judges:
                        judge_path = mode_root / "eval" / judge / raw_path.name
                        if not judge_path.is_file():
                            issues.append({"path": str(judge_path), "problem": "missing judge file"})
                            judge_records[judge] = {}
                            continue
                        coverage["judge_files"] += 1
                        evaluated = index_records(load_json(judge_path), judge_path)
                        coverage["judge_records"] += len(evaluated)
                        extra = set(evaluated) - set(raw)
                        missing = set(raw) - set(evaluated)
                        coverage["extra_judge_records"] += len(extra)
                        coverage["missing_judge_records"] += len(missing)
                        current = {}
                        for key in set(raw) & set(evaluated):
                            stale_fields = [
                                field for field in ("sql", "reason", "is_valid")
                                if raw[key].get(field) != evaluated[key].get(field)
                            ]
                            if stale_fields:
                                coverage["stale_judge_records"] += 1
                                issues.append({
                                    "path": str(judge_path),
                                    "problem": "judge copy differs from raw output",
                                    "fields": stale_fields,
                                    "db_id": key[0],
                                    "question": key[1],
                                    "occurrence": key[2],
                                })
                            else:
                                current[key] = evaluated[key]
                        judge_records[judge] = current

                    for key in raw:
                        labels = [judge_records[judge].get(key) for judge in judges]
                        if any(label is None for label in labels):
                            coverage["records_excluded_incomplete_judges"] += 1
                            continue
                        invalid = [get_invalid_evaluation_metrics(label) for label in labels]
                        if any(invalid):
                            coverage["records_excluded_invalid_metrics"] += 1
                            issues.append({
                                "path": str(raw_path),
                                "problem": "invalid evaluator metrics",
                                "db_id": key[0],
                                "question": key[1],
                                "occurrence": key[2],
                                "judge_issues": invalid,
                            })
                            continue
                        add_record(mode_buckets[mode], labels)
                        if mode == "implicit":
                            add_record(category_buckets[category_name], labels)
                            add_record(category_group_buckets[f"E{category_id[0]}"], labels)

    expected_models = sorted(set(models_by_dataset[DATASETS[0]]))
    if any(sorted(models) != expected_models for models in models_by_dataset.values()):
        issues.append({"problem": "model sets differ across datasets", "models": models_by_dataset})

    category_results = {
        name: export_bucket(category_buckets[name]) for name in CATEGORIES.values()
    }
    category_group_results = {
        code: {"category": name, **export_bucket(category_group_buckets[code])}
        for code, name in CATEGORY_GROUPS.items()
    }
    mode_results = {mode: export_bucket(mode_buckets[mode]) for mode in MODES}
    return {
        "scope": {
            "datasets": list(DATASETS),
            "modes": list(MODES),
            "judges": list(judges),
            "models_by_dataset": models_by_dataset,
            "category_mapping": CATEGORIES,
        },
        "definitions": {
            "unit": "One aligned generation record; the two judge labels are averaged before aggregation.",
            "SQL": "Mean SQL Behavior divided by 2, yielding a [0,1] score.",
            "DET": "Mean binary Detection label.",
            "TYPE": "Mean binary Type Identification label.",
        },
        "coverage": dict(coverage),
        "issues": issues,
        "aggregate_by_mode": mode_results,
        "implicit_by_category": category_group_results,
        "implicit_by_error_type": category_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--judges", nargs=2, default=list(JUDGES))
    parser.add_argument("--output", type=Path, default=Path("output/error_type_metrics.json"))
    args = parser.parse_args()
    report = analyze(args.root, args.judges)
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Aggregate by mode")
    for mode, values in report["aggregate_by_mode"].items():
        print(mode, values)
    print("\nImplicit by error type")
    for category, values in report["implicit_by_error_type"].items():
        print(category, values)
    print("\nImplicit by category")
    for code, values in report["implicit_by_category"].items():
        print(code, values)
    print("\nCoverage", report["coverage"])
    print("Issues", len(report["issues"]))
    print("Report", output)


if __name__ == "__main__":
    main()
