import argparse
import ast
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"

JUDGE_1 = "claude_claude-opus-4-6"
JUDGE_2 = "openai_gpt-5.4"

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "failure"
DEFAULT_EXCLUSION_ROOT = Path(__file__).resolve().parent / "failure_exclusions"


def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected JSON list: {path}"
        )

    return data



def attribution_issue(item):
    """Return why an output cannot support mechanism attribution, else None."""
    reason = item.get("reason")
    sql = item.get("sql")
    if not isinstance(reason, str):
        return "reason_non_string"

    def is_schema_placeholder(value):
        if isinstance(value, dict) and "type" in value:
            return True
        if not isinstance(value, str):
            return False
        try:
            parsed = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            return False
        return isinstance(parsed, dict) and "type" in parsed

    if is_schema_placeholder(reason):
        return "reason_schema_placeholder"
    if is_schema_placeholder(sql):
        return "sql_schema_placeholder"
    if not reason.strip() and (sql is None or not str(sql).strip()):
        return "reason_and_sql_empty"
    return None

def get_key(item):

    if item.get("sample_id") is not None:
        return (
            "sample_id",
            item["sample_id"]
        )

    return (
        "db_question",
        item.get("db_id", ""),
        item.get("question", "")
    )


def get_json_files(folder):

    if not os.path.isdir(folder):
        return []

    return sorted(
        f
        for f in os.listdir(folder)
        if f.lower().endswith(".json")
        and os.path.isfile(
            os.path.join(folder, f)
        )
    )


def process_file(
    file1,
    file2,
    file_name,
    dataset_name,
    model_name,
    stage
):

    data1 = load_json(file1)
    data2 = load_json(file2)

    # A failure includes wrong type identification OR failure to abstain.
    # Invalid judge scores are not evidence of model failure.
    def is_failure(item):
        valid = (item.get("type_identification") in (0, 1)
                 and item.get("detection") in (0, 1)
                 and item.get("sql_behavior") in (0, 1, 2))
        return valid and (item["type_identification"] == 0 or item["sql_behavior"] != 2)

    score_fields = ("detection", "type_identification", "sql_behavior")

    judge2_map = {
        get_key(item): item
        for item in data2
        if is_failure(item)
    }

    results = []
    exclusions = []

    for item1 in data1:

        if not is_failure(item1):
            continue

        key = get_key(item1)

        # 判断 key 是否在 Judge2 中完全匹配且满足条件
        item2 = judge2_map.get(key)

        if item2 is None:
            continue

        # Matching IDs alone must not pair different model outputs.
        if any(item1.get(k) != item2.get(k) for k in ("db_id", "question", "reason", "sql")):
            continue

        result = dict(item1)
        result["selection_policy"] = "both_judges_exact_metric_agreement_v4"
        result["judge_scores"] = {
            JUDGE_1: {k: item1.get(k) for k in score_fields},
            JUDGE_2: {k: item2.get(k) for k in score_fields},
        }
        result["judge_disagreement"] = any(
            item1.get(k) != item2.get(k) for k in score_fields
        )

        # 保存文件路径到 source 字段
        result["source"] = os.path.normpath(
            os.path.join(
                dataset_name,
                model_name,
                stage,
                "eval",
                JUDGE_1,
                file_name
            )
        )

        if result["judge_disagreement"]:
            result["attribution_exclusion_reason"] = "judge_metric_disagreement"
            exclusions.append(result)
            continue

        issue = attribution_issue(result)
        if issue:
            excluded = dict(result)
            excluded["attribution_exclusion_reason"] = issue
            exclusions.append(excluded)
            continue

        results.append(result)

    return (
        results,
        exclusions,
        len(data1),
        len(data2)
    )


def process_model(
    model_dir,
    dataset_name,
    output_dir,
    exclusion_root,
):

    model_name = os.path.basename(
        model_dir
    )

    all_results = []
    all_exclusions = []

    total_files = 0
    total_samples = 0

    print("\n" + "=" * 100)
    print(f"DATASET: {dataset_name}")
    print(f"MODEL:   {model_name}")
    print("=" * 100)

    stages = sorted(
        os.listdir(model_dir)
    )

    for stage in stages:

        stage_dir = os.path.join(
            model_dir,
            stage
        )

        if not os.path.isdir(stage_dir):
            continue

        eval_dir = os.path.join(
            stage_dir,
            "eval"
        )

        if not os.path.isdir(eval_dir):
            print(
                f"[SKIP] {stage}: "
                f"no eval directory"
            )
            continue

        judge1_dir = os.path.join(
            eval_dir,
            JUDGE_1
        )

        judge2_dir = os.path.join(
            eval_dir,
            JUDGE_2
        )

        if not os.path.isdir(judge1_dir):

            print(
                f"[SKIP] {stage}: "
                f"missing {JUDGE_1}"
            )

            continue

        if not os.path.isdir(judge2_dir):

            print(
                f"[SKIP] {stage}: "
                f"missing {JUDGE_2}"
            )

            continue

        files1 = set(
            get_json_files(judge1_dir)
        )

        files2 = set(
            get_json_files(judge2_dir)
        )

        common_files = sorted(
            files1 & files2
        )

        print(
            f"[STAGE] {stage}: "
            f"Judge1 files={len(files1)}, "
            f"Judge2 files={len(files2)}, "
            f"common={len(common_files)}"
        )

        if not common_files:
            continue

        for file_name in common_files:

            file1 = os.path.join(
                judge1_dir,
                file_name
            )

            file2 = os.path.join(
                judge2_dir,
                file_name
            )

            # 此处补齐了丢失的 model_name 和 stage 参数
            results, exclusions, n1, n2 = process_file(
                file1,
                file2,
                file_name,
                dataset_name,
                model_name,
                stage
            )

            all_results.extend(
                results
            )
            all_exclusions.extend(exclusions)

            total_files += 1
            total_samples += len(results)

            print(
                f"    {file_name}: "
                f"Judge1={n1}, "
                f"Judge2={n2}, "
                f"Attributable failure={len(results)}, "
                f"excluded={len(exclusions)}"
            )

    # ========================================================
    # 输出
    # ========================================================

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        f"{model_name}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=4
        )

    exclusion_dir = os.path.join(exclusion_root, dataset_name)
    os.makedirs(exclusion_dir, exist_ok=True)
    exclusion_file = os.path.join(exclusion_dir, f"{model_name}.json")
    with open(exclusion_file, "w", encoding="utf-8") as f:
        json.dump(all_exclusions, f, ensure_ascii=False, indent=4)

    print("-" * 100)
    print(
        f"Paired files : {total_files}"
    )
    print(
        f"Attributable : {total_samples}"
    )
    print(
        f"Excluded     : {len(all_exclusions)}"
    )
    print(
        f"Output       : {output_file}"
    )
    print(
        f"Exclusion log: {exclusion_file}"
    )


def process_dataset(root, output_root, exclusion_root):

    dataset_name = os.path.basename(
        root
    )

    output_dir = os.path.join(
        output_root,
        dataset_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print("\n\n")
    print("#" * 100)
    print(
        f"DATASET: {dataset_name}"
    )
    print(
        f"ROOT   : {root}"
    )
    print(
        f"OUTPUT : {output_dir}"
    )
    print("#" * 100)

    model_names = sorted(
        os.listdir(root)
    )

    print(
        f"Found {len(model_names)} entries."
    )

    for model_name in model_names:

        model_dir = os.path.join(
            root,
            model_name
        )

        if not os.path.isdir(model_dir):
            continue

        print(
            f"\n[MODEL] {model_name}"
        )

        process_model(
            model_dir,
            dataset_name,
            output_dir,
            exclusion_root,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Build the strict two-judge consensus failure pool."
    )
    parser.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help="Directory containing one result directory per dataset.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=("bird", "spider"),
        help="Datasets to rebuild.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Destination for attributable failures.",
    )
    parser.add_argument(
        "--exclusion-root", type=Path, default=DEFAULT_EXCLUSION_ROOT,
        help="Destination for disagreement and malformed-output audit rows.",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        root = args.results_root / dataset

        if not os.path.isdir(root):

            print(
                f"[WARNING] "
                f"Root does not exist: {root}"
            )

            continue

        process_dataset(
            os.fspath(root),
            os.fspath(args.output_root),
            os.fspath(args.exclusion_root),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
