import json
import copy
from pathlib import Path
from tqdm import tqdm
from utils import (ROOT, get_eval_args, get_prompt_builder_from_filename,
                   resolve_jobs, load_dataset)


def build_judge_sample(item, original, reference_schema):
    """Retain complete judge reference and distinguish generation exposure."""
    return {
        "db_id": item.get("db_id"), "question": item.get("question"),
        "schema": reference_schema,
        "evidence": item.get("evidence", original.get("evidence", "")),
        "original_schema": item.get("original_schema"),
        "original_schema_linker": item.get("original_schema_linker"),
        "original_evidence_provenance": (
            "saved generation input" if "evidence" in item else "matched source dataset"),
    }


def eval_main(args):
    import schema_linking
    from dotenv import load_dotenv
    from eval.eval import Evaluator

    load_dotenv(ROOT / ".env")
    with open(args.result_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    originals = {}
    for original in load_dataset(args.data_path):
        key = (original.get("db_id"), original.get("question"))
        if key in originals:
            raise ValueError(f"Duplicate original sample key: {key}")
        originals[key] = original
    # The evaluator always receives the complete database schema.  Schema
    # corruption such as the 3_2 column removal belongs only to the tested
    # model's generation input and must not be applied again to the judge.
    schema_linker = schema_linking.load_schema_linker(
        "VanillaLinker",
        gold_schema_path=str(ROOT / "data" / args.dataset / "dev_gold_schema.json"),
    )
    evaluator = Evaluator(args, args.prompt_builder)

    results = []

    total = 0
    sum_sql_behavior = 0
    sum_detection = 0
    sum_type = 0

    def safe_get(d, key):
        return 0 if d.get(key) is None else d.get(key)

    for item in tqdm(dataset):
        db_name = item.get("db_id")
        question = item.get("question")

        schema = schema_linker.get_schema(args.database_path, db_name, question, None)

        original = originals.get((db_name, question))
        if original is None:
            raise ValueError(f"Original judge context not found: {(db_name, question)}")
        sample = build_judge_sample(item, original, schema)
        sample["original_schema_linker"] = sample["original_schema_linker"] or args.schema_linker

        model_output = {
            "sql": item.get("sql"),
            "reason": item.get("reason")
        }

        try:
            eval_result = evaluator(sample, model_output)
        except Exception as e:
            print(f"[ERROR] eval failed: {e}")
            eval_result = {
                "sql_behavior": 0,
                "detection": 0,
                "type_identification": 0
            }

        merged = {**item, **eval_result,
                  "evidence": sample["evidence"],
                  "original_evidence_provenance": sample["original_evidence_provenance"],
                  "evaluation_context_policy": "original_knowledge_and_reference_schema_v1"}
        results.append(merged)

        total += 1
        sum_sql_behavior += safe_get(eval_result, "sql_behavior")
        sum_detection += safe_get(eval_result, "detection")
        sum_type += safe_get(eval_result, "type_identification")

    Path(args.eval_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.eval_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("\n===== Evaluation Summary =====")
    if total > 0:
        print(f"Total samples: {total}")
        print(f"SQL Behavior Avg: {sum_sql_behavior / total:.4f}")
        print(f"Detection Acc: {sum_detection / total:.4f}")
        print(f"Type Identification Acc: {sum_type / total:.4f}")


def eval_all(options):
    # resolve_jobs is the single source of truth for split, input file,
    # database root, schema linker, and output filename selection.
    templates = list(resolve_jobs(options))
    result_root = ROOT / ("validity" if templates[0].split == "valid" else "results") / templates[0].dataset
    requested = ({name.replace("/", "_") for name in options.target_models}
                 if options.target_models else None)
    model_names = sorted(
        path.name for path in result_root.iterdir()
        if path.is_dir() and (requested is None or path.name in requested)
    ) if result_root.is_dir() else []
    if requested:
        missing = requested - set(model_names)
        if missing:
            raise ValueError(f"Target model result directories not found: {sorted(missing)}")
    if not model_names:
        raise ValueError(f"No model result directories found in {result_root}")

    judge_name = options.language_model.replace("/", "_")
    print(f"\n===== DATASET: {templates[0].dataset} | SPLIT: {templates[0].split} "
          f"| MODE: {options.mode} | JUDGE: {options.language_model} =====\n")

    for model_name in model_names:
        model_dir = result_root / model_name / options.mode
        if not model_dir.is_dir():
            continue
        print(f"\n====== MODEL: {model_name} ======")

        for template in templates:
            filename = Path(template.result_path).name
            result_path = model_dir / filename
            eval_output_path = model_dir / "eval" / judge_name / filename
            job = {
                "dataset": template.dataset,
                "split": template.split,
                "target_model": model_name,
                "judge_model": options.language_model,
                "data_path": template.data_path,
                "database_path": template.database_path,
                "schema_linker": "VanillaLinker",
                "result_path": str(result_path),
                "eval_output_path": str(eval_output_path),
                "result_exists": result_path.is_file(),
                "eval_exists": eval_output_path.is_file(),
            }
            print(json.dumps(job, ensure_ascii=False))
            if options.dry_run:
                continue
            if not result_path.is_file():
                print(f"[SKIP MISSING] {result_path}")
                continue
            if eval_output_path.exists() and not options.overwrite:
                print(f"[SKIP DONE] {eval_output_path}")
                continue

            cur_args = copy.copy(template)
            cur_args.language_model = options.language_model
            cur_args.result_path = str(result_path)
            cur_args.eval_output_path = str(eval_output_path)
            cur_args.prompt_builder = get_prompt_builder_from_filename(filename)
            eval_main(cur_args)
            print(f"输出完成: {cur_args.eval_output_path}")


if __name__ == "__main__":
    eval_all(get_eval_args())
