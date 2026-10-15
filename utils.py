import os
import json
import argparse
from pathlib import Path
from types import SimpleNamespace
import eval.build_prompt as prompt_module
import re


def seed_everything(seed: int):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_result_path(args, filename):
    model_name = args.language_model.replace("/", "_")
    output_name = f"{Path(filename).stem}_valid.json"
    return str(ROOT / "validity" / args.dataset / model_name / args.mode / output_name)


def get_prompt_builder_from_filename(filename):
    match = re.search(r'(\d+_\d+)', filename)
    # match = re.search(r'final_(\d+_\d+)', filename)
    name = match.group(1)
    func_name = f"build_eval_prompt_{name}"
    if not hasattr(prompt_module, func_name):
        raise ValueError(f"{func_name} not found in build_prompt.py")
    return getattr(prompt_module, func_name)


def load_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_output_file(output_path, mode='w'):
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)

    return open(output_path, mode, encoding='utf8')


def remove_sql_comments(sql):
    """
    Remove all SQL comments (both inline and full-line) from a SQL string.
    
    Args:
        sql (str): Input SQL string potentially containing comments.
        
    Returns:
        str: SQL string with all comments removed while preserving string literals.
        
    Handles:
    - Both single (') and double (") quoted strings
    - Escaped quotes within strings (e.g. 'It''s valid')
    - Comments appearing anywhere in the SQL
    - Preservation of newlines (to be replaced later)
    """
    result = []
    in_string = None  # Track string state: None, 'single', or 'double'
    i = 0
    n = len(sql)

    while i < n:
        char = sql[i]

        if in_string:
            # Handle quoted string content
            if (in_string == 'single' and char == "'") or (in_string == 'double' and char == '"'):
                # Check for escaped quote (SQL-standard '' or "")
                if i + 1 < n and sql[i + 1] == char:
                    result.extend([char, char])
                    i += 2  # Skip escaped quote
                else:
                    in_string = None
                    result.append(char)
                    i += 1
            else:
                result.append(char)
                i += 1
        else:
            # Handle comment detection
            if char == '-' and i + 1 < n and sql[i + 1] == '-':
                # Skip all characters until end of line
                while i < n and sql[i] not in ('\n', '\r'):
                    i += 1
            elif char in ("'", '"'):
                in_string = 'single' if char == "'" else 'double'
                result.append(char)
                i += 1
            else:
                result.append(char)
                i += 1

    return ''.join(result)


def _argument_parser(description=None, *, automatic_schema=False):
    """Shared experiment options; generation resolves its schema linker per file."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-m', '--method', default='Base')
    parser.add_argument('-d', '--dataset', default='bird',
                        **({'choices': ['bird', 'spider']} if automatic_schema else {}))
    parser.add_argument('-l', '--language_model', default='openai/gpt-5.4')
    parser.add_argument('--mode', default='base', choices=['implicit', 'base', 'explicit'])
    if not automatic_schema:
        parser.add_argument('-s', '--schema_linker', default='VanillaLinker')
    return parser


ROOT = Path(__file__).resolve().parent

EVALUATION_METRIC_VALUES = {
    "sql_behavior": {0, 1, 2},
    "detection": {0, 1},
    "type_identification": {0, 1},
}


def get_invalid_evaluation_metrics(record):
    """Describe missing, malformed, or logically inconsistent eval metrics."""
    if not isinstance(record, dict):
        return {
            "_record": {
                "value": record,
                "reason": f"record must be an object, got {type(record).__name__}",
            }
        }

    issues = {}
    for field, allowed in EVALUATION_METRIC_VALUES.items():
        if field not in record:
            issues[field] = {"value": None, "reason": "missing field"}
            continue
        value = record[field]
        if value is None:
            issues[field] = {"value": None, "reason": "null value"}
        elif type(value) is not int:
            issues[field] = {
                "value": value,
                "reason": f"expected integer, got {type(value).__name__}",
            }
        elif value not in allowed:
            issues[field] = {
                "value": value,
                "reason": f"value must be one of {sorted(allowed)}",
            }

    if (
        record.get("detection") == 0
        and record.get("type_identification") == 1
        and "type_identification" not in issues
    ):
        issues["type_identification"] = {
            "value": 1,
            "reason": "must be 0 when detection is 0",
        }
    return issues


def get_run_args(argv=None):
    parser = _argument_parser("Run valid or flawed NL2SQL datasets.", automatic_schema=True)
    _add_dataset_job_arguments(parser)
    return parser.parse_args(argv)


def _add_dataset_job_arguments(parser):
    parser.add_argument("--split", choices=["valid", "error"], default=None)
    parser.add_argument("--input-dir", type=Path, help="Auto-detect data_valid, bird_e or spider_e")
    parser.add_argument("--files", nargs="+", help="Optional JSON basenames to run")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved jobs without API calls")


def get_eval_args(argv=None):
    parser = _argument_parser("Evaluate generated NL2SQL results.", automatic_schema=True)
    _add_dataset_job_arguments(parser)
    parser.add_argument(
        "--target-models", nargs="+",
        help="Optional result-directory names to evaluate; default is every model.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-evaluate files whose judge output already exists.",
    )
    return parser.parse_args(argv)


def get_repair_eval_args(argv=None):
    parser = _argument_parser(
        "Repair missing or invalid NL2SQL evaluation records.",
        automatic_schema=True,
    )
    _add_dataset_job_arguments(parser)
    parser.add_argument(
        "--target-models", nargs="+",
        help="Optional result-directory names to repair; default is every model.",
    )
    return parser.parse_args(argv)


def resolve_jobs(options):
    dataset = options.dataset
    split = options.split
    if options.input_dir:
        directory = Path(options.input_dir).resolve()
        if directory.name in {"bird_e", "spider_e"}:
            inferred_dataset = directory.name[:-2]
            inferred_split = "error"
        elif directory.name == "data_valid" and directory.parent.name in {"bird", "spider"}:
            inferred_dataset = directory.parent.name
            inferred_split = "valid"
        else:
            raise ValueError("Input directory must be bird/data_valid, spider/data_valid, bird_e or spider_e")
        if split and split != inferred_split:
            raise ValueError("--split conflicts with the input directory")
        dataset, split = inferred_dataset, inferred_split
    else:
        split = split or "error"
        directory = ROOT / "data" / dataset / ("data_valid" if split == "valid" else dataset + "_e")

    paths = sorted(directory.glob("*.json"))
    if options.files:
        if any(Path(name).name != name for name in options.files):
            raise ValueError("--files accepts basenames only")
        missing = set(options.files) - {p.name for p in paths}
        if missing:
            raise ValueError(f"Input files not found: {sorted(missing)}")
        paths = [p for p in paths if p.name in options.files]
    if not paths:
        raise ValueError(f"No JSON datasets found in {directory}")

    jobs = []
    for path in paths:
        match = re.search(r"([1-4]_[1-3])\.json$", path.name)
        if not match:
            raise ValueError(f"Unrecognized experiment filename: {path.name}")
        category = match.group(1)
        rows = load_dataset(str(path))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Expected a non-empty dataset list: {path}")
        db_ids = {row["db_id"] for row in rows}
        if any(not isinstance(db, str) or Path(db).name != db or "/" in db or "\\" in db for db in db_ids):
            raise ValueError(f"Invalid database identifier in {path}")
        roots = ([ROOT / "data/spider/database"] if dataset == "spider" else [
            ROOT / "data/bird/train_databases/train_databases",
            ROOT / "data/bird/dev_databases/dev_databases",
        ])
        matches = [root for root in roots if all((root / db / (db + ".sqlite")).is_file() for db in db_ids)]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely resolve databases for {path}: {matches}")
        linker = "NoGoldSchemaInfo" if split == "error" and category == "3_2" else "VanillaLinker"
        if split == "valid":
            result = Path(build_result_path(options, path.name))
        else:
            output_name = f"final_{category}.json"
            result = (ROOT / "results" / dataset /
                      options.language_model.replace("/", "_") /
                      options.mode / output_name)
        jobs.append(SimpleNamespace(
            **{**vars(options), "dataset": dataset, "split": split},
            data_path=str(path), database_path=str(matches[0]),
            schema_linker=linker, result_path=str(result),
        ))
    return jobs


