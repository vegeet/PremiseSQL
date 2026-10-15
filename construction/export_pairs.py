"""Export human-approved flawed questions and their original valid partners."""

import argparse
import json
import re
from pathlib import Path


def export_pairs(input_dir, output_root, dataset):
    error_folder = output_root / dataset / f"{dataset}_e"
    valid_folder = output_root / dataset / "data_valid"
    error_folder.mkdir(parents=True, exist_ok=True)
    valid_folder.mkdir(parents=True, exist_ok=True)
    summary = {}
    for source in sorted(input_dir.glob("*.json")):
        category = re.search(r"([1-4]_[1-3])\.json$", source.name)
        if not category:
            continue
        rows = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list: {source}")
        flawed = []
        valid = []
        for row in rows:
            if row.get("human_review", {}).get("passed") is not True:
                continue
            question = row.get("rewritten_question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Approved row has no rewritten question: {source}")
            common = {field: row[field] for field in ("sample_id", "db_id", "evidence")
                      if field in row}
            flawed.append({**common, "question": question, "sql": row.get("sql")})
            valid.append({**common, "question": row["question"], "sql": row.get("sql")})
        filename = f"final_{category.group(1)}.json"
        for folder, values in ((error_folder, flawed), (valid_folder, valid)):
            (folder / filename).write_text(
                json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        summary[filename] = len(flawed)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_dir", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--dataset", choices=("bird", "spider"), required=True)
    args = parser.parse_args()
    print(export_pairs(args.review_dir, args.data_root, args.dataset))


if __name__ == "__main__":
    main()
