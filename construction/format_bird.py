"""Format BIRD questions with database schema and foreign keys."""

import argparse
import json
from pathlib import Path


def build_schema(db_item):
    schema = {name: [] for name in db_item["table_names_original"]}
    for table_index, column in db_item["column_names_original"]:
        if table_index >= 0:
            schema[db_item["table_names_original"][table_index]].append(column)
    return schema


def build_foreign_keys(db_item):
    columns = db_item["column_names_original"]
    tables = db_item["table_names_original"]
    keys = []
    for source, target in db_item.get("foreign_keys", []):
        source_table, source_column = columns[source]
        target_table, target_column = columns[target]
        if source_table >= 0 and target_table >= 0:
            keys.append({"from": f"{tables[source_table]}.{source_column}",
                         "to": f"{tables[target_table]}.{target_column}"})
    return keys


def build_inlined_dataset(samples_path, tables_path, output_path):
    samples = json.loads(Path(samples_path).read_text(encoding="utf-8"))
    tables = json.loads(Path(tables_path).read_text(encoding="utf-8"))
    schemas = {item["db_id"]: item for item in tables}
    formatted = []
    for index, item in enumerate(samples):
        db_id = item["db_id"]
        if db_id not in schemas:
            raise ValueError(f"Schema for db_id {db_id} not found")
        db = schemas[db_id]
        formatted.append({
            "sample_id": index,
            "db_id": db_id,
            "schema": build_schema(db),
            "foreign_keys": build_foreign_keys(db),
            "question": item["question"],
            "sql": item.get("SQL") or item.get("sql"),
            "evidence": item.get("evidence", ""),
        })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(formatted, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(formatted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("tables", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"Formatted {build_inlined_dataset(args.questions, args.tables, args.output)} samples")


if __name__ == "__main__":
    main()
