"""Format Spider questions with their schema and foreign keys."""

import argparse
import json
from pathlib import Path


def load_schemas(tables_json_path):
    tables = json.loads(Path(tables_json_path).read_text(encoding="utf-8"))
    schemas = {}
    for db in tables:
        names = db["table_names_original"]
        columns = db["column_names_original"]
        types = db["column_types"]
        table_info = {name: [] for name in names}
        for index, (table_id, column) in enumerate(columns):
            if table_id >= 0:
                table_info[names[table_id]].append({"name": column, "type": types[index]})
        foreign_keys = []
        for source, target in db["foreign_keys"]:
            source_table, source_column = columns[source]
            target_table, target_column = columns[target]
            foreign_keys.append({
                "from": f"{names[source_table]}.{source_column}",
                "to": f"{names[target_table]}.{target_column}",
            })
        schemas[db["db_id"]] = {"tables": table_info, "foreign_keys": foreign_keys}
    return schemas


def convert_spider_to_llm_json(spider_json_path, tables_json_path, output_path):
    schemas = load_schemas(tables_json_path)
    source = json.loads(Path(spider_json_path).read_text(encoding="utf-8"))
    formatted = []
    for index, item in enumerate(source):
        db_id = item["db_id"]
        if db_id not in schemas:
            raise ValueError(f"Schema for db_id {db_id} not found")
        formatted.append({
            "sample_id": index,
            "db_id": db_id,
            "schema": schemas[db_id]["tables"],
            "foreign_keys": schemas[db_id]["foreign_keys"],
            "question": item["question"],
            "sql": item["query"],
        })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(formatted, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(formatted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path, help="Spider train_spider.json or dev.json")
    parser.add_argument("tables", type=Path, help="Spider tables.json")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(f"Formatted {convert_spider_to_llm_json(args.questions, args.tables, args.output)} samples")


if __name__ == "__main__":
    main()
