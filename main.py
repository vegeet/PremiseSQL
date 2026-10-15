from tqdm import tqdm
import json
from pathlib import Path
from utils import (
    ROOT,
    get_run_args,
    resolve_jobs,
    load_dataset,
    seed_everything,
    remove_sql_comments,
)



def runner(args):
    import model
    import schema_linking
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    seed_everything(42)

    agent = model.load_agent(args.method, args)
    dataset = load_dataset(args.data_path)
    schema_linker = schema_linking.load_schema_linker(
        args.schema_linker,
        gold_schema_path=str(ROOT / "data" / args.dataset / "dev_gold_schema.json"),
    )

    for item in tqdm(dataset):
        db_name = item['db_id']
        question = item['question']
        evidence = item.get('evidence', None)

        schema = schema_linker.get_schema(
            args.database_path,
            db_name,
            question,
            evidence
        )

        response = agent(schema, question, evidence)

        yield {
            "db_id": db_name,
            "question": question,
            "evidence": evidence or "",
            "original_schema": schema,
            "original_schema_linker": args.schema_linker,
            "raw_response": response
        }


def run(args):
    with open(args.result_path, "w", encoding="utf-8") as f:

        f.write("[\n")
        first = True

        for item in runner(args):

            response = item["raw_response"]

            # -------------------------
            # 统一取 sql
            # -------------------------
            sql = response.get("sql", "")

            if isinstance(sql, list):
                sql = " ".join(str(s) for s in sql)
            elif not isinstance(sql, str):
                sql = str(sql)

            sql = sql.strip()

            if sql:
                sql = remove_sql_comments(sql)
                sql = sql.replace('\n', ' ').strip()

            if not sql:
                sql = "None"

            # -------------------------
            # 根据 mode 决定字段
            # -------------------------
            if args.mode == "base":
                record = {
                    "db_id": item["db_id"],
                    "question": item["question"],
                    "sql": sql,
                    "reason": response.get("reason", None)
                }
            else:
                record = {
                    "db_id": item["db_id"],
                    "question": item["question"],
                    "sql": sql,
                    "is_valid": response.get("is_valid", False),
                    "reason": response.get("reason", None)
                }

            # Preserve actual exposure, particularly the randomly reduced 3_2 schema.
            record.update({key: item[key] for key in
                           ("evidence", "original_schema", "original_schema_linker")})

            if not first:
                f.write(",\n")
            first = False

            f.write(json.dumps(record, ensure_ascii=False, indent=4))
            f.flush()

        f.write("\n]")


def process_files():
    options = get_run_args()
    # Resolve every job before making any paid request.
    for args in resolve_jobs(options):
        print(json.dumps({key: getattr(args, key) for key in
              ("dataset", "split", "data_path", "database_path", "schema_linker", "result_path")},
              ensure_ascii=False))
        if not options.dry_run:
            Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
            run(args)


if __name__ == "__main__":
    process_files()
