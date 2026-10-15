# Data preparation and expected layout

The source datasets and SQLite databases are intentionally absent. Obtain BIRD and Spider under their respective terms, then prepare the project-local `data/` folder. All paths below are relative to the repository root. Generated files under `data/`, `results/`, and `validity/` are ignored by Git.

## Constructing premise-error questions

`construction/format.py` reads Spider question JSON (`train_spider.json` or `dev.json`) plus `tables.json`. `construction/format_bird.py` reads BIRD question JSON plus its table metadata JSON. Both produce records with `sample_id`, `db_id`, `schema`, `foreign_keys`, `question`, and `sql`; BIRD also preserves `evidence` when present. Then `construction/assign.py` adds `assigned_subcategory` with a fixed seed and the original category weights.

```sh
python -m construction.format path/to/train_spider.json path/to/tables.json data/spider/formatted.json
python -m construction.format_bird path/to/bird_questions.json path/to/bird_tables.json data/bird/formatted.json
python -m construction.assign data/spider/formatted.json data/spider/assigned.json
python -m construction.runner --input data/spider/assigned.json --category 1.1 --output-dir construction/output/spider
```

The last command writes `rewritten_1_1.json`, `reviewed_1_1.json`, and `accepted_1_1.json`. Repeat for each category. The rewrite and review steps call the configured model endpoints. Inspect accepted cases before using them as benchmark inputs; this code does not replace human quality review.

The included review UI records a decision and optional edited question for each accepted candidate. Export only approved pairs:

```sh
python -m construction.human_review construction/output/spider/accepted_1_1.json construction/output/spider/human/final_1_1.json
python -m construction.export_pairs construction/output/spider/human data --dataset spider
```

Each exported flawed item uses the edited `rewritten_question`; its valid partner retains the original `question`. Review every category before export. The export keeps `sample_id`, `db_id`, `evidence` when available, and the source valid SQL for provenance; the generation runner does not read the SQL as a target.

## Generation and evaluation input

The model runner expects JSON lists in these folders:

```text
data/
  bird/
    bird_e/*.json
    data_valid/*.json
    train_databases/train_databases/<db_id>/<db_id>.sqlite
    dev_databases/dev_databases/<db_id>/<db_id>.sqlite
    dev_gold_schema.json
  spider/
    spider_e/*.json
    data_valid/*.json
    database/<db_id>/<db_id>.sqlite
    dev_gold_schema.json
```

Each JSON filename must end in a category code such as `1_1.json` or `final_1_1.json`. Each list entry needs `db_id` and `question`; `evidence` is optional. The benchmark construction metadata may retain `sql`, the source valid question's SQL, but the generation runner does not use it as a target answer. The `3_2` flawed-question condition uses `dev_gold_schema.json` to remove a reproducibly seeded subset of gold columns from the exposed schema. Its original exposure is recorded with the model output for later evaluation.

The construction pipeline's `accepted_*.json` files are intermediate outputs. `export_pairs.py` writes the runner's `bird_e` or `spider_e` layout from human-approved rows. The repository does not redistribute the original benchmark samples or a final paper dataset.
