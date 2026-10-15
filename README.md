# PremiseSQL research code

This repository contains the code used to construct premise-error Text-to-SQL questions and to evaluate how language models respond to them. It covers four stages: question construction, model generation, three-metric evaluation, and evidence-gated failure attribution. The original BIRD and Spider databases, generated responses, annotations, and paper working files are distributed separately if their terms permit.

## Contents

| Path | Purpose |
| --- | --- |
| `construction/format.py`, `construction/format_bird.py` | Convert separately obtained Spider/BIRD questions and schemas to construction records. |
| `construction/assign.py`, `construction/runner.py`, `construction/pipeline/` | Assign taxonomy categories, rewrite questions, review them, and keep accepted cases. |
| `construction/human_review.py`, `construction/export_pairs.py` | Record human decisions and export approved valid/flawed pairs. |
| `main.py`, `model/`, `language_model/`, `schema_linking/` | Generate SQL and validity responses under base, implicit, or explicit settings. |
| `eval_main.py`, `eval/` | Score SQL behavior, flaw detection, and flaw type identification. |
| `failure_analysis/` | Select agreed failures, sample them, assign observable mechanisms M01–M04, and calculate agreement. |
| `error_type_metrics_recompute.py`, `valid_bidirectional_consistency.py` | Recompute aggregate metrics and validity consistency from saved responses. |

The taxonomy definition is in `construction/metadata/premise_error_taxonomy.json`. The construction prompts and attribution prompt are included to preserve the reported methodology.

## Acknowledgements

The model invocation and schema-linking components in this project were adapted from [NL2SQL-Benchmark](https://github.com/Artessay/NL2SQL-Benchmark). We thank its authors for making their benchmark implementation available.

## Setup

Use Python 3.10 or newer. Install dependencies with `python -m pip install -r requirements.txt`. Copy `.env.example` to `.env` and set your own API key, compatible endpoint, and model IDs. Model requests incur provider charges. The scripts have no bundled credentials.

See [DATA_FORMAT.md](DATA_FORMAT.md) for expected inputs, layout, and construction commands. Local checks passed for the failure-attribution unit tests and mock-data smoke paths covering construction, pair export, generation/evaluation dry runs, and client request shape. The full pipeline with the paper's datasets and live model endpoints has not been reproduced in this release preparation.

## Evaluation commands

After preparing the datasets and SQLite databases under `data/` as documented:

```sh
python main.py --dataset bird --split error --mode implicit --dry-run
python main.py --dataset bird --split error --mode implicit
python eval_main.py --dataset bird --split error --mode implicit -l your/judge-model --dry-run
python eval_main.py --dataset bird --split error --mode implicit -l your/judge-model
python -m failure_analysis.wash --results-root results
python -m failure_analysis.sample_pool --data-root data
python failure_main.py --dataset bird --max-samples 10 --dry-run
```

`--dry-run` resolves generation/evaluation jobs or saves a bounded attribution prompt preview without model calls. Full results depend on the exact dataset, model endpoint, judge pair, and recorded generation inputs. Keep model and prompt versions with any reported numbers.

## Release scope

This MIT license covers this repository's code and documentation. BIRD, Spider, their databases and annotations, and any other separately obtained data retain their own terms. Before publishing, verify authorship and reuse rights for copied research code and rotate credentials that were ever embedded in private working copies. No experiment result is claimed as reproduced by this organization pass.
