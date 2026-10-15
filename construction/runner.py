"""Rewrite, review, and filter one premise-error category."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Assigned questions JSON")
    parser.add_argument("--category", required=True, help="Category such as 1.1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path,
                        default=Path(__file__).parent / "metadata" / "premise_error_taxonomy.json")
    parser.add_argument("--rewrite-only", action="store_true")
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    if args.rewrite_only and args.review_only:
        parser.error("Choose at most one of --rewrite-only and --review-only")

    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    category_name = next(
        (name for name, group in taxonomy.items()
         if args.category in group["subcategories"]), None
    )
    if category_name is None:
        parser.error(f"Unknown category: {args.category}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.category.replace(".", "_")
    rewritten = args.output_dir / f"rewritten_{tag}.json"
    reviewed = args.output_dir / f"reviewed_{tag}.json"
    accepted = args.output_dir / f"accepted_{tag}.json"

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if not args.review_only:
        from construction.pipeline.rewrite import rewrite_dataset
        rewrite_dataset(str(args.input), str(args.taxonomy), str(rewritten),
                        category_name, args.category, args.category)
    if not args.rewrite_only:
        if not rewritten.is_file():
            parser.error(f"Missing rewritten input: {rewritten}")
        from construction.pipeline.revise_gemini import review_dataset_gemini
        from construction.pipeline.wash import keep_true
        review_dataset_gemini(str(rewritten), str(reviewed), args.category)
        keep_true(str(reviewed), str(accepted))


if __name__ == "__main__":
    main()
