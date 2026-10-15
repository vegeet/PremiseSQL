"""Assign formatted source questions to premise-error categories."""

import argparse
import json
import random
from pathlib import Path


CATEGORIES = ("1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "3.3", "4.1", "4.2")
DEFAULT_WEIGHTS = (1, 1, 1, 1, 1, 1, 1, 1, 1)


def split_samples_weighted(samples_path, subcategories, output_path, weights=None, seed=42):
    samples = json.loads(Path(samples_path).read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("Input must be a JSON list")
    if not subcategories or len(set(subcategories)) != len(subcategories):
        raise ValueError("Categories must be a nonempty list without duplicates")
    weights = list(weights) if weights is not None else [1] * len(subcategories)
    if len(weights) != len(subcategories) or any(w < 0 for w in weights) or sum(weights) == 0:
        raise ValueError("Weights must match categories, be nonnegative, and sum above zero")

    total = len(samples)
    total_weight = sum(weights)
    counts = [total * w // total_weight for w in weights]
    remainder = total - sum(counts)
    for i in sorted(range(len(subcategories)), key=lambda i: weights[i], reverse=True):
        if remainder == 0:
            break
        counts[i] += 1
        remainder -= 1

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    assigned = []
    offset = 0
    for category, count in zip(subcategories, counts):
        assigned.extend({**item, "assigned_subcategory": category}
                        for item in shuffled[offset:offset + count])
        offset += count

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(assigned, ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(zip(subcategories, counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Formatted source questions")
    parser.add_argument("output", type=Path)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--weights", nargs="+", type=int, default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(split_samples_weighted(args.input, args.categories, args.output,
                                 args.weights, args.seed))


if __name__ == "__main__":
    main()
