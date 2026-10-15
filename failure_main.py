"""Versioned mechanism attribution; full matched dataset by default, with resumable outputs."""
from failure_analysis.runner import FailurePatternEvaluator, failure_main, failure_all, get_args


if __name__ == "__main__":
    failure_all(get_args())
