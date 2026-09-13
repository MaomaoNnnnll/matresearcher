#!/usr/bin/env python3
"""CLI wrapper: evaluate MatResearcher against real baselines.

The implementation lives in ``src/matresearcher/evaluation/`` so that it is
importable as a package (the previous ``from ...scripts.evaluate import ...``
in main.py was a 3-level relative import beyond the top-level package and
raised ImportError at runtime).

Usage:
    python scripts/evaluate.py "your research question" -b all -o results.json
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from matresearcher.evaluation import run_evaluation  # noqa: E402
from matresearcher.evaluation import DEFAULT_QUESTION  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MatResearcher")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "-b", "--baseline", default="all",
        help="Baselines to run (keyword, semantic, hybrid, rag, all)",
    )
    parser.add_argument("-o", "--output", default="evaluation_results.json")
    parser.add_argument(
        "--skip-pipeline", action="store_true",
        help="Only run baselines (skip the full 12-node pipeline).",
    )
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            question=args.question,
            baseline=args.baseline,
            output_path=args.output,
            run_matresearcher=not args.skip_pipeline,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
