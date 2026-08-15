#!/usr/bin/env python3
"""Run a single literature survey.

Usage:
    python scripts/run_survey.py "LLZO ionic conductivity comparison"
    python scripts/run_survey.py -q "What is the optimal sintering temperature for LLZO?" -o output/report.md
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.matresearcher.workflow.engine import MatResearcherWorkflow


async def main():
    parser = argparse.ArgumentParser(
        description="MatResearcher: Literature Survey Runner"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="What are the ionic conductivity values of LLZO solid electrolytes?",
        help="Research question to survey",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to workflow config YAML",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="report.md",
        help="Output report file path",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MatResearcher Literature Survey")
    print(f"{'='*60}")
    print(f"Question: {args.question}")
    print(f"Config: {args.config or 'default'}")
    print(f"Output: {args.output}")
    print()

    # Run workflow
    workflow = MatResearcherWorkflow(args.config)
    result = await workflow.run(args.question)

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = result.get("final_report", result.get("draft_report", ""))
    output_path.write_text(report, encoding="utf-8")

    print(f"\nReport saved to: {output_path.absolute()}")
    print(f"Report length: {len(report)} characters")

    # Print gap summary
    gaps = result.get("scored_gaps", result.get("gaps", []))
    if gaps:
        print(f"\nTop Research Gaps:")
        for i, gap in enumerate(gaps[:3]):
            if hasattr(gap, "description"):
                desc = gap.description[:100]
                score = gap.score.total_score if gap.score else 0
            else:
                desc = gap.get("description", "")[:100]
                s = gap.get("score", {})
                score = s.get("total_score", 0) if isinstance(s, dict) else 0
            print(f"  {i+1}. [{score:.3f}] {desc}...")

    return result


if __name__ == "__main__":
    asyncio.run(main())
