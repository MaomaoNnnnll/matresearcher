"""Regression test: survey report dual-output (survey_report.md + report.md).

Verifies that when output_mode=submission, the survey report body is persisted
to survey_report.md BEFORE the competition-submission rewrite overwrites it,
so the competition deliverable "文献调研报告" has a standalone artifact.

Run: PYTHONPATH=src python scripts/test_survey_dual_output.py
Zero API cost (mock LLM / template path).
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from matresearcher.agents.report_generation import ReportGenerationAgent

SURVEY_MARK = "## 1. 研究问题概述"
SUBMISSION_MARK = "六章参赛方案文档-MOCK"


class MockLLM:
    """Minimal mock: survey system-prompt -> survey body, submission -> submission body."""

    async def complete(self, system: str, prompt: str, **kwargs) -> str:
        if "参赛方案撰写专家" in system:
            return SUBMISSION_MARK + "\n\n（提交稿正文，模板重写）"
        return SURVEY_MARK + "\n\n调研报告正文（LLM 生成）"


def _state() -> dict:
    return {
        "raw_question": "test",
        "candidate_literature": [],
        "filtered_literature": [],
        "fused_table": None,
        "conflicts": [],
        "missing_items": [],
        "scored_gaps": [],
    }


async def run_case(output_mode: str, llm) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as td:
        agent = ReportGenerationAgent(llm=llm, config={"output_mode": output_mode}, log_dir=td)
        try:
            result = await agent.run(_state())
            sp = Path(td) / "survey_report.md"
            assert sp.exists(), "survey_report.md was not persisted"
            survey = sp.read_text(encoding="utf-8")
            draft = result.get("draft_report", "")
            assert draft, "draft_report missing from result"
            return draft, survey
        finally:
            agent.close_log()  # release the .log handle so TemporaryDirectory can clean up (Windows)


async def main() -> None:
    # Case 1: template path (llm=None) — survey_report.md = template survey body
    d1, s1 = await run_case("submission", None)
    assert SURVEY_MARK in s1, "C1: survey_report.md should contain survey body"

    # Case 2: mock LLM + submission — survey_report.md must hold the PRE-rewrite
    # survey body, while draft_report is the rewritten submission.
    d2, s2 = await run_case("submission", MockLLM())
    assert SURVEY_MARK in s2, "C2: survey_report.md should be the survey body"
    assert SUBMISSION_MARK not in s2, "C2: survey_report.md must NOT contain submission text"
    assert SUBMISSION_MARK in d2, "C2: draft_report should be the rewritten submission"

    # Case 3: mock LLM + survey mode — no rewrite; both files carry the survey body
    d3, s3 = await run_case("survey", MockLLM())
    assert SURVEY_MARK in s3 and SURVEY_MARK in d3, "C3: survey body expected in both outputs"

    print(f"C1 template path            PASS (survey={len(s1)} chars)")
    print(f"C2 submission dual-output   PASS (survey={len(s2)} chars, draft={len(d2)} chars)")
    print(f"C3 survey mode              PASS (survey={len(s3)} chars, draft={len(d3)} chars)")
    print("\nALL PASS")


if __name__ == "__main__":
    asyncio.run(main())
