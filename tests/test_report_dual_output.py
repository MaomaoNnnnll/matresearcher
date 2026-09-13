"""Regression test for the survey / submission dual-output split.

Bug (found 2026-09-13): in `output_mode=submission`, report_generation.run()
rewrote the shared `draft` variable into the submission document
(`draft = await self._generate_submission(draft)`). Because the fact-check
revision loop reads `state["draft_report"]` as the previous draft, every
revision round then re-edited the *submission* doc, and the survey-style
sections (参考文献清单 / 构效关系定量分析) were dropped from the final
survey_report.md.

The fix keeps `draft` as the survey report and exposes the submission via a
separate `submission_report` state key. This test locks that contract.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from matresearcher.agents.report_generation import ReportGenerationAgent  # noqa: E402


SURVEY_MD = (
    "# Survey\n\n## 1. 研究问题概述\n\n## 6. 参考文献清单\n\nref1\n\n"
    "## 构效关系定量分析 (Structure-Property Analysis)\n\nsp\n"
)
SUBMISSION_MD = "## 参赛方案文档\n\n一、项目概述\n\nsubmission body\n"


class _FakeLLM:
    """Returns a fixed survey report; never touched by the test directly."""

    async def complete(self, system, prompt, **kwargs):
        return SURVEY_MD


def _make_agent(tmp_path: Path, output_mode: str) -> ReportGenerationAgent:
    agent = ReportGenerationAgent(
        llm=_FakeLLM(), config={"output_mode": output_mode}, log_dir=str(tmp_path)
    )
    # _log_path drives survey_report.md persistence; point it inside tmp_path.
    # The run log dir must exist (the engine creates it in production).
    (tmp_path / "run").mkdir(parents=True, exist_ok=True)
    agent._log_path = str(tmp_path / "run" / "report_generation.log")

    # Stub the data-formatting helpers that need real literature/gap objects.
    agent._format_references = lambda rm: "REF"
    agent._build_knowledge_table = lambda *a, **k: "KT"
    agent._format_conflicts = lambda c: "CONF"
    agent._format_gaps = lambda sg, rm: "GAPS"
    agent._build_evidence_manifest = lambda sg: "## 附录 A. 证据溯源清单\nM"

    # Survey draft comes from a stub so we don't depend on a real LLM call shape.
    async def _fake_generate(**kwargs):
        return SURVEY_MD

    agent._generate_with_llm = _fake_generate  # type: ignore[assignment]
    agent._generate_template_based = _fake_generate  # type: ignore[assignment]

    # Submission doc is produced from whatever survey draft was passed in.
    async def _fake_submission(survey_report: str) -> str:
        assert "参考文献清单" in survey_report, "submission must derive from survey draft"
        return SUBMISSION_MD

    agent._generate_submission = _fake_submission  # type: ignore[assignment]
    return agent


def _base_state(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "raw_question": "Q",
        "candidate_literature": [],
        "filtered_literature": [],
        "fused_table": None,
        "conflicts": [],
        "missing_items": [],
        "scored_gaps": [],
        "fact_check_issues": [],
        "fact_check_revision": 0,
        "draft_report": "",
        "_structure_property_md": "## 构效关系定量分析 (Structure-Property Analysis)\nSP",
    }
    if extra:
        state.update(extra)
    return state


def test_submission_mode_splits_survey_and_submission(tmp_path: Path):
    agent = _make_agent(tmp_path, "submission")
    result = asyncio.run(agent.run(_base_state()))

    # draft_report must stay the SURVEY report (with the two survey-only sections).
    draft = result["draft_report"]
    assert "参考文献清单" in draft
    assert "构效关系定量分析" in draft
    assert "参赛方案文档" not in draft

    # submission_report carries the competition doc, separate from the survey.
    sub = result["submission_report"]
    assert "参赛方案文档" in sub
    assert sub != draft

    # survey_report.md on disk must be the survey version, not the submission.
    written = (tmp_path / "run" / "survey_report.md").read_text(encoding="utf-8")
    assert "参考文献清单" in written
    assert "参赛方案文档" not in written


def test_survey_mode_has_empty_submission_report(tmp_path: Path):
    agent = _make_agent(tmp_path, "survey")
    result = asyncio.run(agent.run(_base_state()))
    assert "参考文献清单" in result["draft_report"]
    assert result["submission_report"] == ""


def test_revision_round_keeps_survey_report_intact(tmp_path: Path):
    """The regression: after a fact-check revision the survey content must survive."""
    agent = _make_agent(tmp_path, "submission")

    # Simulate the previous draft (as stored in state by the prior pass) being a
    # SURVEY report; the revision must re-edit the survey, not the submission.
    async def _fake_revise(previous_draft, issues, question, generate_time):
        assert "参考文献清单" in previous_draft
        return previous_draft.replace("# Survey", "# Survey [REVISED]")

    agent._revise_with_llm = _fake_revise  # type: ignore[assignment]

    state = _base_state(
        {
            "fact_check_revision": 1,
            "fact_check_issues": ["some issue"],
            "draft_report": SURVEY_MD,  # what the prior pass returned
        }
    )
    result = asyncio.run(agent.run(state))

    # Critical: draft_report is still the (revised) SURVEY report, NOT the submission.
    assert "[REVISED]" in result["draft_report"]
    assert "参考文献清单" in result["draft_report"]
    assert "参赛方案文档" not in result["draft_report"]

    # survey_report.md on disk reflects the survey revision, never the submission.
    written = (tmp_path / "run" / "survey_report.md").read_text(encoding="utf-8")
    assert "[REVISED]" in written
    assert "参赛方案文档" not in written
