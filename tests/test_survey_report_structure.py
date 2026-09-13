"""Regression test: survey_report.md must use the canonical section order.

Target layout (user-specified 2026-09-13 reorder):
  # <title>
  ## 1. 研究问题概述 ... ## 5. 研究缺口发现
  ## 6. 构效关系定量分析 (Structure-Property Analysis)
  ## 7. 方法局限性
  ## 参考文献清单            (unnumbered — moved after 方法局限性)
  ## 附录 A. 证据溯源清单

This guards both the section presence AND the exact ordering, and that
survey_report.md is never clobbered by the submission document.
"""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from matresearcher.agents.report_generation import ReportGenerationAgent

STUB_SURVEY_DRAFT = """# LLZO 固态电解质离子电导率文献调研报告

## 1. 研究问题概述
### 1.1 原始问题
x

## 2. 检索方法与文献覆盖
### 2.1 检索策略
x

## 3. 知识体系图
x

## 4. 数据一致性分析
### 4.1 检测到的数据冲突
x

## 5. 研究缺口发现
### 缺口 1：x
x

## 6. 参考文献清单
[1] Author et al. Title. Journal, Year.

## 7. 方法局限性
### 7.1 检索覆盖范围的局限
x
"""

SP_MD = """## 构效关系定量分析 (Structure-Property Analysis)

| Material | Conductivity |
|----------|--------------|
| LLZO     | 1.0e-3 S/cm  |
"""

EVIDENCE_MD = """## 附录 A. 证据溯源清单

- 文献#1 -> passage
"""


def _make_agent(tmp_path: Path, output_mode: str):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    agent = ReportGenerationAgent(
        llm=MagicMock(), config={"output_mode": output_mode}, log_dir=str(run_dir)
    )
    agent._log_path = str(run_dir / "report_generation.log")
    # Mock heavy helpers that touch real data / external services.
    agent._build_evidence_manifest = MagicMock(return_value=EVIDENCE_MD)
    agent._format_cross_checks = MagicMock(return_value="")
    agent._build_knowledge_table = MagicMock(return_value="（无数据）")
    agent._format_references = MagicMock(return_value="[1] ref")
    agent._format_conflicts = MagicMock(return_value="")
    agent._format_gaps = MagicMock(return_value="")
    agent._build_pipeline_summary = MagicMock(return_value="pipeline")
    agent._build_tools_list = MagicMock(return_value="tools")
    agent._build_agent_architecture = MagicMock(return_value="arch")
    # LLM returns the stub 7-chapter survey draft for both report & submission.
    agent.llm.complete = AsyncMock(return_value=STUB_SURVEY_DRAFT)
    return agent, run_dir


def _base_state():
    return {
        "raw_question": "test",
        "candidate_literature": [],
        "filtered_literature": [],
        "fused_table": None,
        "conflicts": [],
        "missing_items": [],
        "scored_gaps": [{"gap": "g"}],
        "_structure_property_md": SP_MD,
        "hypothesis_cross_checks": [],
        "draft_report": "",
        "fact_check_revision": 0,
        "fact_check_issues": [],
    }


async def _run_and_read(tmp_path, output_mode):
    agent, run_dir = _make_agent(tmp_path, output_mode)
    result = await agent.run(_base_state())
    survey_md = (run_dir / "survey_report.md").read_text(encoding="utf-8")
    return result, survey_md


def _assert_survey_structure(survey_md: str):
    # Required sections present
    assert "## 6. 构效关系定量分析" in survey_md, "missing/renumbered structure-property section"
    assert "## 7. 方法局限性" in survey_md, "missing limitations section"
    assert "## 参考文献清单" in survey_md, "missing references list"
    assert "## 附录 A. 证据溯源清单" in survey_md, "missing evidence appendix"
    # References list must be UNNUMBERED (no "## 6. 参考文献清单" / "## 8. ...")
    assert "## 6. 参考文献清单" not in survey_md, "references list should be unnumbered"
    # Correct canonical order: 6.SP < 7.方法局限 < 参考文献 < 附录A
    i_sp = survey_md.index("## 6. 构效关系定量分析")
    i7 = survey_md.index("## 7. 方法局限性")
    i_refs = survey_md.index("## 参考文献清单")
    ia = survey_md.index("## 附录 A. 证据溯源清单")
    assert i_sp < i7 < i_refs < ia, "section order wrong"
    # Must NOT be the submission document
    assert "## 参赛方案文档" not in survey_md, "survey_report.md was clobbered by submission"
    return i_sp, i7, i_refs, ia


@pytest.mark.asyncio
async def test_survey_report_structure_survey_mode(tmp_path):
    _, survey_md = await _run_and_read(tmp_path, "survey")
    _assert_survey_structure(survey_md)


@pytest.mark.asyncio
async def test_survey_report_structure_submission_mode(tmp_path):
    """Dual output: survey_report.md keeps canonical structure; submission separate."""
    result, survey_md = await _run_and_read(tmp_path, "submission")
    _assert_survey_structure(survey_md)
    # draft_report stays the SURVEY version (with canonical structure-property chapter)
    assert "## 6. 构效关系定量分析" in result["draft_report"]
    # submission_report is generated and also carries the evidence appendix
    assert result["submission_report"]
    assert "## 附录 A. 证据溯源清单" in result["submission_report"]


@pytest.mark.asyncio
async def test_revision_round_is_idempotent(tmp_path):
    """A fact-check revision of an already-assembled survey must not duplicate/reorder."""
    agent, run_dir = _make_agent(tmp_path, "survey")
    # First pass produces a canonical survey_report stored as draft_report.
    r1 = await agent.run(_base_state())
    assembled = r1["draft_report"]
    _assert_survey_structure(assembled)  # sanity: first pass is canonical

    # Simulate a revision: previous_draft = assembled survey; revise edits in place.
    async def _fake_revise(previous_draft, issues, question, generate_time):
        return previous_draft.replace("# LLZO 固态电解质离子电导率文献调研报告",
                                     "# LLZO 固态电解质离子电导率文献调研报告 [REVISED]")

    agent._revise_with_llm = _fake_revise  # type: ignore[assignment]
    state = _base_state()
    state["fact_check_revision"] = 1
    state["fact_check_issues"] = ["some issue"]
    state["draft_report"] = assembled

    r2 = await agent.run(state)
    _assert_survey_structure(r2["draft_report"])
    assert "[REVISED]" in r2["draft_report"]
    assert r2["draft_report"].count("## 6. 构效关系定量分析") == 1
    assert r2["draft_report"].count("## 参考文献清单") == 1
