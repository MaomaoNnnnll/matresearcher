"""Regression tests for the fact_check ↔ report_generation rewrite loop.

The bug (2026-09-11): the per-node run cache keyed only by node name and was
read *within* a single run. On the 2nd visit to `fact_check`/`report_generation`
the cached patch was replayed, so `fact_check_revision` never advanced and
`needs_revision` stayed True forever → GraphRecursionError at 10007.

These tests lock in the fix (LOOP_NODES bypasses the cache) and prove the loop
actually terminates.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Optional, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from matresearcher.workflow.engine import (  # noqa: E402
    LOOP_NODES,
    MatResearcherWorkflow,
)
from langgraph.graph import StateGraph, END  # noqa: E402

# Mirrors agents.evidence_verification.max_revisions in config/workflow.yaml
MAX_REVISIONS = 2


class _MockCache:
    """Minimal cache stub that records whether get_node was consulted."""

    def __init__(self):
        self.get_node_calls: list[str] = []

    def has_node(self, name: str) -> bool:
        return True  # pretend every node is cached

    def get_node(self, name: str) -> Optional[dict]:
        self.get_node_calls.append(name)
        return {"__replayed__": name}  # stale patch that would re-trigger the loop

    def save_node(self, name: str, patch: dict) -> None:
        pass


def _workflow_with_cache(cache) -> MatResearcherWorkflow:
    """Build a MatResearcherWorkflow without running __init__ (no YAML/tools)."""
    obj = MatResearcherWorkflow.__new__(MatResearcherWorkflow)
    obj._cache = cache
    obj._current_node = None
    return obj


# ── LOOP_NODES membership ────────────────────────────────────────────────────

def test_loop_nodes_are_the_two_cycle_nodes():
    assert LOOP_NODES == {"report_generation", "fact_check"}


# ── Cache bypass (the actual root-cause guard) ───────────────────────────────

def test_loop_nodes_never_replay_cache():
    cache = _MockCache()
    wf = _workflow_with_cache(cache)

    async def dummy(state):
        return {"x": 1}

    for node in LOOP_NODES:
        wrapped = wf._wrap_node(node, dummy)
        out = asyncio.run(wrapped({"x": 0}))
        # Must run the real node, NOT return the replayed stale patch.
        assert out == {"x": 1}, f"{node} should have executed, got replayed cache"
        assert node not in cache.get_node_calls, f"{node} must bypass cache read"

    # Non-loop nodes should still use the cache (resume acceleration preserved).
    cache2 = _MockCache()
    wf2 = _workflow_with_cache(cache2)
    wrapped = wf2._wrap_node("literature_search", dummy)
    out = asyncio.run(wrapped({"x": 0}))
    assert out == {"__replayed__": "literature_search"}, "non-loop nodes reuse cache"
    assert "literature_search" in cache2.get_node_calls


# ── End-to-end convergence on a real mini graph ─────────────────────────────

def test_fact_check_loop_converges_and_terminates():
    cache = _MockCache()
    wf = _workflow_with_cache(cache)

    class MiniState(TypedDict, total=False):
        fact_check_revision: int
        needs_revision: bool

    async def report_generation(state):
        # No-op body; the real agent rewrites the report here.
        return {}

    async def fact_check(state):
        rev = state.get("fact_check_revision", 0) + 1
        # needs_revision True while budget remains, then flips to False.
        needs = rev <= MAX_REVISIONS
        return {"fact_check_revision": rev, "needs_revision": needs}

    graph = StateGraph(MiniState)
    graph.add_node(
        "report_generation", wf._wrap_node("report_generation", report_generation)
    )
    graph.add_node("fact_check", wf._wrap_node("fact_check", fact_check))
    graph.set_entry_point("report_generation")
    graph.add_edge("report_generation", "fact_check")
    graph.add_conditional_edges(
        "fact_check",
        wf._route_after_fact_check,
        {"revise": "report_generation", "done": END},
    )
    app = graph.compile()

    final = asyncio.run(
        app.ainvoke(
            {"fact_check_revision": 0, "needs_revision": True},
            config={"recursion_limit": 50},
        )
    )

    # Converged: revision advanced past the budget and the loop ended.
    assert final["needs_revision"] is False
    assert final["fact_check_revision"] == MAX_REVISIONS + 1


# ── ① in-place rewrite: report_generation truly edits the prior draft ─────────

from matresearcher.agents.report_generation import ReportGenerationAgent  # noqa: E402


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def complete(self, system, prompt, **kwargs):
        self.calls.append((system, prompt))
        return self.reply


def test_revise_with_llm_passes_prior_draft_and_issues():
    fake = _FakeLLM(reply="# Revised\n\nThe unsupported claim was removed.")
    agent = ReportGenerationAgent(llm=fake, config={})
    prev = (
        "## 构效关系定量分析 (Structure-Property Analysis)\n...\n"
        "Some unsupported claim of 123 S/cm."
    )
    out = asyncio.run(
        agent._revise_with_llm(
            previous_draft=prev,
            issues=["Unsupported numeric claim"],
            question="Q",
            generate_time="2026-09-13",
        )
    )
    # The rewrite path returns the LLM's revised text (not a from-scratch regen).
    # _ensure_required_sections may append a default §7, so assert containment.
    assert out.startswith("# Revised")
    assert "The unsupported claim was removed" in out
    # The revision prompt must carry BOTH the prior draft and the issue list.
    prompt = fake.calls[0][1]
    assert "Some unsupported claim of 123 S/cm." in prompt
    assert "Unsupported numeric claim" in prompt


def test_revise_with_llm_falls_back_to_prior_draft_on_error():
    class _BoomLLM:
        async def complete(self, system, prompt, **kwargs):
            raise RuntimeError("boom")

    agent = ReportGenerationAgent(llm=_BoomLLM(), config={})
    prev = "previous draft content that must be preserved"
    out = asyncio.run(
        agent._revise_with_llm(
            previous_draft=prev, issues=["x"], question="Q", generate_time="t"
        )
    )
    # On LLM failure the loop must keep the prior draft instead of crashing.
    assert out == prev
