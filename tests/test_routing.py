"""Tests for the LangGraph conditional routers.

Both routers decide whether the pipeline retries a step, so a wrong return
value silently changes how many LLM calls a run makes (or ends a run with an
unrevised report). They are pure functions of the state, so they are tested
without building the whole graph.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from matresearcher.workflow.engine import MatResearcherWorkflow  # noqa: E402


def _router():
    """Bypass __init__ (which loads YAML and builds every tool client).

    The coverage router reads `max_coverage_retries`, normally set from
    config/workflow.yaml, so it is stubbed here with the shipped default.
    """
    obj = MatResearcherWorkflow.__new__(MatResearcherWorkflow)
    obj.max_coverage_retries = 2
    obj._cache = None  # the retry path invalidates cached nodes when present
    return obj


# ── fact_check → report_generation | END ─────────────────────────────────────

def test_fact_check_routes_to_revision_when_issues_remain():
    assert _router()._route_after_fact_check({"needs_revision": True}) == "revise"


def test_fact_check_ends_when_report_is_clean():
    assert _router()._route_after_fact_check({"needs_revision": False}) == "done"
    assert _router()._route_after_fact_check({}) == "done"


# ── coverage_check → literature_search | llm_prefilter ───────────────────────

def test_coverage_failure_retries_while_budget_remains():
    state = {
        "coverage_check_result": {"passed": False},
        "coverage_retry_count": 1,
    }
    assert _router()._route_after_coverage(state) == "retry_search"


def test_coverage_failure_stops_after_budget_exhausted():
    state = {
        "coverage_check_result": {"passed": False},
        "coverage_retry_count": 2,
    }
    assert _router()._route_after_coverage(state) == "continue"


def test_coverage_success_continues():
    state = {"coverage_check_result": {"passed": True}, "coverage_retry_count": 0}
    assert _router()._route_after_coverage(state) == "continue"
