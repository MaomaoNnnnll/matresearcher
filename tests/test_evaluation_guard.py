"""Guards for the evaluation harness.

These tests exist because the previous harness shipped baselines whose
``gap_count`` was hardcoded (0/0/1/2) and then divided by those numbers to
print a "gap advantage" percentage — i.e. it manufactured quantitative claims.
Any regression that reintroduces derived numbers from unmeasured runs must fail
here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from matresearcher.evaluation import compute_comparison  # noqa: E402
from matresearcher.evaluation.baselines import BaselineResult, _extract_keywords  # noqa: E402


def _mr(gaps, status="ok"):
    return {"status": status, "gap_count": gaps, "conflict_count": 0,
            "elapsed_seconds": 1.0, "report_length": 100}


def test_no_derived_number_when_baseline_failed():
    results = {
        "matresearcher": _mr(4),
        "baselines": {"keyword": {"status": "error", "gap_count": None,
                                  "papers_retrieved": 0, "elapsed_seconds": 0.1,
                                  "report_length": 0}},
    }
    comparison = compute_comparison(results)
    entry = comparison["baselines"]["keyword"]
    assert entry["gap_advantage_pct"] is None
    assert "gap_delta" not in entry
    assert comparison["comparable"] is False
    assert comparison["warning"]


def test_no_derived_number_when_gap_count_unmeasured():
    results = {
        "matresearcher": _mr(4),
        "baselines": {"semantic": {"status": "ok", "gap_count": None,
                                   "papers_retrieved": 10, "elapsed_seconds": 2.0,
                                   "report_length": 900}},
    }
    comparison = compute_comparison(results)
    assert comparison["baselines"]["semantic"]["gap_advantage_pct"] is None
    assert comparison["comparable"] is False


def test_derived_number_present_when_both_measured():
    results = {
        "matresearcher": _mr(4),
        "baselines": {"hybrid": {"status": "ok", "gap_count": 2,
                                 "papers_retrieved": 20, "elapsed_seconds": 3.0,
                                 "report_length": 1200}},
    }
    comparison = compute_comparison(results)
    entry = comparison["baselines"]["hybrid"]
    assert entry["gap_advantage_pct"] == pytest.approx(100.0)
    assert entry["gap_delta"] == 2
    assert comparison["comparable"] is True


def test_zero_gap_baseline_does_not_produce_infinite_advantage():
    results = {
        "matresearcher": _mr(3),
        "baselines": {"rag": {"status": "ok", "gap_count": 0,
                              "papers_retrieved": 5, "elapsed_seconds": 1.0,
                              "report_length": 800}},
    }
    entry = compute_comparison(results)["baselines"]["rag"]
    # bl_gaps == 0 → no percentage (the old code emitted float('inf')).
    assert "gap_advantage_pct" not in entry
    assert entry["gap_delta"] == 3


def test_baseline_status_is_never_mock():
    res = BaselineResult(method="keyword_search + LLM summary", status="ok")
    assert res.status in {"ok", "error", "skipped"}
    assert res.gap_count is None  # unmeasured by default, never 0-by-assumption


def test_keyword_extraction_drops_stopwords():
    kws = _extract_keywords("What are the ionic conductivity values of LLZO garnet?")
    assert "LLZO" in kws
    assert "what" not in [k.lower() for k in kws]
    assert kws, "keyword extraction must never return an empty query"
