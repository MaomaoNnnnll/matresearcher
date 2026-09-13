"""Evaluation harness: run MatResearcher against real baselines.

Guarantees enforced here (these are the point of the module):

1. No fabricated metrics.  A baseline that fails is recorded as
   ``status="error"`` and is excluded from every derived comparison number.
2. ``gap_advantage`` is only computed when BOTH sides report a measured
   ``gap_count`` (never ``None``).  Otherwise the key is omitted entirely and
   ``comparison["comparable"]`` is False.
3. ``baseline_mode`` is echoed into the output so a reader can tell at a glance
   whether the numbers came from real runs.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .baselines import (
    BaselineResult,
    run_keyword_baseline,
    run_semantic_baseline,
    run_hybrid_baseline,
    run_rag_baseline,
)

__all__ = [
    "run_evaluation",
    "compute_comparison",
    "print_comparison",
    "build_clients",
]

DEFAULT_QUESTION = (
    "What are the ionic conductivity values of LLZO garnet electrolytes?"
)


def build_clients(config: dict | None = None) -> dict[str, Any]:
    """Instantiate the shared tool clients used by the baselines."""
    from ..tools.llm import LLMClient
    from ..tools.sciverse import SciverseClient

    cfg = config or {}
    llm_cfg = cfg.get("agents", {}).get("default_llm", {})

    llm = LLMClient(
        api_base=llm_cfg.get("api_base") or os.getenv("LLM_API_BASE"),
        api_key=llm_cfg.get("api_key") or os.getenv("LLM_API_KEY"),
        model=llm_cfg.get("model") or os.getenv("LLM_MODEL"),
        temperature=llm_cfg.get("temperature", 0.3),
    )
    sciverse = SciverseClient(
        api_base=cfg.get("sciverse_api_base"),
        api_key=cfg.get("sciverse_api_key"),
    )

    # RAG needs an encoder; it is optional so the other baselines still run
    # when the local model is unavailable in the environment.
    embedding = None
    if os.getenv("EVAL_ENABLE_EMBEDDING", "1") not in {"0", "false", "False"}:
        try:
            from ..tools.embedding import EmbeddingModel

            embedding = EmbeddingModel(
                model_name=cfg.get("embedding_model", "BAAI/bge-m3")
            )
        except Exception as e:  # noqa: BLE001
            print(f"[eval] embedding model unavailable, RAG falls back to lexical: {e}")

    return {"llm": llm, "sciverse": sciverse, "embedding": embedding}


async def _run_baselines(
    question: str, which: list[str], clients: dict[str, Any], top_k: int
) -> dict[str, dict]:
    llm = clients["llm"]
    sciverse = clients["sciverse"]
    embedding = clients.get("embedding")

    dispatch = {
        "keyword": lambda: run_keyword_baseline(question, sciverse, llm, top_k),
        "semantic": lambda: run_semantic_baseline(question, sciverse, llm, top_k),
        "hybrid": lambda: run_hybrid_baseline(question, sciverse, llm, top_k),
        "rag": lambda: run_rag_baseline(question, sciverse, llm, embedding, top_k),
    }

    out: dict[str, dict] = {}
    for name in which:
        fn = dispatch.get(name)
        if fn is None:
            print(f"[eval] unknown baseline: {name}")
            continue
        print(f"\n{'=' * 40}\nRunning baseline: {name}\n{'=' * 40}")
        res: BaselineResult = await fn()
        out[name] = res.to_dict()
        flag = {"ok": "OK", "error": "FAILED"}.get(res.status, res.status)
        print(f"  {name}: {flag} gaps={res.gap_count} papers={res.papers_retrieved}")
        if res.error:
            print(f"    reason: {res.error}")
    return out


async def run_evaluation(
    question: str,
    baseline: str = "all",
    output_path: str = "evaluation_results.json",
    config: dict | None = None,
    run_matresearcher: bool = True,
) -> dict:
    """Run baselines (and optionally the full pipeline) and persist results."""
    clients = build_clients(config)
    which = (
        ["keyword", "semantic", "hybrid", "rag"]
        if baseline == "all"
        else [b.strip() for b in baseline.split(",") if b.strip()]
    )

    results: dict[str, Any] = {
        "question": question,
        "timestamp": datetime.now().isoformat(),
        "baseline_mode": "real",          # never "mock" — see module docstring
        "baselines": await _run_baselines(question, which, clients, top_k=30),
        "matresearcher": None,
    }

    if run_matresearcher:
        print(f"\n{'=' * 40}\nRunning MatResearcher (multi-agent)\n{'=' * 40}")
        start = time.time()
        try:
            from ..workflow.engine import MatResearcherWorkflow

            workflow = MatResearcherWorkflow()
            mat_result = await workflow.run(question)
            elapsed = time.time() - start

            report = mat_result.get("final_report", mat_result.get("draft_report", ""))
            gaps = mat_result.get("scored_gaps", mat_result.get("gaps", []))
            conflicts = mat_result.get("conflicts", [])
            step_log = mat_result.get("step_log", [])

            results["matresearcher"] = {
                "status": "ok",
                "elapsed_seconds": round(elapsed, 1),
                "report_length": len(report),
                "gap_count": len(gaps),
                "conflict_count": len(conflicts),
                "step_count": len(step_log),
                "fact_check_status": mat_result.get("fact_check_status"),
                "top_gaps": [
                    {
                        "description": (
                            g.description[:120]
                            if hasattr(g, "description")
                            else g.get("description", "")[:120]
                        ),
                        "score": (
                            g.score.total_score if hasattr(g, "score") and g.score else 0
                        ),
                        "verification": getattr(
                            g, "verification_status", "pending"
                        ),
                    }
                    for g in gaps[:5]
                ],
            }
        except Exception as e:  # noqa: BLE001
            results["matresearcher"] = {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_seconds": round(time.time() - start, 1),
            }

    results["comparison"] = compute_comparison(results)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out.absolute()}")
    print_comparison(results["comparison"])
    return results


def compute_comparison(results: dict) -> dict:
    """Derive comparison numbers — only from measured, successful runs."""
    mr = results.get("matresearcher") or {}
    baselines = results.get("baselines") or {}

    mr_ok = mr.get("status") == "ok"
    mr_gaps = mr.get("gap_count") if mr_ok else None

    per_baseline: dict[str, dict] = {}
    comparable = True
    for name, bl in baselines.items():
        entry = {
            "status": bl.get("status"),
            "papers_retrieved": bl.get("papers_retrieved"),
            "elapsed_seconds": bl.get("elapsed_seconds"),
            "report_length": bl.get("report_length"),
            "gap_count": bl.get("gap_count"),
        }
        bl_gaps = bl.get("gap_count")
        if bl.get("status") == "ok" and isinstance(bl_gaps, int) and isinstance(mr_gaps, int):
            entry["gap_delta"] = mr_gaps - bl_gaps
            if bl_gaps > 0:
                entry["gap_advantage_pct"] = round(
                    (mr_gaps - bl_gaps) / bl_gaps * 100, 1
                )
        else:
            # Either side unmeasured → no derived number at all.
            comparable = False
            entry["gap_advantage_pct"] = None
            entry["note"] = (
                "baseline failed" if bl.get("status") != "ok" else "gap_count not measured"
            )
        per_baseline[name] = entry

    comparison = {
        "comparable": comparable and mr_ok and bool(baselines),
        "matresearcher": {
            "status": mr.get("status"),
            "gaps_found": mr.get("gap_count") if mr_ok else None,
            "conflicts_found": mr.get("conflict_count") if mr_ok else None,
            "elapsed_seconds": mr.get("elapsed_seconds"),
            "report_length": mr.get("report_length"),
        },
        "baselines": per_baseline,
    }
    if not comparison["comparable"]:
        comparison["warning"] = (
            "Not all runs succeeded — derived comparison numbers "
            "(gap_advantage_pct) are omitted or partial. Do not report these "
            "results as a complete benchmark."
        )
    return comparison


def print_comparison(comparison: dict):
    mr = comparison.get("matresearcher", {})
    print(f"\n{'=' * 60}\nEVALUATION RESULTS\n{'=' * 60}")

    print("\nMatResearcher (multi-agent):")
    print(f"  Status:    {mr.get('status')}")
    print(f"  Gaps:      {mr.get('gaps_found', 'N/A')}")
    print(f"  Conflicts: {mr.get('conflicts_found', 'N/A')}")
    print(f"  Time:      {mr.get('elapsed_seconds', 'N/A')}s")
    print(f"  Report:    {mr.get('report_length', 'N/A')} chars")

    print("\nBaselines:")
    for name, bl in comparison.get("baselines", {}).items():
        print(f"  {name} [{bl.get('status')}]:")
        print(f"    Papers: {bl.get('papers_retrieved', 'N/A')}")
        print(f"    Gaps:   {bl.get('gap_count', 'N/A')}")
        print(f"    Time:   {bl.get('elapsed_seconds', 'N/A')}s")
        if bl.get("gap_advantage_pct") is not None:
            print(f"    Gap advantage vs MatResearcher: {bl['gap_advantage_pct']:+.1f}%")
        elif bl.get("note"):
            print(f"    (no derived number: {bl['note']})")

    if not comparison.get("comparable"):
        print(f"\n[WARNING] {comparison.get('warning')}")
