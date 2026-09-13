"""Real baseline implementations for MatResearcher evaluation.

History — why this module exists
--------------------------------
The original ``scripts/evaluate.py`` shipped four baselines whose bodies were
``_generate_mock_summary(...)`` returning ``status="mock"`` and a **hardcoded**
``gap_count`` (0 / 0 / 1 / 2).  ``compute_comparison`` then divided by those
numbers to print a ``gap_advantage`` percentage.  That produced quantitative
comparison claims out of thin air — worse than shipping no baselines at all.

Every baseline here does real work:

  keyword  : Sciverse ``/meta-search``                → LLM summary
  semantic : Sciverse ``/agentic-search``             → LLM summary
  hybrid   : both, merged and deduped by DOI/title    → LLM summary
  rag      : hybrid → chunk → embed → cosine top-k    → LLM summary

and ``gap_count`` is never hardcoded: it is counted by
:func:`count_gaps_in_summary`, which asks the LLM to enumerate the research
gaps stated in each baseline's own output.  Any baseline that fails (missing
API key, network error, empty result) is recorded with ``status="error"`` and
is **excluded** from every derived comparison number.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

from ..tools.llm import LLMClient
from ..tools.sciverse import SciverseClient
from ..tools.embedding import EmbeddingModel

SUMMARY_SYSTEM = (
    "You are a materials-science researcher writing a short literature survey. "
    "Use ONLY the provided literature context. Do not invent numbers or citations."
)

SUMMARY_USER_TEMPLATE = """Research question:
{question}

Literature context ({n_papers} papers):
{context}

Write a concise literature survey (600-1200 words) with:
1. Key findings relevant to the question
2. Quantitative values you can see in the context (with the paper they came from)
3. Research gaps / open questions you can identify from the context

If the context is insufficient, say so explicitly instead of speculating.
"""

GAP_COUNT_SYSTEM = (
    "You extract research gaps from text. Reply with a numbered list, one gap per "
    "line, and nothing else. If there are no gaps, reply with the single word NONE."
)

GAP_COUNT_USER_TEMPLATE = """Text:
\"\"\"
{text}
\"\"\"

List the distinct research gaps / open questions explicitly stated in the text
above. Numbered list, one per line. If none, reply NONE."""


@dataclass
class BaselineResult:
    """Outcome of one baseline run."""

    method: str
    status: str                      # "ok" | "error" | "skipped"
    elapsed_seconds: float = 0.0
    report_length: int = 0
    papers_retrieved: int = 0
    gap_count: Optional[int] = None  # None = not measured (never fabricated)
    summary: str = ""
    error: Optional[str] = None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_keywords(question: str, max_kw: int = 8) -> list[str]:
    """Cheap keyword extraction: drop stopwords, keep content tokens."""
    stop = {
        "the", "a", "an", "of", "for", "and", "or", "in", "on", "to", "is", "are",
        "what", "which", "how", "does", "do", "current", "status", "research",
        "please", "about", "with", "by", "from", "at", "as", "be", "can", "values",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", question)
    seen: list[str] = []
    for t in tokens:
        low = t.lower()
        if low in stop or low in seen:
            continue
        seen.append(t)
        if len(seen) >= max_kw:
            break
    return seen or question.split()[:max_kw]


def _format_context(hits: list[dict], max_papers: int = 12, chars_each: int = 700) -> str:
    """Render search hits into a compact LLM context block."""
    parts: list[str] = []
    for i, h in enumerate(hits[:max_papers], 1):
        title = (h.get("title") or "Untitled").strip()
        year = h.get("year") or "n/a"
        body = (h.get("abstract") or h.get("chunk") or "").strip()
        if len(body) > chars_each:
            body = body[:chars_each] + "..."
        parts.append(f"[{i}] {title} ({year})\n{body}")
    return "\n\n".join(parts)


def _chunk(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += max(1, size - overlap)
    return chunks


async def count_gaps_in_summary(llm: LLMClient, text: str) -> Optional[int]:
    """Count research gaps stated in ``text`` using the LLM.

    Returns None when the count could not be obtained — callers must treat
    None as "unknown", never as 0.
    """
    if not llm or not text.strip():
        return None
    try:
        raw = await llm.complete(
            system=GAP_COUNT_SYSTEM,
            user=GAP_COUNT_USER_TEMPLATE.format(text=text[:8000]),
            temperature=0.0,
            max_tokens=1024,
            stage="evaluation_gap_count",
        )
    except Exception:
        return None

    if not raw:
        return None
    stripped = raw.strip()
    if stripped.upper().startswith("NONE"):
        return 0
    # Count numbered lines like "1. ..." / "1) ..." / "- ..."
    numbered = re.findall(r"^\s*(?:\d+[.)、]|-)\s+\S", stripped, flags=re.MULTILINE)
    if numbered:
        return len(numbered)
    # Fallback: non-empty bullet-ish lines
    lines = [ln for ln in stripped.splitlines() if len(ln.strip()) > 10]
    return len(lines) if lines else 0


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

async def _summarise(
    llm: LLMClient, question: str, hits: list[dict]
) -> tuple[str, int]:
    context = _format_context(hits)
    summary = await llm.complete(
        system=SUMMARY_SYSTEM,
        user=SUMMARY_USER_TEMPLATE.format(
            question=question, n_papers=len(hits), context=context or "(no context retrieved)"
        ),
        temperature=0.3,
        max_tokens=4096,
        stage="evaluation_baseline_summary",
    )
    return summary or "", len(context)


async def run_keyword_baseline(
    question: str, sciverse: SciverseClient, llm: LLMClient, top_k: int = 30
) -> BaselineResult:
    """Keyword metadata search → LLM direct summary (no structured extraction)."""
    start = time.time()
    try:
        keywords = _extract_keywords(question)
        hits = await sciverse.keyword_search(keywords=keywords, top_k=top_k)
        summary, _ = await _summarise(llm, question, hits)
        gaps = await count_gaps_in_summary(llm, summary)
        return BaselineResult(
            method="keyword_search + LLM summary",
            status="ok",
            elapsed_seconds=round(time.time() - start, 1),
            report_length=len(summary),
            papers_retrieved=len(hits),
            gap_count=gaps,
            summary=summary[:500],
            detail={"keywords": keywords},
        )
    except Exception as e:  # noqa: BLE001 - baselines must never kill the run
        return BaselineResult(
            method="keyword_search + LLM summary",
            status="error",
            elapsed_seconds=round(time.time() - start, 1),
            error=f"{type(e).__name__}: {e}",
        )


async def run_semantic_baseline(
    question: str, sciverse: SciverseClient, llm: LLMClient, top_k: int = 30
) -> BaselineResult:
    """Semantic evidence search → LLM direct summary."""
    start = time.time()
    try:
        hits = await sciverse.semantic_search(query=question, top_k=top_k)
        summary, _ = await _summarise(llm, question, hits)
        gaps = await count_gaps_in_summary(llm, summary)
        return BaselineResult(
            method="semantic_search + LLM summary",
            status="ok",
            elapsed_seconds=round(time.time() - start, 1),
            report_length=len(summary),
            papers_retrieved=len(hits),
            gap_count=gaps,
            summary=summary[:500],
        )
    except Exception as e:  # noqa: BLE001
        return BaselineResult(
            method="semantic_search + LLM summary",
            status="error",
            elapsed_seconds=round(time.time() - start, 1),
            error=f"{type(e).__name__}: {e}",
        )


async def run_hybrid_baseline(
    question: str, sciverse: SciverseClient, llm: LLMClient, top_k: int = 30
) -> BaselineResult:
    """Hybrid (semantic + keyword) search → LLM direct summary."""
    start = time.time()
    try:
        keywords = _extract_keywords(question)
        hits = await sciverse.hybrid_search(
            query=question, keywords=keywords, top_k=top_k
        )
        summary, _ = await _summarise(llm, question, hits)
        gaps = await count_gaps_in_summary(llm, summary)
        return BaselineResult(
            method="hybrid_search + LLM summary",
            status="ok",
            elapsed_seconds=round(time.time() - start, 1),
            report_length=len(summary),
            papers_retrieved=len(hits),
            gap_count=gaps,
            summary=summary[:500],
        )
    except Exception as e:  # noqa: BLE001
        return BaselineResult(
            method="hybrid_search + LLM summary",
            status="error",
            elapsed_seconds=round(time.time() - start, 1),
            error=f"{type(e).__name__}: {e}",
        )


async def run_rag_baseline(
    question: str,
    sciverse: SciverseClient,
    llm: LLMClient,
    embedding: Optional[EmbeddingModel] = None,
    top_k: int = 30,
) -> BaselineResult:
    """Single-agent RAG: hybrid search → chunk → embed → cosine retrieve → LLM."""
    start = time.time()
    try:
        keywords = _extract_keywords(question)
        hits = await sciverse.hybrid_search(
            query=question, keywords=keywords, top_k=top_k
        )

        chunks: list[str] = []
        for h in hits:
            body = (h.get("abstract") or h.get("chunk") or "").strip()
            title = h.get("title") or ""
            chunks.extend(_chunk(f"{title}. {body}" if title else body))

        if embedding is not None and chunks:
            try:
                q_vec = embedding.embed(question)
                doc_vecs = embedding.embed_batch(chunks[:200])
                scored = [
                    (float(embedding.similarity(q_vec, v)), c)
                    for v, c in zip(doc_vecs, chunks[:200])
                ]
                scored.sort(key=lambda x: x[0], reverse=True)
                context_chunks = [c for _, c in scored[:12]]
            except Exception:
                context_chunks = chunks[:12]
        else:
            context_chunks = chunks[:12]

        context = "\n\n".join(f"- {c}" for c in context_chunks)
        summary = await llm.complete(
            system=SUMMARY_SYSTEM,
            user=SUMMARY_USER_TEMPLATE.format(
                question=question,
                n_papers=len(hits),
                context=context or "(no context retrieved)",
            ),
            temperature=0.3,
            max_tokens=4096,
            stage="evaluation_baseline_summary",
        )
        summary = summary or ""
        gaps = await count_gaps_in_summary(llm, summary)
        return BaselineResult(
            method="RAG_single_agent",
            status="ok",
            elapsed_seconds=round(time.time() - start, 1),
            report_length=len(summary),
            papers_retrieved=len(hits),
            gap_count=gaps,
            summary=summary[:500],
            detail={"chunks_indexed": len(chunks), "chunks_used": len(context_chunks)},
        )
    except Exception as e:  # noqa: BLE001
        return BaselineResult(
            method="RAG_single_agent",
            status="error",
            elapsed_seconds=round(time.time() - start, 1),
            error=f"{type(e).__name__}: {e}",
        )
