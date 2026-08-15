"""Literature Filter Agent (Step 5).

Responsibilities:
- Dedup by DOI/title
- Score relevance using Reranker model
- Filter by threshold
- Parseability filter (Step 5.1): drop papers with no full-text access path
  (abstract-only / no path), backfill A/B-grade candidates from the ranking
  pool so downstream PDF parsing never hits a paper without full text.
- Return ranked literature list
"""
from __future__ import annotations
from typing import Optional
from ..state import WorkflowState
from .base import BaseAgent
from ..tools.reranker import RerankerModel


class LiteratureFilterAgent(BaseAgent):
    name = "literature_filter"
    role = "文献筛选 Agent"

    def __init__(self, llm=None, config=None, reranker: Optional[RerankerModel] = None, log_dir=None):
        super().__init__(llm, config, log_dir=log_dir)
        self.reranker = reranker

    async def run(self, state: WorkflowState) -> dict:
        candidates = state.get("candidate_literature", [])
        question = state.get("raw_question", "")
        top_k = self.config.get("rerank_top_k", 10) # 恢复 20（充分模式）；调通流程阶段曾临时降为 10
        min_score = self.config.get("min_relevance_score", 0.5)
        parseable_enabled = self.config.get("parseable_filter", True)
        min_kept = self.config.get("min_parseable_kept", 8)
        pool_extra = self.config.get("parseable_pool_extra", 10)

        # Compute actual score range from Sciverse relevance scores
        sciverse_scores = [
            lit.relevance_score for lit in candidates
            if lit.relevance_score is not None
        ]
        if sciverse_scores:
            actual_min = min(sciverse_scores)
            actual_max = max(sciverse_scores)
            self.log(
                f"Filtering {len(candidates)} candidates, top_k={top_k}, "
                f"rerank_threshold={min_score}, "
                f"Sciverse_score_range=[{actual_min:.4f}, {actual_max:.4f}]"
            )
        else:
            self.log(
                f"Filtering {len(candidates)} candidates, top_k={top_k}, "
                f"rerank_threshold={min_score}, Sciverse_scores=N/A"
            )

        if not candidates:
            self.log("No candidates to filter", "red")
            return {"filtered_literature": []}

        # Build abstract texts for reranking
        abstracts = []
        for lit in candidates:
            text = f"{lit.metadata.title}. {lit.metadata.abstract or ''}"
            abstracts.append(text[:2000])  # truncate to avoid overflow

        # Rerank with an enlarged pool so parseability filtering can backfill.
        # sorted_pool = full ordered ranking (>= min_score); filtered = top_k.
        if self.reranker:
            try:
                pool_size = max(top_k, top_k + pool_extra)
                ranked = self.reranker.rerank(question, abstracts, top_k=pool_size, min_score=min_score)
                sorted_pool = [candidates[idx] for idx, _ in ranked]
                filtered = sorted_pool[:top_k]
                # Attach scores (from the pool ranking, not just top_k)
                for lit, (idx, score) in zip(filtered, ranked[:top_k]):
                    lit.relevance_score = score
            except Exception as e:
                self.log(f"Reranker failed ({e}), falling back to Sciverse scores", "yellow")
                sorted_pool = self._rank_all(candidates)
                filtered = sorted_pool[:top_k]
        else:
            self.log("Reranker not available, falling back to Sciverse scores", "yellow")
            sorted_pool = self._rank_all(candidates)
            filtered = sorted_pool[:top_k]

        # Step 5.1: Parseability filter — drop C/D grades, backfill from pool
        if parseable_enabled and filtered:
            before = len(filtered)
            filtered = self._filter_parseable(filtered, sorted_pool, min_kept)
            if len(filtered) != before:
                self.log(
                    f"Parseability filter: {before} → {len(filtered)} papers "
                    f"(dropped {before - len(filtered)} without full-text path)",
                    "yellow",
                )

        self.log(f"Filtered to {len(filtered)} papers")
        return {"filtered_literature": filtered}

    # ── Parseability grading & filtering (Step 5.1) ──

    @staticmethod
    def _grade_parseability(lit) -> tuple[str, str]:
        """Grade a paper by its full-text access path.

        Returns (grade, reason) where grade ∈ {A, B, C, D}:
          A — doc_id + is_content_accessible=True → Sciverse /content (near-certain)
          B — pdf_url or chunk fallback path exists
          C — abstract-only (doc_id present but /content unavailable and no
              pdf_url/chunk; or no doc_id/pdf_url/chunk at all)
          D — no access path at all (not even an abstract)
        """
        meta = lit.metadata
        has_doc = bool(meta.doc_id and str(meta.doc_id).strip())
        accessible = bool(meta.is_content_accessible)
        has_pdf = bool(meta.pdf_url and str(meta.pdf_url).strip())
        has_chunk = bool(meta.chunk and str(meta.chunk).strip())
        has_abstract = bool(meta.abstract and len(meta.abstract.strip()) > 50)

        if has_doc and accessible:
            return "A", "doc_id + is_content_accessible=True（Sciverse /content）"
        if has_pdf or has_chunk:
            return "B", "pdf_url 或 chunk 兜底路径"
        if has_doc:
            return "C", "is_content_accessible=False 且无 pdf_url/chunk（仅摘要）"
        if has_abstract:
            return "C", "无 doc_id/pdf_url/chunk（仅摘要）"
        return "D", "无任何可解析路径且无摘要"

    def _filter_parseable(
        self, filtered: list, sorted_pool: list, min_kept: int
    ) -> list:
        """Drop C/D-grade papers; backfill A/B-grade papers from the sorted pool.

        Every drop is logged with its grade and reason for traceability.
        Backfilled papers are logged with [backfill] so the report can note
        "N 篇因无可解析路径被剔除，从排序池补充 M 篇" for review-friendliness.
        """
        kept: list = []
        dropped: list[tuple] = []
        for lit in filtered:
            grade, reason = self._grade_parseability(lit)
            if grade in ("A", "B"):
                kept.append(lit)
            else:
                dropped.append((lit, grade, reason))

        # Backfill A/B candidates from the pool if below min_kept
        kept_ids = {lit.id for lit in kept}
        if len(kept) < min_kept:
            for lit in sorted_pool:
                if len(kept) >= min_kept:
                    break
                if lit.id in kept_ids:
                    continue
                grade, reason = self._grade_parseability(lit)
                if grade in ("A", "B"):
                    kept.append(lit)
                    kept_ids.add(lit.id)
                    self.log(
                        f"  [backfill] {lit.id} [{grade}] 从排序池补充 ({reason})",
                        "dim",
                    )

        for lit, grade, reason in dropped:
            self.log(f"  [drop] {lit.id} [{grade}] {reason} — 已剔除", "yellow")

        return kept

    @staticmethod
    def _rank_all(candidates: list) -> list:
        """Full ranking fallback: sort by Sciverse relevance_score descending.

        Papers without a score (e.g. from meta-search) are placed at the end
        and sorted by citation_count. Papers with neither score nor citation_count
        are placed last.
        """
        scored = [lit for lit in candidates if lit.relevance_score is not None]
        unscored = [lit for lit in candidates if lit.relevance_score is None]

        scored.sort(key=lambda lit: lit.relevance_score, reverse=True)

        # Within unscored, prefer papers with higher citation_count
        unscored.sort(
            key=lambda lit: (
                lit.metadata.citation_count is not None,   # has citation_count → True (sorts after False)
                lit.metadata.citation_count or 0,
            ),
            reverse=True,
        )

        return scored + unscored

    @staticmethod
    def _fallback_rank(candidates: list, top_k: int) -> list:
        """Fallback ranking: top-k slice of the full ranking (kept for compat)."""
        return LiteratureFilterAgent._rank_all(candidates)[:top_k]
