"""Literature Search Agent (Step 2).

Responsibilities:
- Phase 1: Per-subtask dimension-specific search (keywords + semantic_query per subtask)
- Phase 2: Reformulated queries — semantically equivalent rewrites (tagged: reformulated)
- Phase 3: LLM query expansion when results are still sparse (tagged: expanded)
- Return candidate literature list with query_source tagging for traceability
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Optional

from ..state import WorkflowState
from .base import BaseAgent, PROMPTS_DIR
from ..tools.sciverse import SciverseClient
from ..tools.doi_enrichment import DOIEnricher
from ..models.literature import Literature, LiteratureMetadata


class LiteratureSearchAgent(BaseAgent):
    name = "literature_search"
    role = "文献检索 Agent"

    def __init__(self, llm=None, config=None, sciverse: Optional[SciverseClient] = None, log_dir=None):
        super().__init__(llm, config, log_dir=log_dir)
        self.sciverse = sciverse

    async def run(self, state: WorkflowState) -> dict:
        strategy = state.get("search_strategy", {})
        subtasks = state.get("subtasks", [])
        top_k = self.config.get("sciverse_top_k", 50)
        query_expansion = self.config.get("query_expansion", True)
        self.log(f"Query expansion: {query_expansion}, top_k={top_k}")

        all_results: list[dict] = []
        phase_counts: dict[str, int] = {}

        if self.sciverse:
            # ── Phase 1: Per-subtask dimension-specific search ──
            if subtasks:
                self.log(f"Phase 1: Searching {len(subtasks)} subtasks by dimension")
                for st in subtasks:
                    st_kw = st.get("keywords", [])
                    st_sq = st.get("semantic_query", "")
                    st_filters = self._convert_subtask_filters(st.get("filters", {}))
                    st_id = st.get("id", "?")
                    st_dim = st.get("dimension", "?")

                    if not st_sq and not st_kw:
                        self.log(f"  Subtask {st_id} ({st_dim}): no query/keywords, skipping", "dim")
                        continue

                    self.log(f"  Subtask {st_id} ({st_dim}): sq='{st_sq[:50]}', kw={st_kw}")
                    results = await self._safe_search(
                        query=st_sq or " ".join(st_kw),
                        keywords=st_kw,
                        top_k=top_k,
                        filters=st_filters,
                    )
                    for r in results:
                        r["_query_source"] = "original"
                    all_results.extend(results)
                    self.log(f"    → {len(results)} results")

                # Phase 1 stage summary
                p1 = len(all_results)
                phase_counts["phase1_subtask"] = p1
                self.log(f"Phase 1 total: {p1} results across {len(subtasks)} subtasks")

            keywords = strategy.get("primary_keywords", [])
            filters = strategy.get("filters", {})

            # ── Phase 2: Reformulated queries (semantically equivalent rewrites) ──
            reformulated_queries = strategy.get("reformulated_queries", [])
            if reformulated_queries:
                self.log(f"Phase 2: Reformulated queries ({len(reformulated_queries)} rewrites)")
                p2_before = len(all_results)
                for rq in reformulated_queries:
                    results = await self._safe_semantic_search(
                        rq, top_k=top_k, filters=filters
                    )
                    for r in results:
                        r["_query_source"] = "reformulated"
                    all_results.extend(results)
                    self.log(f"  [reformulated] '{rq[:50]}...' → {len(results)} results", "dim")
                p2_count = len(all_results) - p2_before
                phase_counts["phase2_reformulated"] = p2_count
                self.log(f"Phase 2 total: {p2_count} results")

            # ── Phase 3: LLM expansion (when coverage still low and LLM available) ──
            if query_expansion and self.llm and len(all_results) < top_k:
                self.log(f"Coverage low ({len(all_results)} < {top_k}), expanding queries via LLM")
                p3_before = len(all_results)
                semantic_queries = strategy.get("semantic_queries", [])
                expanded = await self._expand_queries(semantic_queries, keywords)
                for eq in expanded:
                    results = await self._safe_semantic_search(eq, top_k=top_k, filters=filters)
                    for r in results:
                        r["_query_source"] = "expanded"
                    all_results.extend(results)
                    self.log(f"  [expanded] '{eq[:60]}' → {len(results)} results", "dim")
                p3_count = len(all_results) - p3_before
                phase_counts["phase3_expanded"] = p3_count
                self.log(f"Phase 3 total: {p3_count} results")
        else:
            # Wording matters: nothing here fabricates data (all_results stays
            # empty and the run yields zero candidates), but the old message
            # said "using mock results", which reads as fabricated literature
            # and contradicts the "zero mock data" claim in the proposal.
            self.log(
                "Sciverse API not configured — no literature can be retrieved "
                "(set SCIVERSE_API_KEY in ~/.matresearcher/secrets.env); "
                "returning an empty candidate list",
                "yellow",
            )

        # Convert to Literature objects (passes _query_source through)
        candidate_literature = self._to_literature(all_results)

        # ── Title→DOI enrichment ──
        # Sciverse /agentic-search returns a null DOI for the majority of hits, so
        # recover resolvable DOIs from the paper title (Crossref primary, Sciverse
        # meta-search secondary) before dedup + persistence. Best-effort: a failure
        # leaves records unchanged and the run continues.
        if self.config.get("doi_enrichment", True):
            enricher = DOIEnricher(
                sciverse=self.sciverse,
                timeout=self.config.get("doi_enrichment_timeout", 10.0),
                use_sciverse_meta=self.config.get("doi_enrichment_sciverse_meta", True),
                crossref_email=self.config.get("doi_enrichment_email"),
            )
            try:
                await enricher.enrich(candidate_literature)
                n_doi = sum(1 for l in candidate_literature if l.metadata.doi)
                self.log(
                    f"DOI enrichment: {n_doi}/{len(candidate_literature)} candidates "
                    f"now carry a resolvable DOI"
                )
            except Exception as e:  # noqa: BLE001 - enrichment must never abort the run
                self.log(f"  DOI enrichment failed (best-effort, continuing): {e}", "yellow")
            finally:
                await enricher.close()

        # Dedup by DOI/title (prefer original over reformulated/expanded, and
        # always keep the copy that actually carries a DOI)
        before_dedup = len(candidate_literature)
        candidate_literature = self._dedup(candidate_literature)
        removed = before_dedup - len(candidate_literature)
        if removed > 0:
            self.log(f"Deduplication: {before_dedup} → {len(candidate_literature)} ({removed} duplicates removed)")

        # Summary stats
        src_counts: dict[str, int] = {}
        for lit in candidate_literature:
            s = lit.metadata.query_source
            src_counts[s] = src_counts.get(s, 0) + 1
        self.log(
            f"Final: {len(candidate_literature)} candidates "
            f"(source: {' | '.join(f'{k}={v}' for k, v in sorted(src_counts.items()))})"
        )
        return {"candidate_literature": candidate_literature}

    @staticmethod
    def _convert_subtask_filters(filters: dict) -> dict:
        """Convert subtask filter format (year_start/year_end) to Sciverse format (year_from/year_to)."""
        result: dict = {}
        if "year_start" in filters:
            result["year_from"] = filters["year_start"]
        if "year_end" in filters:
            result["year_to"] = filters["year_end"]
        for key in ("lang", "language", "journal"):
            if key in filters:
                result[key] = filters[key]
        return result

    async def _safe_search(self, query: str, keywords: list[str], top_k: int,
                           filters: dict, *, max_retries: int = 3) -> list[dict]:
        """Search with per-query retry on transient errors."""
        for attempt in range(max_retries):
            try:
                return await self.sciverse.hybrid_search(
                    query=query, keywords=keywords, top_k=top_k, filters=filters,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    self.log(f"  Retry {attempt+1}/{max_retries} in {delay}s: {e}", "dim")
                    await asyncio.sleep(delay)
                else:
                    self.log(f"  Search failed after {max_retries} retries: {e}", "yellow")
                    return []

    async def _safe_semantic_search(self, query: str, top_k: int,
                                    filters: dict, *, max_retries: int = 3) -> list[dict]:
        """Semantic search with per-query retry on transient errors."""
        for attempt in range(max_retries):
            try:
                return await self.sciverse.semantic_search(
                    query, top_k=top_k, filters=filters,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    self.log(f"  Retry {attempt+1}/{max_retries} in {delay}s: {e}", "dim")
                    await asyncio.sleep(delay)
                else:
                    self.log(f"  Semantic search failed after {max_retries} retries: {e}", "yellow")
                    return []

    async def _expand_queries(self, queries: list[str], keywords: list[str]) -> list[str]:
        """Use LLM to generate expanded search queries from prompt template."""
        try:
            prompt_path = PROMPTS_DIR / "query_expansion.txt"
            template = Path(prompt_path).read_text(encoding="utf-8")
            topic_hint = queries[0] if queries else "solid-state battery materials"
            user = template.format(
                topic=topic_hint,
                existing_queries="\n".join(f"- {q}" for q in queries),
                existing_keywords=", ".join(keywords),
            )
            result = await self.llm.complete_json(
                "You are a materials science query expansion expert. "
                "Return ONLY a valid JSON array of strings.",
                user,
                stage=self.name,
            )
            # Handle dict-wrapped response (LLM sometimes returns {"queries": [...]})
            if isinstance(result, list):
                expanded = result
            elif isinstance(result, dict):
                expanded = result.get("queries", result.get("expanded_queries", []))
                if not isinstance(expanded, list):
                    self.log("  Query expansion returned unexpected dict format", "yellow")
                    expanded = []
            else:
                self.log(f"  Query expansion returned unexpected type {type(result)}", "yellow")
                expanded = []
            for i, eq in enumerate(expanded):
                self.log(f"  Expanded query #{i+1}: {eq}", "dim")
            return expanded
        except Exception as e:
            self.log(f"  Query expansion failed: {e}", "yellow")
            return []

    def _to_literature(self, results: list[dict]) -> list[Literature]:
        """Convert Sciverse API results to Literature objects.

        Carries _query_source through to LiteratureMetadata.query_source.
        """
        literature = []
        for i, r in enumerate(results):
            query_src = r.get("_query_source", "original")
            lit = Literature(
                id=f"lit_{i:04d}",
                relevance_score=r.get("score"),  # Sciverse raw relevance score (agentic-search); meta-search has no score
                metadata=LiteratureMetadata(
                    doi=_extract_doi(r),
                    title=r.get("title", ""),
                    authors=r.get("authors", []),
                    journal=r.get("journal"),
                    year=r.get("year"),
                    abstract=r.get("abstract"),
                    keywords=r.get("keywords", []),
                    pdf_url=r.get("pdf_url"),
                    doc_id=r.get("doc_id"),
                    chunk=r.get("chunk"),
                    is_content_accessible=r.get("is_content_accessible", True),
                    citation_count=r.get("citation_count"),
                    source="sciverse",
                    query_source=query_src,
                ),
            )
            literature.append(lit)
        return literature

    def _dedup(self, literature: list[Literature]) -> list[Literature]:
        """Deduplicate by title (preferring the copy that carries a DOI).

        Sciverse returns the same paper from multiple query sources, and the
        agentic-search copy usually has ``doi=null`` while the meta-search copy
        carries the real DOI. Keying on ``doi if present else title`` used to let
        both copies survive as different keys, so the DOI-bearing copy was never
        preferred. We now collapse by normalized title and always keep the copy
        that has a DOI (then higher-priority source, then higher relevance).
        """
        # Priority order for query_source
        _SRC_PRIORITY = {"original": 0, "reformulated": 1, "expanded": 2, "fallback": 3}

        def quality(lit: Literature) -> tuple:
            m = lit.metadata
            return (
                0 if m.doi else 1,                     # 1) prefer a copy WITH a DOI
                _SRC_PRIORITY.get(m.query_source, 99),  # 2) prefer original source
                -(lit.relevance_score or 0.0),          # 3) prefer higher relevance
            )

        by_title: dict[str, Literature] = {}
        no_title: dict[str, Literature] = {}  # keyed by DOI when title is absent
        for lit in literature:
            m = lit.metadata
            title = (m.title or "").lower().strip()
            if title:
                cur = by_title.get(title)
                if cur is None or quality(lit) < quality(cur):
                    by_title[title] = lit
            elif m.doi:
                no_title.setdefault(m.doi, lit)

        survivors = list(by_title.values())
        # A title-less, DOI-bearing copy is redundant if its DOI already appears
        # among the title-keyed survivors (same paper, just missing a title).
        title_dois = {lit.metadata.doi for lit in survivors if lit.metadata.doi}
        merged = list(survivors)
        for lit in no_title.values():
            if lit.metadata.doi in title_dois:
                continue
            merged.append(lit)
        return merged

def _extract_doi(raw: dict) -> Optional[str]:
    """Robustly extract a real DOI from a Sciverse/Crossref-style API record.

    Sciverse responses are inconsistent: some carry a top-level ``doi``, others
    nest it (e.g. ``externalIds.DOI``), and many only expose a ``pdf_url`` /
    ``link`` whose path contains the ``10.xxxx/...`` DOI string. Missing DOIs
    would otherwise leave evidence verification unable to cite a resolvable
    reference, so we triangulate from every plausible source.
    """
    if not isinstance(raw, dict):
        return None

    # 1. Direct top-level keys
    for key in ("doi", "DOI", "Doi"):
        v = raw.get(key)
        if isinstance(v, str) and v.strip():
            return _normalize_doi(v)

    # 2. Nested external-id blocks (Semantic Scholar / Crossref conventions)
    for container in ("externalIds", "identifiers", "external_ids"):
        obj = raw.get(container)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and k.upper() in ("DOI", "DOI_URL") and isinstance(v, str) and v.strip():
                    return _normalize_doi(v)
        elif isinstance(obj, list):  # e.g. [{"type": "DOI", "value": "10.x/..."}]
            for item in obj:
                if isinstance(item, dict) and str(item.get("type", "")).upper() == "DOI":
                    val = item.get("value") or item.get("id")
                    if isinstance(val, str) and val.strip():
                        return _normalize_doi(val)

    # 3. Parse from any URL-like field that embeds a DOI
    for key in ("pdf_url", "url", "link", "links", "full_text_url", "source_url"):
        v = raw.get(key)
        candidates = v if isinstance(v, list) else [v]
        for c in candidates:
            if not isinstance(c, str):
                continue
            d = _normalize_doi(c)
            if d:
                return d
    return None


_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+")


def _normalize_doi(value: str) -> Optional[str]:
    """Strip resolver prefixes / whitespace and validate the DOI shape."""
    if not value:
        return None
    s = value.strip().rstrip(".")
    # Drop common resolver prefixes: https://doi.org/, dx.doi.org/, doi:
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?i)^doi[:\s]+", "", s)
    s = s.strip()
    if not _DOI_RE.match(s):
        return None
    return s
