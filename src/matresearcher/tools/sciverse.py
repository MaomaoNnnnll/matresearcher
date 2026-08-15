"""Sciverse API client for literature search and full-text location.

Sciverse provides:
- 4.66 billion academic metadata records
- 28 million+ full-text evidence fragments
- Semantic evidence retrieval + structured metadata search + full-text content

API docs: https://sciverse.space/docs
Base URL:  https://api.sciverse.space
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Optional

import httpx
from tenacity import retry, retry_base, stop_after_attempt, wait_exponential

# ── Retry helpers (tenacity 9.x compatible: retry= accepts retry_base subclasses only) ─

class _retry_transient_http(retry_base):
    """Retry on transient HTTP errors (5xx, network), skip client errors (4xx)."""
    def __call__(self, retry_state):
        if not retry_state.outcome.failed:
            return False
        exc = retry_state.outcome.exception()
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code not in (400, 401, 403, 404, 429)
        return True  # retry non-HTTP errors (network, timeout, etc.)


class _retry_on_429(retry_base):
    """Retry ONLY on HTTP 429 (Too Many Requests)."""
    def __call__(self, retry_state):
        if not retry_state.outcome.failed:
            return False
        exc = retry_state.outcome.exception()
        return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


_SEMANTIC_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=15),
    retry=_retry_transient_http(),
)

_META_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=15),
    retry=_retry_transient_http(),
)

_CONTENT_RETRY = retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(min=3, max=60),
    retry=_retry_on_429(),
)


# ── URL normalizer ────────────────────────────────────────────────────────

def _normalize_url_field(value) -> str | None:
    """Accept str, list[str] or None; return the first string or None."""
    if value is None:
        return None
    if isinstance(value, list):
        # Take the first string from the list
        return str(value[0]) if value else None
    return str(value)


# Negation/turn markers that separate a claim's positive assertion from the
# "what this paper lacks" part. Used by _fulltext_locate to keep only the
# verifiable assertion as the search signal.
_NEG_TRUNC_RE = re.compile(
    r"(?i)\b(without|but|however|lacks|lacking|missing|absent|despite|"
    r"beyond|never|does not|do not|did not|fails to|fail to|no "
    r"investigation|not address|doesn'?t|don'?t|while not)\b"
)


# ── Client ─────────────────────────────────────────────────────────────────

class SciverseClient:
    """Async client for the Sciverse API (api.sciverse.space)."""

    DEFAULT_BASE = "https://api.sciverse.space"

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.api_base = (
            api_base or os.getenv("SCIVERSE_API_BASE", self.DEFAULT_BASE)
        ).rstrip("/")
        self.api_key = api_key or os.getenv("SCIVERSE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Sciverse API key is empty. "
                "Set SCIVERSE_API_KEY in ~/.matresearcher/secrets.env "
                "(or as a system environment variable). "
                "Get a key at https://sciverse.space/tokens"
            )
        self._timeout = timeout
        self._content_lock = asyncio.Lock()  # global: serialize /content calls
        self._client = httpx.AsyncClient(
            base_url=self.api_base,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    # ── Public API ──────────────────────────────────────────────────────

    @_SEMANTIC_RETRY
    async def semantic_search(
        self,
        query: str,
        top_k: int = 50,
        filters: dict | None = None,
    ) -> list[dict]:
        """Semantic evidence retrieval via agentic-search.

        Returns list of paper dicts with: doi, title, authors, abstract,
        year, journal, chunk (evidence snippet), score, doc_id, page_no.
        """
        payload: dict = {"query": query, "top_k": min(top_k, 100)}
        if filters:
            payload["filters"] = self._normalize_filters(filters)
        resp = await self._client.post("/agentic-search", json=payload)
        resp.raise_for_status()
        return [self._normalize_agentic_hit(h) for h in resp.json().get("hits", [])]

    @_META_RETRY
    async def keyword_search(
        self,
        keywords: list[str],
        top_k: int = 50,
        filters: dict | None = None,
    ) -> list[dict]:
        """Structured metadata search via meta-search.

        Combines keywords into a query string and applies optional filters.
        Returns list of paper dicts with: doi, title, authors, abstract,
        year, journal, citation_count, doc_id, unique_id.
        """
        page_size = min(top_k, 200)
        query = " ".join(keywords) if keywords else None
        payload: dict = {"page_size": page_size, "page": 1}
        if query:
            payload["query"] = query
        meta_filters = self._build_meta_filters(filters)
        if meta_filters:
            payload["filters"] = meta_filters
        resp = await self._client.post("/meta-search", json=payload)
        resp.raise_for_status()
        return [self._normalize_meta_hit(r) for r in resp.json().get("results", [])]

    async def hybrid_search(
        self,
        query: str,
        keywords: list[str],
        top_k: int = 50,
        filters: dict | None = None,
    ) -> list[dict]:
        """Hybrid search combining semantic + keyword approaches, then dedup."""
        sem_results = await self.semantic_search(query, top_k, filters)
        kw_results = await self.keyword_search(keywords, top_k, filters)
        # Merge and dedup by DOI → title
        seen: set[str] = set()
        merged: list[dict] = []
        for r in sem_results + kw_results:
            key = r.get("doi") or r.get("title", "")
            if key and key not in seen:
                seen.add(key)
                merged.append(r)
        return merged[:top_k]

    async def locate_passage(self, doi: str, claim: str, doc_id: str | None = None) -> dict:
        """Locate a passage that supports a claim in a specific paper.

        Two entry paths (A-1 dual-anchor design):

        Path A — doc_id-first (recommended when the caller has a doc_id):
        The doc_id comes from Sciverse search results (already confirmed in
        library). Skip get_metadata entirely: semantic-search the claim and
        match hits by doc_id, then fall back to scanning the paper's full text
        by doc_id. This works even when the DOI is missing (the 94% case).

        Path B — DOI-based (original behavior):
        1. Normalize the target DOI (lowercase, strip URL/doi: prefixes).
        2. get_metadata(doi) first — confirm the paper is actually in the
           library and grab its doc_id. Returns reason="not_in_library" early
           when absent, so callers see WHY verification failed.
        3. agentic-search with the claim as query, match hits by normalized DOI
           (tolerates case/prefix drift in either side).
        4. If semantic match fails but the paper has a doc_id, fall back to
           scanning a limited window of full text for the claim's key terms.

        Returns dict with keys: doi, page, section, passage, match_score,
        reason ("semantic_match" | "semantic_match_docid" | "semantic_match_title"
        | "fulltext_located" | "not_in_library" | "no_match_in_results" |
        "invalid_doi" | "error"), doc_id.
        """
        # ── Path A: doc_id-first (A-0 verified: doc_id is a stable global key) ──
        if doc_id:
            result = await self._locate_by_doc_id(doi, claim, doc_id)
            if result:
                return result

        target = self._normalize_doi(doi)
        if not target:
            return self._empty_passage_result(doi, reason="invalid_doi")

        # Step 1: confirm in library + grab doc_id
        meta = {}
        try:
            meta = await self.get_metadata(target)
        except Exception:
            meta = {}
        if not meta:
            return self._empty_passage_result(doi, reason="not_in_library")
        meta_doc_id = meta.get("doc_id")

        # Step 2: semantic search + multi-signal match.
        # NOTE: agentic-search hits do NOT carry a `doi` field, so matching
        # by DOI alone always fails. Match on normalized DOI (when present),
        # then doc_id, then title similarity.
        try:
            hits = await self.semantic_search(claim, top_k=20)
        except Exception:
            hits = []
        meta_title = meta.get("title", "") or ""
        for h in hits:
            hit_doi = self._normalize_doi(h.get("doi") or "")
            if hit_doi and hit_doi == target:
                return self._passage_result(doi, h, meta_doc_id, "semantic_match")
            if meta_doc_id and h.get("doc_id") and h.get("doc_id") == meta_doc_id:
                return self._passage_result(doi, h, meta_doc_id, "semantic_match_docid")
            if meta_title and self._titles_similar(meta_title, h.get("title", "")):
                return self._passage_result(doi, h, meta_doc_id, "semantic_match_title")

        # Step 3: full-text fallback (only when the paper is in the library)
        if meta_doc_id:
            passage, score = await self._fulltext_locate(meta_doc_id, claim)
            if passage:
                return {
                    "doi": doi,
                    "page": None,
                    "section": None,
                    "passage": passage,
                    "match_score": score,
                    "reason": "fulltext_located",
                    "doc_id": meta_doc_id,
                }

        return self._empty_passage_result(doi, reason="no_match_in_results", doc_id=meta_doc_id)

    async def _locate_by_doc_id(self, doi: str, claim: str, doc_id: str) -> dict | None:
        """doc_id 优先定位路径（A-1）。

        doc_id 来自 Sciverse 检索结果的 metadata，已隐含「文献在库」。
        先尝试 semantic 匹配（快、能拿 chunk 作 passage），失败再按 doc_id
        直接扫描全文（最可靠，不依赖检索召回）。返回 result dict 或 None。
        """
        try:
            hits = await self.semantic_search(claim, top_k=20)
        except Exception:
            hits = []
        for h in hits:
            if h.get("doc_id") and h.get("doc_id") == doc_id:
                return self._passage_result(doi, h, doc_id, "semantic_match_docid")

        passage, score = await self._fulltext_locate(doc_id, claim)
        if passage:
            return {
                "doi": doi,
                "page": None,
                "section": None,
                "passage": passage,
                "match_score": score,
                "reason": "fulltext_located",
                "doc_id": doc_id,
            }
        return None

    @staticmethod
    def _passage_result(doi: str, hit: dict, doc_id, reason: str) -> dict:
        """Build a successful locate_passage result from a semantic hit."""
        return {
            "doi": doi,
            "page": hit.get("page_no"),
            "section": None,  # agentic-search doesn't expose section
            "passage": hit.get("chunk", ""),
            "match_score": hit.get("score", 0.0),
            "reason": reason,
            "doc_id": doc_id,
        }

    @staticmethod
    def _strip_html(text) -> str:
        return re.sub(r"<[^>]+>", "", text or "")

    @staticmethod
    def _title_key(title: str) -> set[str]:
        """Tokenize a title for fuzzy comparison (lowercase, drop stopwords)."""
        t = SciverseClient._strip_html(title).lower()
        tokens = set(re.findall(r"[a-z0-9]+", t))
        tokens -= {"the", "a", "an", "of", "and", "with", "for", "in", "on", "to"}
        return tokens

    @classmethod
    def _titles_similar(cls, t1: str, t2: str, threshold: float = 0.6) -> bool:
        """Jaccard-style overlap between two titles (robust to HTML/markup)."""
        a, b = cls._title_key(t1), cls._title_key(t2)
        if not a or not b:
            return False
        inter = len(a & b)
        return inter / max(len(a), len(b)) >= threshold

    @staticmethod
    def _normalize_doi(value) -> str:
        """Normalize a DOI for comparison: lowercase, strip, drop URL/doi: prefixes."""
        if not value:
            return ""
        v = str(value).strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
            if v.startswith(prefix):
                v = v[len(prefix):]
                break
        return v.strip()

    async def _fulltext_locate(self, doc_id: str, claim: str, max_chars: int = 16000) -> tuple[str, float]:
        """Scan a limited window of full text for the claim's key terms.

        The claim is truncated at the first negation/turn marker (without, but,
        however, lacks, ...) so that only the positive assertion is used as the
        search signal — "what the paper reports", not "what it lacks".

        Matching is full-text coverage (fraction of significant terms present
        anywhere in the document) rather than per-sentence overlap, because
        domain terms (doping, sintering, conductivity) spread across sentences.
        A sentence is still returned as the display passage.
        """
        try:
            text = await self.read_full_text(doc_id, max_chars=max_chars)
        except Exception:
            return "", 0.0
        if not text:
            return "", 0.0

        # Truncate at the first negation/turn marker → positive assertion only
        m = _NEG_TRUNC_RE.search(claim)
        pos = claim[: m.start()] if m else claim
        sentences = re.split(r"(?<=[.!?。；])", pos)
        key = max(
            (s.strip() for s in sentences if len(s.strip()) > 15),
            key=len,
            default=pos,
        )

        # Significant tokens: words >3 chars + numeric/scientific notation terms
        terms = [
            t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", key)
            if len(t) > 3
        ]
        terms += re.findall(r"\d+(?:\.\d+)?(?:\s*[×xX]\s*10[⁻⁽]?\d+[⁾]?)?", key)
        if not terms:
            return "", 0.0

        text_lower = text.lower()
        # Full-text term coverage (>=50% of terms present, min 3 hits)
        hits = sum(1 for t in terms if t in text_lower)
        ratio = hits / len(terms)
        if ratio < 0.5 or hits < 3:
            return "", 0.0

        # Best sentence as the display passage
        best_sent, best_hits = "", 0
        for sent in re.split(r"(?<=[.!?。；])", text_lower):
            s = sent.strip()
            if len(s) < 20:
                continue
            sh = sum(1 for t in terms if t in s)
            if sh > best_hits:
                best_hits, best_sent = sh, s
        display = best_sent[:500] if best_sent else text[:500]

        # Heuristic score: base 0.5 (passes threshold) + coverage bonus
        score = min(0.95, 0.5 + ratio * 0.4)
        return display, score

    async def get_metadata(self, doi: str) -> dict:
        """Fetch metadata for a specific DOI via meta-search."""
        payload: dict = {
            "filters": [{"field": "doi", "operator": "FILTER_OP_EQ", "value": doi}],
            "page_size": 1,
        }
        resp = await self._client.post("/meta-search", json=payload)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return self._normalize_meta_hit(results[0]) if results else {}

    @_CONTENT_RETRY
    async def get_content(
        self,
        doc_id: str,
        offset: int = 0,
        limit: int = 4096,
    ) -> dict:
        """Read full-text content by doc_id (from agentic-search / meta-search).

        Returns: {"text": str, "chars_returned": int, "next_offset": int, "more": bool}

        All /content calls are globally serialized via _content_lock to avoid
        Sciverse 429 rate-limiting. Lock is released on exception so tenacity
        retry does not block other callers during backoff.
        """
        async with self._content_lock:
            await asyncio.sleep(1.0)  # global throttle between ALL /content requests
            resp = await self._client.get(
                "/content",
                params={"doc_id": doc_id, "offset": offset, "limit": limit},
            )
            resp.raise_for_status()
            return resp.json()

    async def read_full_text(self, doc_id: str, max_chars: int = 50000) -> str:
        """Stream-read full text in chunks, concatenating results.

        Throttling is handled globally by get_content() via _content_lock.
        """
        parts: list[str] = []
        offset = 0
        while True:
            data = await self.get_content(doc_id, offset=offset, limit=4096)
            parts.append(data.get("text", ""))
            if not data.get("more"):
                break
            offset = data.get("next_offset", offset + 4096)
            if len("".join(parts)) >= max_chars:
                break
        return "".join(parts)

    async def close(self):
        await self._client.aclose()

    # ── Normalizers (public API fields → unified internal dict) ─────────

    @staticmethod
    def _normalize_agentic_hit(h: dict) -> dict:
        """agentic-search hit → unified paper dict.

        API returns `author` as array<string> (e.g. ["Jane Doe", "John Smith"]).
        """
        authors_raw = h.get("author", [])
        if isinstance(authors_raw, list):
            authors = [str(a) for a in authors_raw]
        else:
            authors = []
        return {
            "doi": h.get("doi"),
            "title": h.get("title", ""),
            "authors": authors,
            "abstract": h.get("abstract", ""),
            "year": h.get("publication_published_year"),
            "journal": h.get("publication_venue_name_unified"),
            "keywords": [],
            "pdf_url": _normalize_url_field(h.get("access_oa_url") or h.get("pdf_url")),
            "doc_id": h.get("doc_id"),
            "chunk": h.get("chunk"),
            # Extended fields
            "score": h.get("score"),
            "page_no": h.get("page_no"),
            "source_type": h.get("source_type"),
            "citation_count": h.get("citation_count"),
            "lang": h.get("lang"),
            "is_content_accessible": h.get("is_content_accessible", True),
        }

    @staticmethod
    def _normalize_meta_hit(r: dict) -> dict:
        """meta-search result → unified paper dict."""
        authors_raw = r.get("author", [])
        if isinstance(authors_raw, list):
            authors = [
                a.get("name", a) if isinstance(a, dict) else a
                for a in authors_raw
            ]
        else:
            authors = []
        return {
            "doi": r.get("doi"),
            "title": r.get("title", ""),
            "authors": authors,
            "abstract": r.get("abstract", ""),
            "year": r.get("publication_published_year"),
            "journal": r.get("publication_venue_name_unified"),
            "keywords": r.get("keywords", []),
            "pdf_url": _normalize_url_field(r.get("access_oa_url") or r.get("pdf_url")),
            "doc_id": r.get("doc_id"),
            "chunk": None,
            # Extended fields
            "unique_id": r.get("unique_id"),
            "citation_count": r.get("citation_count"),
            "lang": r.get("language"),
            "access_is_oa": r.get("access_is_oa"),
            "is_content_accessible": r.get("is_content_accessible", True),
        }

    @staticmethod
    def _empty_passage_result(
        doi: str, reason: str = "no_match", doc_id: str | None = None
    ) -> dict:
        return {
            "doi": doi, "page": None, "section": None, "passage": "",
            "match_score": 0.0, "reason": reason, "doc_id": doc_id,
        }

    # ── Filter helpers ──────────────────────────────────────────────────

    # Fields accepted by agentic-search /filters (per API docs).
    _AGENTIC_FILTER_KEYS = frozenset({
        "lang", "title", "author", "publication_venue_name_unified",
        "publication_venue_type", "publication_published_date",
        "publication_published_year", "citation_count",
        "influential_citation_count", "topics",
    })

    @staticmethod
    def _normalize_filters(raw: dict) -> dict:
        """Normalize caller-friendly filter dict to agentic-search format.

        Maps common keys like {"year_from": 2020} → {"publication_published_year": {"gte": 2020}}.
        Drops keys not in the agentic-search supported filter set.
        """
        filtered = {k: v for k, v in raw.items() if v is not None}
        # Step 1 — map alias keys before stripping
        if "year_from" in filtered and "publication_published_year" not in filtered:
            filtered["publication_published_year"] = {"gte": filtered.pop("year_from")}
        if "year_to" in filtered:
            existing = filtered.get("publication_published_year", {})
            if isinstance(existing, dict):
                existing["lte"] = filtered.pop("year_to")
            filtered["publication_published_year"] = existing
        # Step 2 — drop unknown keys (after alias mapping to keep the mapped result)
        filtered = {
            k: v for k, v in filtered.items()
            if k in SciverseClient._AGENTIC_FILTER_KEYS
        }
        return filtered

    @staticmethod
    def _build_meta_filters(raw: dict | None) -> list[dict] | None:
        """Convert simple filter dict to meta-search FilterItem array."""
        if not raw:
            return None
        result: list[dict] = []
        field_map = {
            "year_from": ("publication_published_year", "FILTER_OP_GTE"),
            "year_to": ("publication_published_year", "FILTER_OP_LTE"),
            "lang": ("language", "FILTER_OP_EQ"),
            "language": ("language", "FILTER_OP_EQ"),
            "journal": ("publication_venue_name_unified", "FILTER_OP_EQ"),
        }
        for key, value in raw.items():
            if value is None:
                continue
            if key in field_map:
                fld, op = field_map[key]
                result.append({"field": fld, "operator": op, "value": value})
        return result if result else None
