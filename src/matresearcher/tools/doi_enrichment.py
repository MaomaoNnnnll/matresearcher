"""Title → DOI enrichment for literature records.

Sciverse's ``/agentic-search`` endpoint (the primary retrieval path used by the
literature-search agent) returns a ``doi`` field that is ``null`` for the large
majority of hits, so the evidence-verification stage frequently sees papers with
``doc_id`` but no resolvable DOI. This module recovers DOIs from the paper *title*
using a best-effort, three-layer lookup:

1. **Crossref REST API** (preferred, free, no key required):
   ``https://api.crossref.org/works?query.bibliographic=<title>&rows=5``.
2. **Sciverse ``/meta-search``** queried by title (uses the existing API key and
   returns DOIs that the agentic-search path omits).
3. **Fallback link** when no DOI can be found: render a resolvable Sciverse paper
   page from ``doc_id`` so citations still have a citable URL.

Design guarantees:
- Only records whose DOI is currently empty are enriched (never overwritten).
- Every candidate DOI is validated by a title-similarity check
  (``SIMILARITY_THRESHOLD`` default 0.72) to avoid mis-attributing a DOI to the
  wrong paper.
- Lookups are throttled (``min_interval``) and cached by normalized title so a
  survey run is cheap, polite, and reproducible across resume attempts.
- All failures are best-effort: a network/timeout error simply leaves the record
  unchanged and the pipeline continues.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

import httpx

from ..models.literature import Literature
from ..tools.sciverse import SciverseClient

_CROSSREF_URL = "https://api.crossref.org/works"
_SIMILARITY_THRESHOLD = 0.72
_STOPWORDS = {
    "the", "a", "an", "of", "and", "with", "for", "in", "on", "to", "by", "at",
    "from", "via", "using", "study", "effect", "effects", "analysis", "based",
    "review", "survey", "perspective", "approach", "method", "methods",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+")


def _norm_title_key(title: str) -> str:
    """Lowercase, HTML-stripped, stopword-free token key for comparison."""
    t = re.sub(r"<[^>]+>", "", title or "").lower()
    toks = set(_TOKEN_RE.findall(t))
    toks -= _STOPWORDS
    return " ".join(sorted(toks))


def _titles_match(a: str, b: str, threshold: float = _SIMILARITY_THRESHOLD) -> bool:
    """Token-overlap similarity between two titles (robust to HTML/markup)."""
    ka, kb = _norm_title_key(a), _norm_title_key(b)
    if not ka or not kb:
        return False
    sa, sb = set(ka.split()), set(kb.split())
    inter = len(sa & sb)
    return inter / max(len(sa), len(sb)) >= threshold


def _similarity_score(a: str, b: str) -> float:
    ka, kb = _norm_title_key(a).split(), _norm_title_key(b).split()
    if not ka or not kb:
        return 0.0
    inter = len(set(ka) & set(kb))
    return inter / max(len(ka), len(kb))


def _normalize_doi(value: str) -> Optional[str]:
    """Strip resolver prefixes / whitespace and validate the DOI shape.

    DOIs are case-insensitive, so the result is lowercased for consistent
    storage/dedup (matches SciverseClient._normalize_doi elsewhere).
    """
    if not value:
        return None
    s = str(value).strip().rstrip(".").lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?i)^doi[:\s]+", "", s).strip()
    if not _DOI_RE.match(s):
        return None
    return s


class DOIEnricher:
    """Recover missing DOIs for a list of Literature records, in place."""

    def __init__(
        self,
        sciverse: Optional[SciverseClient] = None,
        *,
        timeout: float = 10.0,
        use_crossref: bool = True,
        use_sciverse_meta: bool = True,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
        min_interval: float = 0.15,
        max_concurrency: int = 6,
        crossref_email: Optional[str] = None,
    ):
        self.sciverse = sciverse
        self._timeout = timeout
        self._use_crossref = use_crossref
        self._use_sciverse_meta = use_sciverse_meta
        self._threshold = similarity_threshold
        self._min_interval = min_interval
        self._sem = asyncio.Semaphore(max_concurrency)
        self._throttle_lock = asyncio.Lock()
        self._last_call = 0.0
        self._cache: dict[str, Optional[str]] = {}  # norm_title -> doi or None
        headers = {"Accept": "application/json"}
        if crossref_email:
            headers["User-Agent"] = f"matresearcher/1.0 (mailto:{crossref_email})"
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    # ── public API ────────────────────────────────────────────────────────
    async def enrich(self, literature: list[Literature]) -> list[Literature]:
        """Enrich every record lacking a DOI. Returns the same (mutated) list."""
        targets = [
            lit for lit in literature
            if not lit.metadata.doi and (lit.metadata.title or "").strip()
        ]
        if targets:
            await asyncio.gather(*(self._enrich_one(lit) for lit in targets))
        return literature

    async def close(self):
        await self._client.aclose()

    # ── internals ─────────────────────────────────────────────────────────
    async def _enrich_one(self, lit: Literature) -> None:
        try:
            title = lit.metadata.title.strip()
            norm = _norm_title_key(title)
            # Cache hit (None = looked up before and not found)
            if norm in self._cache:
                cached = self._cache[norm]
                if cached:
                    lit.metadata.doi = cached
                elif lit.metadata.doc_id and not lit.metadata.url:
                    lit.metadata.url = (
                        f"https://sciverse.space/paper/{lit.metadata.doc_id}"
                    )
                return

            doi = await self._lookup(title)
            self._cache[norm] = doi
            if doi:
                lit.metadata.doi = doi
            elif lit.metadata.doc_id and not lit.metadata.url:
                lit.metadata.url = (
                    f"https://sciverse.space/paper/{lit.metadata.doc_id}"
                )
        except Exception:
            # Best-effort: never let one bad record abort the batch.
            pass

    async def _lookup(self, title: str) -> Optional[str]:
        if self._use_crossref:
            doi = await self._crossref(title)
            if doi:
                return doi
        if self._use_sciverse_meta and self.sciverse is not None:
            doi = await self._sciverse_meta(title)
            if doi:
                return doi
        return None

    async def _throttle(self):
        async with self._throttle_lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    async def _crossref(self, title: str) -> Optional[str]:
        await self._throttle()
        try:
            resp = await self._client.get(
                _CROSSREF_URL,
                params={"query.bibliographic": title, "rows": 20},
            )
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
        except Exception:
            return None
        best: Optional[str] = None
        best_score = 0.0
        for item in items:
            cand_title = item.get("title")
            cand_title = cand_title[0] if isinstance(cand_title, list) and cand_title else ""
            if not cand_title or not _titles_match(title, cand_title, self._threshold):
                continue
            doi = _normalize_doi(str(item.get("DOI", "")))
            if not doi:
                continue
            # Prefer the highest-similarity match among valid candidates
            score = _similarity_score(title, cand_title)
            if best is None or score > best_score:
                best, best_score = doi, score
        return best

    async def _sciverse_meta(self, title: str) -> Optional[str]:
        try:
            results = await self.sciverse.keyword_search([title], top_k=5)
        except Exception:
            return None
        for r in results:
            doi = _normalize_doi(str(r.get("doi") or ""))
            if not doi:
                continue
            if _titles_match(title, r.get("title", ""), self._threshold):
                return doi
        return None
