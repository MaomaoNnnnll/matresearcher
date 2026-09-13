"""Unit tests for title→DOI enrichment (DOIEnricher).

The enrichment talks to external services (Crossref, Sciverse). These tests mock
the network layer so they run fast and offline, while still exercising the real
parsing/validation logic (title similarity guard, DOI normalization, cache,
fallback-link rendering).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from matresearcher.models.literature import Literature, LiteratureMetadata  # noqa: E402
from matresearcher.tools.doi_enrichment import (  # noqa: E402
    DOIEnricher,
    _normalize_doi,
    _titles_match,
)


# ── Fakes ───────────────────────────────────────────────────────────────────
class _FakeResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeCrossref:
    """Mimics httpx.AsyncClient for GET /works?query.bibliographic=..."""
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    async def get(self, url, params=None):
        self.calls += 1
        return _FakeResp(self.payload)

    async def aclose(self):
        return None


class FakeSciverse:
    """Mimics the subset of SciverseClient used by _sciverse_meta."""
    def __init__(self, results: list[dict]):
        self.results = results

    async def keyword_search(self, keywords, top_k=50, filters=None):
        return self.results


def _crossref_payload(*titles_dois: tuple[str, str]) -> dict:
    items = [{"title": [t], "DOI": d} for t, d in titles_dois]
    return {"message": {"items": items}}


def _lit(title: str, *, doi: str | None = None, doc_id: str | None = None) -> Literature:
    return Literature(
        id="lit_test",
        metadata=LiteratureMetadata(title=title, doi=doi, doc_id=doc_id),
    )


def _swap_client(enricher: DOIEnricher, fake: FakeCrossref):
    """Replace the real httpx client with a fake and return the original."""
    real = enricher._client
    enricher._client = fake
    return real


async def _teardown(enricher: DOIEnricher, real):
    await enricher.close()
    await real.aclose()


# ── Helpers / pure functions ─────────────────────────────────────────────────
def test_normalize_doi_variants():
    assert _normalize_doi("https://doi.org/10.1000/xyz123") == "10.1000/xyz123"
    assert _normalize_doi("doi:10.1000/xyz123") == "10.1000/xyz123"
    assert _normalize_doi("10.1000/xyz123") == "10.1000/xyz123"
    assert _normalize_doi(" 10.1109/TSE.2020.1234567 ") == "10.1109/tse.2020.1234567"
    assert _normalize_doi("not-a-doi") is None
    assert _normalize_doi("") is None


def test_titles_match_threshold():
    assert _titles_match("A study of solid electrolyte interfaces",
                         "A study of solid electrolyte interfaces")
    assert not _titles_match("A study of solid electrolyte interfaces",
                             "Quantum computing with trapped ions")
    # Minor wording insertion ("Al-doped", "garnet") still matches at >= 0.85
    assert _titles_match("Enhanced ionic conductivity in doped LLZO garnet",
                         "Enhanced ionic conductivity in Al-doped LLZO garnet")
    # But a short title padded with extra words can fall below 0.85 (guarded)
    assert not _titles_match("Enhanced ionic conductivity in doped LLZO",
                             "Enhanced ionic conductivity in Al-doped LLZO garnet")


# ── Crossref layer (primary) ─────────────────────────────────────────────────
async def test_crossref_hit_sets_doi():
    fake = FakeCrossref(_crossref_payload(
        ("Enhanced ionic conductivity in doped LLZO garnet", "10.1000/llzo"),
    ))
    e = DOIEnricher(use_sciverse_meta=False)
    real = _swap_client(e, fake)
    try:
        lit = _lit("Enhanced ionic conductivity in doped LLZO garnet")
        await e.enrich([lit])
        assert lit.metadata.doi == "10.1000/llzo"
    finally:
        await _teardown(e, real)


async def test_crossref_guards_wrong_title():
    # Top Crossref hit title is unrelated → must NOT mis-attribute a DOI.
    fake = FakeCrossref(_crossref_payload(
        ("Quantum computing with trapped ions", "10.9999/wrong"),
    ))
    e = DOIEnricher(use_sciverse_meta=False)
    real = _swap_client(e, fake)
    try:
        lit = _lit("Enhanced ionic conductivity in doped LLZO garnet")
        await e.enrich([lit])
        assert lit.metadata.doi is None
    finally:
        await _teardown(e, real)


async def test_prefers_best_matching_crossref_item():
    fake = FakeCrossref(_crossref_payload(
        ("Unrelated paper title one", "10.1000/a"),
        ("Enhanced ionic conductivity in doped LLZO garnet electrolyte", "10.1000/llzo"),
        ("Enhanced ionic conductivity in doped LLZO garnet", "10.1000/llzo-exact"),
    ))
    e = DOIEnricher(use_sciverse_meta=False)
    real = _swap_client(e, fake)
    try:
        lit = _lit("Enhanced ionic conductivity in doped LLZO garnet")
        await e.enrich([lit])
        # The exact-title match has the highest similarity score
        assert lit.metadata.doi == "10.1000/llzo-exact"
    finally:
        await _teardown(e, real)


# ── Sciverse meta layer (secondary) ──────────────────────────────────────────
async def test_sciverse_meta_fallback():
    e = DOIEnricher(use_crossref=False, use_sciverse_meta=True)
    real = _swap_client(e, FakeCrossref({"message": {"items": []}}))
    e.sciverse = FakeSciverse([
        {"title": "Enhanced ionic conductivity in doped LLZO garnet", "doi": "10.2000/meta"},
        {"title": "Different paper", "doi": "10.2000/other"},
    ])
    try:
        lit = _lit("Enhanced ionic conductivity in doped LLZO garnet")
        await e.enrich([lit])
        assert lit.metadata.doi == "10.2000/meta"
    finally:
        await _teardown(e, real)


async def test_sciverse_meta_skipped_when_title_mismatch():
    e = DOIEnricher(use_crossref=False, use_sciverse_meta=True)
    real = _swap_client(e, FakeCrossref({"message": {"items": []}}))
    e.sciverse = FakeSciverse([
        {"title": "Totally unrelated title", "doi": "10.2000/meta"},
    ])
    try:
        lit = _lit("Enhanced ionic conductivity in doped LLZO garnet")
        await e.enrich([lit])
        assert lit.metadata.doi is None
    finally:
        await _teardown(e, real)


# ── Integration behavior ─────────────────────────────────────────────────────
async def test_enrich_does_not_overwrite_existing_doi():
    fake = FakeCrossref(_crossref_payload(
        ("Enhanced ionic conductivity in doped LLZO garnet", "10.1000/new"),
    ))
    e = DOIEnricher(use_sciverse_meta=False)
    real = _swap_client(e, fake)
    try:
        lit = _lit("Enhanced ionic conductivity in doped LLZO garnet", doi="10.old/original")
        await e.enrich([lit])
        assert lit.metadata.doi == "10.old/original"  # untouched
    finally:
        await _teardown(e, real)


async def test_fallback_url_from_doc_id_when_no_doi():
    fake = FakeCrossref({"message": {"items": []}})
    e = DOIEnricher(use_sciverse_meta=False)
    real = _swap_client(e, fake)
    try:
        lit = _lit("Some paper without a DOI in any source", doc_id="doc_abc123")
        await e.enrich([lit])
        assert lit.metadata.doi is None
        assert lit.metadata.url == "https://sciverse.space/paper/doc_abc123"
    finally:
        await _teardown(e, real)


async def test_cache_reuse_avoids_second_lookup():
    fake = FakeCrossref(_crossref_payload(
        ("Shared paper title", "10.1000/shared"),
    ))
    e = DOIEnricher(use_sciverse_meta=False)
    real = _swap_client(e, fake)
    try:
        lit1 = _lit("Shared paper title")
        await e._enrich_one(lit1)
        assert fake.calls == 1
        lit2 = _lit("Shared paper title")  # same normalized title → cache hit
        await e._enrich_one(lit2)
        assert fake.calls == 1, "second identical title must hit cache, not re-query"
        assert lit2.metadata.doi == lit1.metadata.doi == "10.1000/shared"
    finally:
        await _teardown(e, real)


async def test_citation_shows_url_when_doi_missing():
    lit = Literature(
        id="l1",
        metadata=LiteratureMetadata(
            title="Paper with only a landing URL",
            authors=["Doe, J."],
            year=2024,
            doc_id="doc_xyz",
            url="https://sciverse.space/paper/doc_xyz",
        ),
    )
    c = lit.citation
    assert "URL: https://sciverse.space/paper/doc_xyz" in c
    assert "N/A" not in c


if __name__ == "__main__":
    # Manual run: python tests/test_doi_enrichment.py
    async def _all():
        for name, fn in list(globals().items()):
            if name.startswith("test_") and asyncio.iscoroutinefunction(fn):
                await fn()
                print(f"ok: {name}")
            elif name.startswith("test_") and callable(fn):
                fn()
                print(f"ok: {name}")
    asyncio.run(_all())
    print("ALL TESTS PASSED")
