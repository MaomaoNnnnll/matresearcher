"""Literature data models."""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field, HttpUrl

if TYPE_CHECKING:
    from .knowledge import KnowledgeRecord


class LiteratureMetadata(BaseModel):
    """Bibliographic metadata for a literature entry."""
    doi: Optional[str] = None
    title: str
    authors: list[str] = Field(default_factory=list)
    journal: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    pdf_url: Optional[str] = None
    doc_id: Optional[str] = None    # Sciverse doc_id for full-text retrieval via /content
    chunk: Optional[str] = None     # Text snippet from agentic-search (fallback)
    is_content_accessible: bool = True  # Sciverse flag: whether /content is available for this paper
    citation_count: Optional[int] = None  # Citation count from Sciverse (agentic + meta)
    source: str = "sciverse"  # sciverse | local | manual
    query_source: str = "original"  # original | reformulated | expanded | fallback — which query brought this paper


class ParsedDocument(BaseModel):
    """Structured output from MinerU / PDF parsing."""
    literature_id: str
    sections: list[dict] = Field(default_factory=list)  # [{"heading": "...", "content": "..."}]
    tables: list[dict] = Field(default_factory=list)   # [{"caption": "...", "rows": [...]}]
    figure_descriptions: list[str] = Field(default_factory=list)
    full_text: str = ""
    page_count: int = 0
    parse_status: str = "success"  # success | partial | failed
    error_message: Optional[str] = None


class Literature(BaseModel):
    """A complete literature entry with metadata and parsing state."""
    id: str
    metadata: LiteratureMetadata
    relevance_score: Optional[float] = None
    parsed_document: Optional[ParsedDocument] = None
    knowledge_records: list["KnowledgeRecord"] = Field(default_factory=list)
    verification_status: str = "pending"  # pending | verified | anomaly

    @property
    def is_parsed(self) -> bool:
        return self.parsed_document is not None and self.parsed_document.parse_status in ("success", "partial")

    @property
    def citation(self) -> str:
        """Formatted citation string."""
        authors = ", ".join(self.metadata.authors[:3])
        if len(self.metadata.authors) > 3:
            authors += " et al."
        year = self.metadata.year or "n.d."
        title = self.metadata.title[:80]
        journal = self.metadata.journal or ""
        return f"{authors} ({year}). {title}. {journal}. DOI: {self.metadata.doi or 'N/A'}"


# Resolve forward reference to KnowledgeRecord (Pydantic v2 requires explicit rebuild).
from .knowledge import KnowledgeRecord   # noqa: E402 (import after class defs)

Literature.model_rebuild()
