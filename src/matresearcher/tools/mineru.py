"""MinerU PDF parser wrapper.

MinerU is an open-source document parsing engine (AGPL-3.0) that converts
PDF to structured content: headings, paragraphs, tables, formulas, figures.

Modes:
1. API mode: calls a running MinerU service (MINERU_API_URL)
2. CLI mode: invokes `magic-pdf` CLI (requires MinerU installed locally)
3. Mock mode: returns raw text (fallback when MinerU is unavailable)
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional

import httpx
from rich.console import Console

from ..models.literature import ParsedDocument

console = Console()


class MinerUParser:
    """Wraps MinerU for PDF → structured document parsing."""

    def __init__(self, api_url: str | None = None, extract_tables: bool = True, extract_figures: bool = True):
        self.api_url = (api_url or os.getenv("MINERU_API_URL", "")).rstrip("/")
        self.extract_tables = extract_tables
        self.extract_figures = extract_figures

    async def parse(self, literature_id: str, pdf_path: str) -> ParsedDocument:
        """Parse a PDF file and return structured document."""
        if self.api_url:
            return await self._parse_via_api(literature_id, pdf_path)
        elif _mineru_cli_available():
            return await self._parse_via_cli(literature_id, pdf_path)
        else:
            # Not a mock: PyMuPDF extracts the real text layer. Renamed because
            # "mock parser" in the logs read as fabricated document content.
            console.print(
                "[yellow]Warning: MinerU not available, falling back to "
                "PyMuPDF text extraction (layout/tables/figures unavailable)[/yellow]"
            )
            return await self._parse_fallback_pymupdf(literature_id, pdf_path)

    async def _parse_via_api(self, literature_id: str, pdf_path: str) -> ParsedDocument:
        """Call MinerU REST API."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                with open(pdf_path, "rb") as f:
                    resp = await client.post(
                        f"{self.api_url}/parse",
                        files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
                        params={"extract_tables": self.extract_tables, "extract_figures": self.extract_figures},
                    )
                    resp.raise_for_status()
                    data = resp.json()
            return ParsedDocument(
                literature_id=literature_id,
                sections=data.get("sections", []),
                tables=data.get("tables", []),
                figure_descriptions=data.get("figures", []),
                full_text=data.get("full_text", ""),
                page_count=data.get("page_count", 0),
                parse_status="success",
            )
        except Exception as e:
            return ParsedDocument(
                literature_id=literature_id,
                parse_status="failed",
                error_message=str(e),
            )

    async def _parse_via_cli(self, literature_id: str, pdf_path: str) -> ParsedDocument:
        """Call MinerU CLI (magic-pdf)."""
        try:
            with tempfile.TemporaryDirectory() as outdir:
                result = subprocess.run(
                    ["magic-pdf", "-p", pdf_path, "-o", outdir, "-m", "auto"],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"MinerU CLI failed: {result.stderr}")

                # Read the structured output (Markdown + JSON)
                md_path = os.path.join(outdir, "auto", os.path.basename(pdf_path).replace(".pdf", ".md"))
                content_path = os.path.join(outdir, "auto", os.path.basename(pdf_path).replace(".pdf", "_content_list.json"))

                full_text = ""
                if os.path.exists(md_path):
                    with open(md_path, encoding="utf-8") as f:
                        full_text = f.read()

                sections = _split_markdown_to_sections(full_text)

                return ParsedDocument(
                    literature_id=literature_id,
                    sections=sections,
                    full_text=full_text,
                    parse_status="success" if full_text else "partial",
                )
        except Exception as e:
            return ParsedDocument(
                literature_id=literature_id,
                parse_status="failed",
                error_message=str(e),
            )

    async def _parse_fallback_pymupdf(self, literature_id: str, pdf_path: str) -> ParsedDocument:
        """Fallback: read raw text from PDF via PyMuPDF (real text, no layout)."""
        try:
            # Try PyMuPDF if available, otherwise return empty
            import fitz  # type: ignore
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()
            sections = _split_markdown_to_sections(full_text)
            return ParsedDocument(
                literature_id=literature_id,
                sections=sections,
                full_text=full_text,
                page_count=len(sections),
                parse_status="partial",
            )
        except ImportError:
            return ParsedDocument(
                literature_id=literature_id,
                full_text="[PDF parsing unavailable - install MinerU or PyMuPDF]",
                parse_status="failed",
                error_message="No PDF parser available",
            )


def _mineru_cli_available() -> bool:
    """Check if magic-pdf CLI is available."""
    try:
        result = subprocess.run(["magic-pdf", "--version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _split_markdown_to_sections(text: str) -> list[dict]:
    """Split markdown text into sections by headings."""
    sections = []
    current_heading = ""
    current_content = []
    for line in text.split("\n"):
        if line.startswith("#"):
            if current_heading or current_content:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_content),
                })
            current_heading = line.lstrip("# ").strip()
            current_content = []
        else:
            current_content.append(line)
    if current_heading or current_content:
        sections.append({
            "heading": current_heading,
            "content": "\n".join(current_content),
        })
    return sections
