"""Knowledge Extraction Agent (Step 7) + Data Quality Check (Step 7a).

Step 7: Extract structured knowledge from parsed PDFs using LLM.
Step 7a: Three-way quality gate (PASS / REVIEW / FAIL); flagged records
         (REVIEW + FAIL) are stored separately and written to
         ``quality_report.json`` in the run's log directory.

Quality checks (L1 hard rules → FAIL, L1 soft rules → REVIEW):
- Numeric values must have supporting raw_quotes (deep extraction)   [FAIL]
- Conductivity must fall within the absolute plausible range          [FAIL]
- Conductivity must fall within the material-family window
  (oxide / sulfide / halide / polymer / borohydride lookup)           [REVIEW/FAIL]
- Suspected unit confusion (S/cm ↔ mS/cm, mV ↔ V) detected by
  rescaling the value into the family window — reported with a
  correction hint, never silently corrected                            [REVIEW]
- Temperature values must be in plausible range (0 ~ 3000 K)          [FAIL]
- Electrochemical window must be > 0 V and < 10 V                     [FAIL]

Cross-record checks (L2, run after all papers are extracted):
- R9 Arrhenius: for each material composition with ≥4 (T, σ) pairs spanning >50 K,
  fit ln(σ) vs 1/T; flag implausible Ea (<0.05 or >1.5 eV) or
  individual points deviating >2 ln-units from the trend              [REVIEW]
- R10 Statistical outlier: for each family with ≥5 conductivity
  values, flag log10(σ) outliers beyond max(3.0×MAD, 1.5) from
  the family median                                                    [REVIEW]
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..state import WorkflowState
from .base import BaseAgent
from ..models.knowledge import KnowledgeRecord, NumericValue
from ..rules.material_config import (
    load_material_config,
    family_keywords,
    conductivity_window,
)


KNOWLEDGE_EXTRACTION_PROMPT = """You are a materials science expert. Extract structured knowledge from the following paper section.

Extract these fields as a JSON object (use null for missing values):

- material_composition: Chemical formula of the material being studied (e.g. "Li7La3Zr2O12", "Li6PS5Cl")
- crystal_structure: Crystal structure (e.g. "garnet", "perovskite", "cubic", "tetragonal")
- ionic_conductivity: {{"value": number, "unit": "S/cm or mS/cm etc.", "temperature_K": number, "raw_quote": "exact text"}}
- electrochemical_window: {{"value": number, "unit": "V", "raw_quote": "exact text"}}
- synthesis_method: Description of how the material was synthesized
- sintering_temperature: {{"value": number, "unit": "K or C", "raw_quote": "exact text"}}
- test_temperature: {{"value": number, "unit": "K or C", "raw_quote": "exact text"}}
- pressure: {{"value": number, "unit": "MPa or GPa etc.", "raw_quote": "exact text"}}
- simulation_method: "DFT", "MD", "Monte Carlo", or null
- key_findings: 1-2 sentence summary of the main results

IMPORTANT:
1. ALWAYS include raw_quote for every numeric value - copy the exact sentence from the text
2. raw_quote must be a SINGLE LINE, at most 120 characters, plain text only: no LaTeX code ($...$, \\mathrm, etc.), no markdown image links, no tables. If the value comes from a merged table row, quote only the relevant cell
3. Pay attention to units - if unit is "mS/cm", include it exactly as written
4. If multiple materials are studied, extract the primary/best-performing one
5. The paper text may contain LaTeX math formulas ($...$, $$...$$) - read them by their mathematical meaning and transcribe any values you extract as plain text. Do NOT copy LaTeX code into ANY field
6. Return ONLY valid JSON, no markdown fences, no extra text

Paper text:
{text}"""


LIGHT_EXTRACTION_PROMPT = """You are a materials science expert. Quickly extract only the most essential information from this paper.

Extract ONLY these 4 fields as a JSON object (use null for missing values):

- material_composition: Chemical formula of the primary material studied (e.g. "Li7La3Zr2O12", "Li6PS5Cl")
- ionic_conductivity: {{"value": number, "unit": "S/cm or mS/cm etc.", "temperature_K": number}}
- test_temperature: {{"value": number, "unit": "K or C"}}
- synthesis_method: Brief description of the primary synthesis approach (e.g. "solid-state reaction", "sol-gel", "ball milling")

IMPORTANT:
1. Be concise — only extract the single best-performing material if multiple are studied
2. Do NOT extract raw_quote or detailed fields like crystal_structure, electrochemical_window, sintering_temperature, pressure, simulation_method, or key_findings
3. Pay attention to units
4. Return ONLY valid JSON, no markdown fences, no extra text

Paper text:
{text}"""


class KnowledgeExtractionAgent(BaseAgent):
    name = "knowledge_extraction"
    role = "知识提取 Agent"

    # ── Semantic chunking constants ──
    # Abbreviations whose periods should NOT be treated as sentence boundaries
    _ABBREVIATIONS = [
        "e.g.", "i.e.", "et al.", "vs.", "cf.", "etc.", "viz.",
        "Fig.", "Eq.", "Ref.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.",
        "Vol.", "No.", "pp.", "Suppl.", "approx.", "ca.", "resp.",
        "esp.", "incl.", "excl.", "min.", "max.", "temp.",
    ]

    # Regex to detect academic paper section/subsection headers
    # (compiled once at class-level for performance)
    _SECTION_PAT = re.compile(
        r"^[#]{1,4}\s"                     # Markdown: ##, ###, ####
        r"|^\s*(?:\d+\.)+\s+\S"            # 1.  2.1  3.2.1
        r"|^\s*[IVX]+\.\s+\S"              # I.  IV.
        r"|^\s*\(\d+\)\s+\S"               # (1)  (2)
        r"|^\s*(?:Introduction|Methods?|Experimental"
        r"|Results?\s*(?:and\s+Discussion)?"
        r"|Discussion|Conclusions?|Abstract|Background"
        r"|Related\s+Work|Acknowledgments?"
        r"|References?|Supplementary|Appendix)[\s:]*$"
        r"|^\s*(?:引言|前言|实验方法?|实验部分|结果与讨论"
        r"|结果|讨论|结论|摘要|背景|参考文献?|致谢|附录|补充材料)[\s:]*$",
        re.IGNORECASE,
    )

    # ── Numeric parsing helpers (unit coefficients, superscripts, °C→K) ──
    _SUPERSCRIPT_MAP = {
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-",
    }
    # Matches "10⁻³", "10^-4", "10-3" inside a unit string
    _UNIT_COEF_RE = re.compile(
        r"10\s*\^?\s*([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+|[-+]?\d+)", re.IGNORECASE
    )
    # Matches a full scalar like "1×10⁻³", "1.5x10^-4"
    _SCI_VALUE_RE = re.compile(
        r"^\s*([+-]?\d+(?:\.\d+)?)\s*[×xX*]\s*10\s*\^?\s*"
        r"([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+|[-+]?\d+)\s*$"
    )

    # ── R5: material-family classification for threshold lookup ──
    # Loaded from config/materials.yaml via rules/material_config.py (with
    # built-in fallbacks), so the domain can be retargeted by editing the YAML
    # instead of this Python file. Keywords are matched case-insensitively
    # against material_composition (covers plain-language names like
    # "argyrodite" and aliases like "LGPS" / "LLZO").
    _MATERIAL_CFG = load_material_config()
    _FAMILY_KEYWORDS = family_keywords(_MATERIAL_CFG)

    # Element-symbol probes for the formula fallback (case-sensitive).
    # An element symbol in a chemical formula is an uppercase letter
    # (or pair) followed by another uppercase letter, a digit or the end —
    # e.g. "S" in "LiPSCl" matches, the "s" in "Li2S"... "S" at end matches.
    _ELEMENT_RES = {
        "S": re.compile(r"S(?=[A-Z]|$)"),
        "O": re.compile(r"O(?=[A-Z]|$)"),
        "Cl": re.compile(r"Cl(?=[A-Z]|$)"),
        "Br": re.compile(r"Br(?=[A-Z]|$)"),
        "I": re.compile(r"I(?=[A-Z]|$)"),
        "F": re.compile(r"F(?=[A-Z]|$)"),
    }

    # R5: per-family conductivity windows at room temperature (S/cm),
    # sourced from config/materials.yaml (fallback defaults if absent).
    # Stored as ((plausible_range_lo, plausible_range_hi), warn_above, label)
    # to keep the consumption site unchanged.
    _FAMILY_CONDUCTIVITY = {
        fam: (
            tuple(w.get("plausible_range", [1e-12, 1.0])),
            w.get("warn_above", 1.0),
            w.get("label", fam),
        )
        for fam, w in _MATERIAL_CFG.get("conductivity_windows", {}).items()
    }

    @classmethod
    def _detect_material_family(cls, composition: str | None) -> str:
        """Classify a material into a family for threshold lookup (R5).

        Order: (1) keyword/alias table over the raw string,
               (2) element-based fallback on letters only:
                   S → sulfide; Cl/Br/I/F → halide; O → oxide.
        Returns "unknown" when nothing matches (global rules then apply).
        """
        if not composition:
            return "unknown"
        low = composition.lower()
        for family, keywords in cls._FAMILY_KEYWORDS:
            if any(kw in low for kw in keywords):
                return family
        # Strip parenthetical notes (e.g. "(x = 0.15)", "(LLZO)") before
        # element analysis — otherwise trailing "x" or alias text breaks
        # the O/S/Cl lookahead patterns.
        clean = re.sub(r"\([^)]*\)", "", composition)
        letters = re.sub(r"[^A-Za-z]", "", clean)
        if cls._ELEMENT_RES["S"].search(letters):
            return "sulfide"
        if any(cls._ELEMENT_RES[el].search(letters) for el in ("Cl", "Br", "I", "F")):
            return "halide"
        if cls._ELEMENT_RES["O"].search(letters):
            return "oxide"
        return "unknown"

    @staticmethod
    def _unit_confusion_hint(v: float, window: tuple[float, float]) -> str | None:
        """R6: detect likely S/cm ↔ mS/cm unit mix-ups.

        If rescaling by 1e3 (milli prefix dropped, or wrongly added) lands the
        value back inside the family window, the LLM probably mishandled the
        prefix. Detection only — the correction is reported as a hint in
        quality_issues, never applied silently.
        """
        lo, hi = window
        for factor, cause in (
            (1e-3, "mS/cm value mislabeled as S/cm"),
            (1e3, "S/cm value mislabeled as mS/cm"),
        ):
            w = v * factor
            if lo <= w <= hi:
                return (
                    f"Conductivity {v:.2e} S/cm suspected unit confusion "
                    f"({cause}) — corrected value would be {w:.2e} S/cm; "
                    f"check raw_quote / original_unit"
                )
        return None

    @staticmethod
    def _format_range(rng: tuple[float, float]) -> str:
        return f"[{rng[0]:g}, {rng[1]:g}]"

    @staticmethod
    def _parse_scalar_number(v) -> float | None:
        """Parse a scalar numeric value that may be '1×10⁻³', '1.2e-4' or plain."""
        if isinstance(v, (int, float)):
            return float(v)
        if not isinstance(v, str):
            return None
        s = v.strip()
        m = KnowledgeExtractionAgent._SCI_VALUE_RE.match(s)
        if m:
            mant, exp = m.group(1), m.group(2)
            exp = "".join(
                KnowledgeExtractionAgent._SUPERSCRIPT_MAP.get(c, c) for c in exp
            )
            try:
                return float(mant) * 10.0 ** int(exp)
            except ValueError:
                return None
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _parse_unit_coefficient(unit) -> float | None:
        """Extract a ×10^n coefficient from a unit like '×10⁻³ S/cm'.

        Returns the multiplier (e.g. 1e-3) or None when the unit carries no
        pure power-of-ten coefficient.
        """
        if not unit or not isinstance(unit, str):
            return None
        m = KnowledgeExtractionAgent._UNIT_COEF_RE.search(unit)
        if not m:
            return None
        exp_str = "".join(
            KnowledgeExtractionAgent._SUPERSCRIPT_MAP.get(c, c) for c in m.group(1)
        )
        try:
            exp = int(exp_str)
        except ValueError:
            return None
        if exp == 0:
            return None
        return 10.0 ** exp

    @staticmethod
    def _normalize_temp_unit(nv: NumericValue | None) -> NumericValue | None:
        """Normalize a temperature NumericValue to Kelvin (idempotent).

        LLMs frequently return °C values with negative numbers (e.g. -15 °C
        emitted as -15), which the temperature_range check would otherwise
        reject as out-of-range. Converting C → K fixes the false positive.
        """
        if nv is None or nv.value is None or not nv.unit:
            return nv
        u = str(nv.unit).strip().lower()
        if u in ("c", "°c", "℃", "degc"):
            nv.value = nv.value + 273.15
            nv.unit = "K"
        return nv

    _CONDUCTIVITY_PREFIX_FACTORS = {
        "": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "μ": 1e-6,
        "n": 1e-9, "p": 1e-12,
    }
    _CONDUCTIVITY_UNIT_RE = re.compile(
        r"([munµμp]?)\s*s\s*(?:/|·|\.)?\s*(cm|m)\b", re.IGNORECASE
    )

    @staticmethod
    def _normalize_conductivity_unit(nv: NumericValue | None) -> NumericValue | None:
        """Convert a conductivity NumericValue to S/cm (idempotent).

        LLMs often report "1 mS/cm" with value 1.0 — the plausibility band is
        defined in S/cm, so without conversion such values are wrongly flagged
        as implausibly high. Handles mS/cm, µS/cm, S/m, S·cm⁻¹, and any ×10ⁿ
        coefficient embedded in the unit string.
        """
        if nv is None or nv.value is None or not nv.unit:
            return nv
        u = str(nv.unit).strip().lower()
        coef = KnowledgeExtractionAgent._parse_unit_coefficient(u) or 1.0
        m = KnowledgeExtractionAgent._CONDUCTIVITY_UNIT_RE.search(u)
        if not m:
            # No recognizable S-unit — leave untouched.
            return nv
        prefix, base = m.group(1).lower(), m.group(2)
        factor = KnowledgeExtractionAgent._CONDUCTIVITY_PREFIX_FACTORS.get(prefix, 1.0)
        if base == "m":
            factor *= 1e-2  # S/m → S/cm
        scale = coef * factor
        if scale != 1.0:
            nv.value = nv.value * scale
        nv.unit = "S/cm"
        return nv

    def __init__(self, llm=None, config=None, log_dir=None, run_cache=None):
        super().__init__(llm, config, log_dir=log_dir)
        self.quality_rules = self.config.get("data_quality_checks", {
            "conductivity_range": (1e-10, 1.0),    # S/cm — absolute bounds
            "temperature_range": (0.0, 3000.0),     # K
            "window_range": (0.0, 10.0),            # V
            "conductivity_warn_above": 0.5,         # S/cm — warn for implausible-high values
        })

        # Run-level cache (per-paper resume support). Optional; None disables.
        self._run_cache = run_cache

        # Debug output dir for failed chunks (content-filter / JSON diagnostics)
        self._debug_dir = None
        if self._log_path:
            self._debug_dir = Path(self._log_path).parent / "debug" / "failed_chunks"

    async def run(self, state: WorkflowState) -> dict:
        filtered = state.get("filtered_literature", [])
        if not filtered:
            self.log("No literature to extract from", "yellow")
            return {}

        # ── Layered extraction: deep (core) vs light (auxiliary) ──
        # Sort by relevance_score descending; unscored papers go last
        scored_lit = [lit for lit in filtered if lit.relevance_score is not None]
        unscored_lit = [lit for lit in filtered if lit.relevance_score is None]
        scored_lit.sort(key=lambda x: x.relevance_score, reverse=True)

        deep_top_n = self.config.get("deep_extraction_top_n", 8)
        core_lit = scored_lit[:deep_top_n]
        auxiliary_lit = scored_lit[deep_top_n:] + unscored_lit

        self.log(
            f"Layered extraction: {len(core_lit)} core (deep) + "
            f"{len(auxiliary_lit)} auxiliary (light) = {len(filtered)} total"
        )

        # ── Concurrency control ──
        max_concurrent = self.config.get("max_concurrent_extraction", 5)
        semaphore = asyncio.Semaphore(max_concurrent)
        self.log(f"Parallel extraction with max {max_concurrent} concurrent LLM calls")

        all_records = []
        anomaly_records = []

        # Phase 1: Parallel deep extraction for core literature (full 9 fields)
        self.log(f"--- Phase 1: Deep extraction ({len(core_lit)} papers, parallel) ---")
        deep_tasks = [
            self._extract_one_parallel(lit, depth="deep", semaphore=semaphore)
            for lit in core_lit
        ]
        deep_results = await asyncio.gather(*deep_tasks, return_exceptions=True)

        # Phase 2: Parallel light extraction for auxiliary literature (4 fields)
        self.log(f"--- Phase 2: Light extraction ({len(auxiliary_lit)} papers, parallel) ---")
        light_tasks = [
            self._extract_one_parallel(lit, depth="light", semaphore=semaphore)
            for lit in auxiliary_lit
        ]
        light_results = await asyncio.gather(*light_tasks, return_exceptions=True)

        # ── Aggregate results from both phases ──
        for result in deep_results + light_results:
            if isinstance(result, Exception):
                self.log(f"Extraction task failed: {result}", "red")
                continue
            lit, verified, anomalies = result
            all_records.extend(verified)
            anomaly_records.extend(anomalies)

        # ── L2: Cross-record checks (R9 Arrhenius + R10 statistical outlier) ──
        # These run AFTER all papers are extracted, because they need records
        # from multiple papers to detect cross-literature anomalies.  Records
        # demoted here are moved from PASS → REVIEW and removed from
        # lit.knowledge_records so they don't enter fusion.
        cross_issues: dict[str, list[str]] = {}
        cross_issues.update(self._arrhenius_check(all_records))
        cross_issues.update(self._statistical_outlier_check(all_records))

        if cross_issues:
            demoted_ids = set(cross_issues.keys())
            for r in all_records:
                if r.id in demoted_ids:
                    r.quality_status = "review"
                    r.quality_issues = cross_issues[r.id]
                    anomaly_records.append(r)
            all_records = [r for r in all_records if r.id not in demoted_ids]
            # Remove demoted records from lit.knowledge_records so fusion
            # never sees them.
            for lit in filtered:
                if lit.knowledge_records:
                    lit.knowledge_records = [
                        r for r in lit.knowledge_records if r.id not in demoted_ids
                    ]
            self.log(
                f"Cross-record checks (R9/R10): {len(demoted_ids)} record(s) "
                f"demoted PASS → REVIEW"
            )

        total = len(all_records) + len(anomaly_records)
        n_review = sum(1 for r in anomaly_records if getattr(r, "quality_status", "") == "review")
        n_fail = len(anomaly_records) - n_review
        self.log(f"Extraction complete: {len(all_records)} PASS + {n_review} REVIEW + "
                 f"{n_fail} FAIL = {total} total "
                 f"({len(core_lit)} deep + {len(auxiliary_lit)} light)")

        # Step 7a artifact: persist the three-way verdict so REVIEW/FAIL
        # records (with unit-confusion correction hints) stay inspectable
        # outside the pipeline.
        self._write_quality_report(all_records, anomaly_records)

        return {
            "filtered_literature": filtered,
            "anomaly_records": anomaly_records,
        }

    async def _extract_one_parallel(
        self,
        lit,
        depth: str,
        semaphore: asyncio.Semaphore,
    ) -> tuple:
        """Extract knowledge from a single paper with concurrency control.

        Returns (lit, verified_records, anomaly_records) — does NOT mutate
        shared state, so it is safe for concurrent execution.
        """
        if not lit.is_parsed:
            self.log(f"Skipping {lit.id}: not parsed", "yellow")
            return lit, [], []

        # ── Per-paper cache: resume mid-way after a crash ──
        cache = self._run_cache
        if cache is not None and cache.has_doc(lit.id, depth):
            cached = cache.get_doc(lit.id, depth)
            if cached is not None:
                verified = cached.get("verified", [])
                anomalies = cached.get("anomalies", [])
                # Re-judge cached anomalies under the CURRENT quality rules.
                # Rules may have been relaxed since the cache was written
                # (light raw_quote requirement, numeric-data bar, ×10ⁿ unit
                # coefficients, °C→K normalization) — re-running the rule layer
                # costs zero LLM tokens and promotes now-valid records back to
                # verified without forcing a full re-extraction.
                if anomalies:
                    for rec in anomalies:
                        if rec.ionic_conductivity:
                            self._normalize_conductivity_unit(rec.ionic_conductivity)
                        if rec.sintering_temperature:
                            self._normalize_temp_unit(rec.sintering_temperature)
                        if rec.test_temperature:
                            self._normalize_temp_unit(rec.test_temperature)
                    re_verified, re_anomaly = self._quality_check(
                        anomalies, depth=depth
                    )
                    if re_verified:
                        verified = list(verified) + re_verified
                        anomalies = re_anomaly
                        self.log(
                            f"  {lit.id}: re-judged {len(re_verified)} cached "
                            f"anomaly -> verified under new rules"
                        )
                        try:
                            cache.save_doc(lit.id, depth, verified, anomalies)
                        except Exception:
                            pass
                lit.knowledge_records = verified
                self.log(
                    f"  {lit.id}: {len(verified)} records (cache hit), "
                    f"{len(anomalies)} flagged "
                    f"({sum(1 for r in anomalies if r.quality_status == 'review')} review, "
                    f"{sum(1 for r in anomalies if r.quality_status == 'anomaly')} fail)"
                )
                return lit, verified, anomalies
            self.log(f"  {lit.id}: cache corrupt, re-extracting", "yellow")

        doc = lit.parsed_document
        depth_label = "DEEP" if depth == "deep" else "LIGHT"
        self.log(f"[{depth_label}] Extracting from {lit.id}: {lit.metadata.title[:60]}...")

        try:
            doi = lit.metadata.doi if lit.metadata else None
            records = await self._extract_from_document_parallel(
                lit.id, doc.full_text, depth=depth, semaphore=semaphore, doi=doi
            )
            # Step 7: Quality check (depth-aware rules)
            verified, anomalies = self._quality_check(records, depth=depth)
            lit.knowledge_records = verified
            # Persist per-paper result so a crash doesn't lose completed papers
            if cache is not None:
                try:
                    cache.save_doc(lit.id, depth, verified, anomalies)
                except Exception as e:
                    self.log(f"  {lit.id}: cache save failed: {e}", "yellow")
            self.log(
                f"  {lit.id}: {len(verified)} verified, {len(anomalies)} flagged "
                f"({sum(1 for r in anomalies if r.quality_status == 'review')} review, "
                f"{sum(1 for r in anomalies if r.quality_status == 'anomaly')} fail)"
            )
            return lit, verified, anomalies
        except Exception as e:
            self.log(f"Failed to extract from {lit.id}: {e}", "red")
            lit.verification_status = "anomaly"
            return lit, [], []

    # ── Legacy sequential extraction (kept for backward compatibility) ──
    async def _extract_one(
        self, lit, depth: str, all_records: list, anomaly_records: list
    ):
        """Sequential extraction — mutates shared lists. Deprecated in favor of
        _extract_one_parallel but kept as fallback."""
        lit_out, verified, anomalies = await self._extract_one_parallel(
            lit, depth, asyncio.Semaphore(1)
        )
        all_records.extend(verified)
        anomaly_records.extend(anomalies)

    async def _extract_from_document_parallel(
        self, lit_id: str, full_text: str, depth: str, semaphore: asyncio.Semaphore,
        doi: str | None = None,
    ) -> list[KnowledgeRecord]:
        """Extract knowledge with parallel chunk processing.

        Each chunk's LLM call is governed by the shared semaphore so that
        the total concurrent LLM calls across ALL papers never exceeds the limit.

        Args:
            lit_id: Literature ID for record naming.
            full_text: Parsed document full text.
            depth: "deep" (9 fields) or "light" (4 fields).
            semaphore: Shared semaphore for global concurrency control.
            doi: Real DOI of the paper for evidence traceability.
        """
        # Select prompt template by depth
        if depth == "light":
            prompt_template = LIGHT_EXTRACTION_PROMPT
            max_tokens = 2048
        else:
            prompt_template = KNOWLEDGE_EXTRACTION_PROMPT
            max_tokens = 16384  # deep extraction may need room for long synthesis_method values

        records = []

        if self.llm:
            # Strip HTML tags from Sciverse /content before chunking
            # (prevents content-filter rejections of <sub>/<sup>/<i> tags)
            clean_text = self._strip_html(full_text)
            # Upstream cleaning: drop image-placeholder lines, convert LaTeX
            # math to plain text, and separate column-merged table rows.
            # This runs BEFORE chunking so no chunk ever carries the two
            # top triggers of "non-JSON content" extraction failures.
            clean_text = self._clean_extraction_text(clean_text)
            chunk_size = self.config.get("chunk_max_chars", 4000)
            chunk_overlap = self.config.get("chunk_overlap_sentences", 0)
            chunks = self._chunk_text(
                clean_text, max_chars=chunk_size, overlap_sentences=chunk_overlap
            )
            self.log(
                f"  Semantic chunking: {len(clean_text)} chars → "
                f"{len(chunks)} chunks (max {chunk_size} chars/sentence-level)",
                "dim",
            )

            async def _extract_chunk(chunk: str, attempt: int = 1) -> dict | list:
                """Extract from one chunk, guarded by the shared semaphore.

                Error recovery (three branches, mutually exclusive):
                1. RateLimitError / RetryError → 30-60s backoff, retry once
                2. ValueError (empty response) → SAME-text retry after 15 s —
                   transient provider-side event, NEVER shorten
                3. ValueError (truncated / non-JSON) → shorter chunk, retry once
                4. All other errors → propagate

                NOTE (2026-08-14 replay experiment): empty responses are
                TRANSIENT, not deterministic content rejections — the same
                chunks failed 0/3 back-to-back but succeeded 6/6 spaced 8 s
                apart.  tenacity in llm.py already retries them same-text
                (3 attempts, exp backoff); the branches below handle whatever
                survives.

                A per-slot 2s delay spaces consecutive calls to avoid burst
                patterns that overwhelm low-tier API rate limits.
                """
                async with semaphore:
                    try:
                        # Latex guard (P1): never send raw LaTeX residue to the
                        # LLM.  LaTeX residue is ONE known trigger of
                        # content-filter empty responses — but empty responses
                        # are mostly TRANSIENT provider-side events (2026-08-14
                        # replay: 0/3 back-to-back vs 6/6 spaced 8 s), so the
                        # guard is hygiene, not the whole fix.  Cleans the chunk
                        # on the fly; returns None (chunk skipped, logged) when
                        # residue survives re-cleaning.
                        send_chunk = self._latex_guard(lit_id, chunk)
                        if send_chunk is None:
                            return None
                        result = await self.llm.complete_json(
                            "You are a materials science expert. Extract structured data as JSON.",
                            prompt_template.replace("{text}", send_chunk),
                            max_tokens=max_tokens,
                            temperature=0.1,
                            stage=self.name,
                        )
                        # Per-slot throttle: 2 s minimum inter-request gap
                        await asyncio.sleep(2.0)
                        return result
                    except ValueError as e:
                        err_msg = str(e)
                        # Empty response = TRANSIENT provider-side event
                        # (2026-08-14 replay: 0/3 back-to-back → 6/6 spaced
                        # 8 s).  Retry the SAME text after a pause — never
                        # shorten: the chunk content is fine, the provider was
                        # momentarily empty.  (tenacity in llm.py normally
                        # retries first; this branch is a defensive fallback.)
                        if attempt == 1 and "empty response" in err_msg.lower():
                            self.log(
                                f"  Empty response on chunk (attempt 1) — "
                                f"same-text retry after 15 s...",
                                "yellow",
                            )
                            await asyncio.sleep(15.0)
                            return await _extract_chunk(chunk, attempt=2)
                        if attempt == 1 and (
                            "truncat" in err_msg.lower()
                            or "non-JSON" in err_msg.lower()
                        ):
                            # Retry with a genuinely shorter chunk to avoid
                            # truncation / token overflow.
                            # Use min(len(chunk)//2, 1500) so it always shrinks.
                            # Log detailed diagnostics first: the ValueError message
                            # already contains the LLM's raw output preview when the
                            # response was non-JSON, so it must NOT be truncated away.
                            dump_path = self._dump_failed_chunk(
                                lit_id, chunk, err_msg,
                                llm_output=getattr(e, "raw_content", None),
                            )
                            self.log(
                                f"  Chunk extraction failed — {err_msg[:300]}\n"
                                f"    {self._chunk_features(chunk)}\n"
                                f"    preview: \"{chunk[:150].replace(chr(10), ' ')}\""
                                + (f"\n    dumped to: {dump_path}" if dump_path else ""),
                                "yellow",
                            )
                            short_len = min(len(chunk) // 2, 1500)
                            if short_len < len(chunk):
                                cut = self._smart_cut(chunk, short_len)
                                self.log(
                                    f"  Retrying with shorter chunk "
                                    f"({len(chunk)}→{len(cut)} chars, "
                                    f"cut at safe boundary)",
                                    "dim",
                                )
                                return await _extract_chunk(cut, attempt=2)
                            else:
                                self.log(
                                    f"  Chunk too short ({len(chunk)} chars) to shorten further, "
                                    f"skipping retry",
                                    "dim",
                                )
                        raise
                    except Exception as e:
                        # Tenacity (llm.py @retry) wraps ValueError/RateLimitError
                        # in RetryError after 3 failed retries.  We must unwrap to
                        # determine the real cause and apply the right recovery.
                        err_type = type(e).__name__
                        err_str = str(e)
                        is_rate_limit = (
                            "RateLimitError" in err_type
                            or "RateLimitError" in err_str
                            or "rate_limit" in err_str.lower()
                            or "429" in err_str
                        )
                        if is_rate_limit and attempt <= 2:
                            # Progressive backoff: 30 s on 1st retry, 60 s on 2nd
                            backoff_s = 30 if attempt == 1 else 60
                            self.log(
                                f"  Rate limit hit on chunk (attempt {attempt}), "
                                f"backing off {backoff_s}s...",
                                "yellow",
                            )
                            await asyncio.sleep(backoff_s)
                            return await _extract_chunk(chunk, attempt=attempt + 1)

                        # Check if the underlying error is a ValueError (RetryError wrapper)
                        is_value_error = (
                            "ValueError" in err_type or "ValueError" in err_str
                        )
                        if is_value_error and attempt == 1:
                            # Tenacity (llm.py) already retried SAME-text 3x
                            # with exp backoff, then wrapped the ValueError in
                            # RetryError.  Distinguish the real cause:
                            #   empty response → TRANSIENT provider-side event
                            #     (2026-08-14 replay: 0/3 back-to-back → 6/6
                            #     spaced 8 s).  One more SAME-text attempt after
                            #     a longer pause rides out longer failure
                            #     windows — NEVER shorten a clean chunk.
                            #   non-JSON / truncation → genuinely shorten.
                            # Log detailed diagnostics first: unwrap the
                            # RetryError to see the REAL cause, and dump the
                            # offending chunk for offline analysis.
                            underlying = self._unwrap_error(e)
                            detail = (
                                f"{type(underlying).__name__}: "
                                f"{str(underlying)[:300]}"
                            )
                            dump_path = self._dump_failed_chunk(
                                lit_id, chunk, detail,
                                llm_output=getattr(underlying, "raw_content", None),
                            )
                            self.log(
                                f"  Chunk extraction failed (attempt 1) — {detail}\n"
                                f"    {self._chunk_features(chunk)}\n"
                                f"    preview: \"{chunk[:150].replace(chr(10), ' ')}\""
                                + (
                                    f"\n    dumped to: {dump_path}"
                                    if dump_path else ""
                                ),
                                "yellow",
                            )
                            if "empty response" in str(underlying).lower():
                                self.log(
                                    f"  Empty response persisted after tenacity "
                                    f"retries — one more SAME-TEXT attempt after "
                                    f"15 s (no shortening)",
                                    "yellow",
                                )
                                await asyncio.sleep(15.0)
                                return await _extract_chunk(chunk, attempt=2)
                            short_len = min(len(chunk) // 2, 1500)
                            if short_len < len(chunk):
                                cut = self._smart_cut(chunk, short_len)
                                self.log(
                                    f"  Retrying with shorter chunk "
                                    f"({len(chunk)}→{len(cut)} chars, "
                                    f"cut at safe boundary)",
                                    "dim",
                                )
                                return await _extract_chunk(
                                    cut, attempt=2
                                )
                            else:
                                self.log(
                                    f"  Chunk too short ({len(chunk)} chars) to "
                                    f"shorten further, skipping retry",
                                    "dim",
                                )
                        raise

            # Fire all chunk extractions in parallel (bottlenecked by semaphore)
            chunk_tasks = [_extract_chunk(chunk) for chunk in chunks]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

            for i, result in enumerate(chunk_results):
                if result is None:
                    # latex-guard skip: chunk still had LaTeX residue after
                    # re-cleaning — already logged + dumped by _latex_guard.
                    continue
                if isinstance(result, Exception):
                    # Unwrap tenacity RetryError to surface the real failure cause
                    underlying = self._unwrap_error(result)
                    err_type = type(underlying).__name__
                    err_str = str(underlying)[:300]
                    if "empty response" in err_str.lower():
                        preview = chunks[i][:100].replace("\n", " ")
                        hint = (
                            f" (transient provider-side empty response; "
                            f"preview: \"{preview}...\")"
                        )
                    elif "non-JSON" in err_str.lower() or "truncat" in err_str.lower():
                        hint = " (max_tokens may be insufficient)"
                    else:
                        hint = ""
                    self.log(
                        f"  Chunk {i} extraction failed [{err_type}]: {err_str}{hint}",
                        "yellow",
                    )
                    continue
                record = self._parse_record(lit_id, result, len(records))
                if record and record.material_composition:
                    records.append(record)
        else:
            # Regex-based fallback (deep only)
            if depth == "deep":
                record = self._regex_extract(lit_id, full_text)
                if record and record.material_composition:
                    records.append(record)

        # Deduplicate by material_composition
        seen_materials = set()
        unique = []
        for r in records:
            if r.material_composition and r.material_composition not in seen_materials:
                seen_materials.add(r.material_composition)
                unique.append(r)

        # Stamp real DOI on all records for downstream evidence traceability
        if doi:
            for r in unique:
                r.doi = doi

        return unique

    # ── Legacy sequential extraction (kept for backward compatibility) ──
    async def _extract_from_document(
        self, lit_id: str, full_text: str, depth: str = "deep"
    ) -> list[KnowledgeRecord]:
        """Sequential chunk extraction — deprecated, use _extract_from_document_parallel."""
        return await self._extract_from_document_parallel(
            lit_id, full_text, depth, asyncio.Semaphore(1)
        )

    # ── Diagnostics helpers for chunk extraction failures ──

    @staticmethod
    def _unwrap_error(e: Exception) -> Exception:
        """Unwrap tenacity RetryError to surface the underlying exception.

        tenacity raises RetryError after exhausting retries; the real error
        (e.g. ValueError containing the LLM's raw output preview) is nested
        in ``last_attempt``.  Falling back to ``__cause__`` covers other wrappers.
        """
        if hasattr(e, "last_attempt"):
            try:
                return e.last_attempt.exception()
            except Exception:
                pass
        return getattr(e, "__cause__", None) or e

    @staticmethod
    def _chunk_features(chunk: str) -> str:
        """Summarize suspicious content features of a chunk (for diagnostics).

        Content-filter rejections are often triggered by specific patterns:
        non-ASCII chars, LaTeX command remnants, HTML tags, control chars,
        or Unicode sub/superscripts (chemical formulas).
        """
        import unicodedata
        non_ascii = [c for c in chunk if ord(c) > 127]
        latex_cmds = len(re.findall(r"\\[a-zA-Z]+\{", chunk))
        html_tags = len(re.findall(r"<[^>]+>", chunk))
        ctrl_chars = len(
            [
                c for c in chunk
                if unicodedata.category(c) == "Cc" and c not in "\n\r\t"
            ]
        )
        unicode_sub = len(re.findall(r"[\u2080-\u209C]", chunk))
        unicode_sup = len(re.findall(r"[\u2070-\u209F\u00B2\u00B3\u00B9]", chunk))
        return (
            f"len={len(chunk)}, non_ascii={len(non_ascii)}, "
            f"latex_cmds={latex_cmds}, html_tags={html_tags}, "
            f"ctrl_chars={ctrl_chars}, unicode_sub={unicode_sub}, "
            f"unicode_sup={unicode_sup}"
        )

    # ── P1: LaTeX pre-flight guard & safe-boundary chunk cutting ──

    @staticmethod
    def _latex_fingerprint(chunk: str) -> int:
        """Count LaTeX command residues (backslash followed by a letter)."""
        return len(re.findall(r"\\[a-zA-Z]+", chunk))

    def _latex_guard(self, lit_id: str, chunk: str) -> str | None:
        """Pre-flight LaTeX check before a chunk is sent to the LLM.

        Raw LaTeX residue OUTSIDE formula blocks (bare ``\\begin{array}``
        blocks, spaced commands like ``\\mathrm {N a}``) is a deterministic
        trigger of content-filter empty responses — the prompt claims the
        text is clean while it is full of LaTeX.  This guard:

        1. Returns the chunk untouched if it has no LaTeX fingerprint
           OUTSIDE ``$...$`` / ``$$...$$`` blocks (B1 keeps only REAL-MATH
           formula LaTeX intact by design; chemistry formulas are already
           plain text, so math-block-internal commands are the only
           legitimate residue).
        2. Re-runs the upstream cleaner on the chunk; returns the cleaned
           text if residue outside formula blocks is gone.
        3. Otherwise dumps the chunk, logs a ``[latex-guard]`` skip notice
           and returns None — the caller skips this chunk entirely, instead
           of letting the LLM burn 3 deterministic empty-response retries.

        Returns:
            Cleaned chunk text, or None when the chunk must be skipped.
        """
        if not re.search(r"\\[a-zA-Z]+", KnowledgeExtractionAgent._math_outside(chunk)):
            return chunk  # no residue outside formula blocks
        cleaned = self._clean_extraction_text(chunk)
        # Second pass on PROSE only: strip bare commands that lost their $
        # delimiters (the upstream cleaner's $-pass can't see them).  Formula
        # blocks are re-protected so they survive this pass untouched.
        cleaned = KnowledgeExtractionAgent._reclean_outside_math(cleaned)
        residue = re.findall(
            r"\\[a-zA-Z]+", KnowledgeExtractionAgent._math_outside(cleaned)
        )
        if residue:
            dump_path = None
            if getattr(self, "_debug_dir", None):
                dump_path = self._dump_failed_chunk(
                    lit_id,
                    chunk,
                    f"latex_guard: {len(residue)} LaTeX commands remain after "
                    f"re-clean (e.g. {residue[:3]!r}) — chunk skipped",
                )
            self.log(
                f"  [latex-guard] {len(residue)} LaTeX residues survived "
                f"re-cleaning, skipping chunk"
                + (f" (dumped to: {dump_path})" if dump_path else ""),
                "yellow",
            )
            return None
        self.log(
            f"  [latex-guard] chunk cleaned on the fly "
            f"({len(chunk)}→{len(cleaned)} chars, "
            f"{len(re.findall(r'\\\\[a-zA-Z]+', KnowledgeExtractionAgent._math_outside(chunk)))} "
            f"LaTeX cmds removed)",
            "dim",
        )
        return cleaned

    @staticmethod
    def _smart_cut(chunk: str, max_len: int) -> str:
        """Cut a chunk at the nearest safe boundary (length ≤ max_len).

        Priority (in order):
        1. If an unclosed ``\\begin{env}`` appears past the last ``\\end{env}``
           in the window, back up to the end of that complete environment —
           never cut mid-formula (the old hard ``chunk[:1500]`` cut could land
           inside a LaTeX block and STILL produce garbage).
        1b. If a ``$$`` formula block is unclosed inside the window, back up
           to the last ``$$`` so the kept prefix never holds a truncated
           formula (B1 keeps REAL-MATH blocks verbatim — a half block would
           fail the guard and be dropped).
        2. Last sentence boundary (。！？ / ". " / "! " / "? " / newline).
        3. Last space (word boundary).
        4. Hard cut as a final fallback.
        """
        if len(chunk) <= max_len:
            return chunk
        window = chunk[:max_len]
        # (1) avoid cutting inside an open LaTeX environment
        last_ends = [m.end() for m in re.finditer(r"\\end\{[^}]*\}", window)]
        if last_ends and re.search(r"\\begin\{[^}]*\}", window[last_ends[-1]:]):
            return chunk[: last_ends[-1]].rstrip()
        # (1b) avoid cutting inside a $$...$$ formula block (odd count of $$
        #      in the window ⇒ an unclosed block opened before the cut point)
        if window.count("$$") % 2 == 1:
            idx = window.rfind("$$")
            if idx > max_len * 0.3:
                return chunk[:idx].rstrip()
        # (2) sentence boundary
        for sep in ("。", "！", "？", ". ", "! ", "? ", "\n"):
            idx = window.rfind(sep)
            if idx > max_len * 0.5:
                return window[: idx + len(sep)].rstrip()
        # (3) last space
        idx = window.rfind(" ")
        if idx > max_len * 0.5:
            return window[:idx].rstrip()
        # (4) hard cut
        return window.rstrip()

    def _dump_failed_chunk(
        self, lit_id: str, chunk: str, detail: str,
        llm_output: str | None = None,
    ) -> str | None:
        """Dump a failed chunk to a debug file for offline analysis.

        When ``llm_output`` is provided, the LLM's FULL raw response is
        appended too (the error message only carries a 500-char preview,
        which is usually not enough to tell "unescaped char in string"
        apart from "unclosed JSON structure").

        Returns the file path, or None if the debug dir is unavailable.
        """
        if not self._debug_dir:
            return None
        try:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%H%M%S")
            path = self._debug_dir / f"chunk_{lit_id}_{ts}.txt"
            content = (
                f"# Failed chunk ({lit_id}) at {datetime.now().isoformat()}\n"
                f"# error: {detail}\n"
                f"# length: {len(chunk)} chars\n"
                f"# {'=' * 60}\n"
                f"{chunk}\n"
            )
            if llm_output:
                content += (
                    f"\n# {'=' * 60}\n"
                    f"# LLM raw output ({len(llm_output)} chars)\n"
                    f"# {'=' * 60}\n"
                    f"{llm_output}\n"
                )
            path.write_text(content, encoding="utf-8")
            return str(path)
        except OSError:
            return None

    # ── L2: Cross-record checks (R9 Arrhenius + R10 statistical outlier) ──

    @staticmethod
    def _arrhenius_check(
        records: list[KnowledgeRecord],
    ) -> dict[str, list[str]]:
        """R9: Cross-record Arrhenius plausibility check.

        For each **material composition** (not family!) with ≥4 (T, σ) pairs
        spanning >50 K, fit ln(σ) = a + b·(1/T).  Flag records when:

        * Fitted Ea = −b·k_B is physically implausible
          (<0.05 eV or >1.5 eV) → all points in the group get REVIEW.
        * An individual point's residual |ln(σ_actual) − ln(σ_predicted)|
          exceeds 2.0 (≈7.4× deviation) → that point gets REVIEW.

        Grouping by material composition (not family) is essential: the
        Arrhenius equation σ = σ₀·exp(−Ea/kT) applies to a single material.
        Mixing different materials (e.g. Ga-doped LLZO and Li3BO3) in one
        fit produces a meaningless slope and false-positive flags.

        Returns ``{record_id: [issue_strings]}``.
        """
        k_B = 8.617e-5  # eV/K

        # Group (T, σ, record) by normalized material composition
        groups: dict[str, list[tuple[float, float, KnowledgeRecord]]] = defaultdict(list)
        for r in records:
            if not r.ionic_conductivity or r.ionic_conductivity.value is None:
                continue
            sigma = r.ionic_conductivity.value
            if sigma <= 0:
                continue
            # Temperature: prefer ionic_conductivity.temperature_K, then test_temperature
            temp = r.ionic_conductivity.temperature_K
            if temp is None or temp <= 0:
                if r.test_temperature and r.test_temperature.value and r.test_temperature.value > 0:
                    temp = r.test_temperature.value
            if temp is None or temp <= 0:
                continue

            comp_key = re.sub(r"\s+", "", (r.material_composition or "unknown").lower())
            groups[comp_key].append((temp, sigma, r))

        issues: dict[str, list[str]] = {}

        for comp_key, pts in groups.items():
            if len(pts) < 4:
                continue
            temps = [p[0] for p in pts]
            if max(temps) - min(temps) < 50.0:  # need >50 K spread
                continue

            # Least-squares fit: ln(σ) = a + b·(1/T)
            xs = [1.0 / t for t in temps]
            ys = [math.log(p[1]) for p in pts]
            n = len(pts)
            sx, sy = sum(xs), sum(ys)
            sxy = sum(x * y for x, y in zip(xs, ys))
            sx2 = sum(x * x for x in xs)
            denom = n * sx2 - sx * sx
            if abs(denom) < 1e-30:
                continue
            slope = (n * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n
            ea = -slope * k_B  # eV

            if ea < 0.05 or ea > 1.5:
                for _t, _s, r in pts:
                    issues.setdefault(r.id, []).append(
                        f"R9: Arrhenius fit for {comp_key} gives Ea={ea:.3f} eV "
                        f"(plausible 0.05–1.5 eV) — possible temperature/unit error"
                    )
            else:
                for t, s, r in pts:
                    predicted = intercept + slope * (1.0 / t)
                    residual = math.log(s) - predicted
                    if abs(residual) > 2.0:
                        issues.setdefault(r.id, []).append(
                            f"R9: Conductivity {s:.2e} S/cm at {t:.0f}K deviates "
                            f"from Arrhenius trend by {abs(residual):.1f} ln-units "
                            f"(≈{math.exp(abs(residual)):.0f}×) — verify extraction"
                        )

        return issues

    @staticmethod
    def _statistical_outlier_check(
        records: list[KnowledgeRecord],
    ) -> dict[str, list[str]]:
        """R10: Cross-record statistical outlier detection in log-space.

        For each material family with ≥5 conductivity values, compute the
        median and MAD of log10(σ).  Flag records where::

            |log10(σ) − median| > max(3.0 × MAD, 1.5)

        The 3.0×MAD factor corresponds to ≈2σ in a normal distribution.
        The 1.5-order floor prevents false positives when the group is
        tightly clustered (different doping compositions within the same
        family can legitimately span 1 order of magnitude).

        Returns ``{record_id: [issue_strings]}``.
        """
        groups: dict[str, list[tuple[float, KnowledgeRecord]]] = defaultdict(list)
        for r in records:
            if not r.ionic_conductivity or r.ionic_conductivity.value is None:
                continue
            sigma = r.ionic_conductivity.value
            if sigma <= 0:
                continue
            family = KnowledgeExtractionAgent._detect_material_family(r.material_composition)
            groups[family].append((sigma, r))

        issues: dict[str, list[str]] = {}

        for family, pts in groups.items():
            if len(pts) < 5:
                continue

            log_vals = sorted(math.log10(s) for s, _ in pts)
            n = len(log_vals)
            mid = n // 2
            median = log_vals[mid] if n % 2 == 1 else (log_vals[mid - 1] + log_vals[mid]) / 2.0

            abs_devs = sorted(abs(lv - median) for lv in log_vals)
            mad = abs_devs[mid] if n % 2 == 1 else (abs_devs[mid - 1] + abs_devs[mid]) / 2.0

            threshold = max(3.0 * mad, 1.5)  # ≥1.5 orders of magnitude

            for sigma, r in pts:
                dev = abs(math.log10(sigma) - median)
                if dev > threshold:
                    issues.setdefault(r.id, []).append(
                        f"R10: Conductivity {sigma:.2e} S/cm is a statistical "
                        f"outlier for {family} family (deviates {dev:.1f} orders "
                        f"from median {10 ** median:.2e} S/cm, threshold {threshold:.1f})"
                    )

        return issues

    def _write_quality_report(
        self, verified: list[KnowledgeRecord], flagged: list[KnowledgeRecord]
    ) -> str | None:
        """Persist the three-way quality verdict to ``quality_report.json``.

        Written next to the agent log file (the run's log directory). Each
        REVIEW/FAIL entry carries the record id, material, detected family,
        status, human-readable issues (including R6 unit-confusion correction
        hints) and the key numeric values — enough for a human to adjudicate
        without re-running the pipeline.

        Returns the file path, or None when logging is disabled / write fails.
        """
        if not self._log_path:
            return None
        try:
            out = Path(self._log_path).parent / "quality_report.json"

            def entry(r: KnowledgeRecord) -> dict:
                return {
                    "id": r.id,
                    "literature_id": r.literature_id,
                    "material_composition": r.material_composition,
                    "family": self._detect_material_family(r.material_composition),
                    "status": r.quality_status,
                    "issues": r.quality_issues,
                    "ionic_conductivity_S_cm": (
                        r.ionic_conductivity.value if r.ionic_conductivity else None
                    ),
                    "electrochemical_window_V": (
                        r.electrochemical_window.value
                        if r.electrochemical_window else None
                    ),
                    "test_temperature_K": (
                        r.test_temperature.value if r.test_temperature else None
                    ),
                    "key_findings": (r.key_findings[:200] if r.key_findings else None),
                }

            n_review = sum(1 for r in flagged if r.quality_status == "review")
            n_fail = sum(1 for r in flagged if r.quality_status == "anomaly")
            payload = {
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "pass": len(verified),
                    "review": n_review,
                    "fail": n_fail,
                },
                "review_records": [
                    entry(r) for r in flagged if r.quality_status == "review"
                ],
                "fail_records": [
                    entry(r) for r in flagged if r.quality_status == "anomaly"
                ],
            }
            out.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.log(
                f"Quality report written: {out.name} "
                f"({len(verified)} pass / {n_review} review / {n_fail} fail)"
            )
            return str(out)
        except OSError as e:
            self.log(f"Quality report write failed: {e}", "yellow")
            return None

    def _quality_check(
        self, records: list[KnowledgeRecord], depth: str = "deep"
    ) -> tuple[list[KnowledgeRecord], list[KnowledgeRecord]]:
        """Step 7a: Three-way quality gate — PASS / REVIEW / FAIL.

        Returns (verified, flagged). Flagged records carry
        quality_status="review" (soft issues: suspected unit confusion,
        family-window outliers, unusually-high values) or "anomaly"
        (hard rule violations: absolute range breach >2 orders off the
        family window, missing raw_quote, no numeric data).

        Hard vs soft keeps the two-tuple contract used by the run cache,
        verify_quality_fix.py and test_extraction_lit0016.py — callers that
        only care about "not verified" still work unchanged.

        Rules are depth-aware to avoid over-strict rejection:
        - light extraction (auxiliary literature) is explicitly told NOT to
          emit raw_quote, so that requirement only applies to deep records;
        - light records with only a material_composition (reviews, mechanism /
          characterization papers without a single numeric value) still count
          as valid data points.
        """
        verified = []
        flagged = []  # review + fail, keeps the old "anomalies" semantics
        rules = self.quality_rules
        is_light = depth != "deep"

        for record in records:
            hard_issues = []  # → FAIL (anomaly)
            soft_issues = []  # → REVIEW

            family = self._detect_material_family(record.material_composition)
            fam_rule = self._FAMILY_CONDUCTIVITY.get(family)

            # Check conductivity plausibility
            if record.ionic_conductivity and record.ionic_conductivity.value is not None:
                v = record.ionic_conductivity.value
                low, high = rules["conductivity_range"]
                if v < low or v > high:
                    hard_issues.append(
                        f"Conductivity {v:.2e} S/cm out of absolute plausible range "
                        f"{self._format_range((low, high))}"
                    )
                elif fam_rule:
                    # R5: material-family window
                    (f_low, f_high), f_warn, label = fam_rule
                    if v < f_low or v > f_high:
                        hint = self._unit_confusion_hint(v, (f_low, f_high))
                        if hint:
                            # R6: a milli-prefix rescale lands inside the
                            # window → likely extraction unit mix-up
                            soft_issues.append(hint)
                        elif v > 0 and (v / f_high > 100 or f_low / v > 100):
                            hard_issues.append(
                                f"Conductivity {v:.2e} S/cm implausible for {label} "
                                f"(expected {self._format_range((f_low, f_high))} S/cm, "
                                f"off by >2 orders of magnitude)"
                            )
                        else:
                            soft_issues.append(
                                f"Conductivity {v:.2e} S/cm outside typical {label} "
                                f"window {self._format_range((f_low, f_high))} S/cm — "
                                f"verify LLM extraction"
                            )
                    elif v > f_warn:
                        soft_issues.append(
                            f"Conductivity {v:.2e} S/cm unusually high for {label} — "
                            f"verify LLM extraction"
                        )
                else:
                    # Unknown family: fall back to the global warn threshold
                    warn_above = rules.get("conductivity_warn_above", 0.5)
                    if v > warn_above:
                        soft_issues.append(
                            f"Conductivity {v:.2e} S/cm unusually high for solid "
                            f"electrolytes — verify LLM extraction"
                        )

                # Must have raw_quote — enforced ONLY for deep extraction;
                # the light prompt explicitly skips raw_quote to save tokens.
                if not is_light and not record.ionic_conductivity.raw_quote:
                    hard_issues.append("Conductivity missing raw_quote")

            # Check temperature plausibility
            for temp_field, temp_name in [
                (record.sintering_temperature, "sintering_temperature"),
                (record.test_temperature, "test_temperature"),
            ]:
                if temp_field and temp_field.value is not None:
                    v = temp_field.value
                    low, high = rules["temperature_range"]
                    if v < low or v > high:
                        hard_issues.append(
                            f"{temp_name} {v} out of plausible range "
                            f"{self._format_range((low, high))}"
                        )

            # Check electrochemical window
            if record.electrochemical_window and record.electrochemical_window.value is not None:
                v = record.electrochemical_window.value
                low, high = rules["window_range"]
                if v < low or v > high:
                    # R6: mV value mislabeled as V (e.g. 3500 "V" → 3.5 V)
                    if v > high and low <= v / 1000.0 <= high:
                        soft_issues.append(
                            f"Electrochemical window {v:g} V suspected mV/V confusion — "
                            f"corrected value would be {v / 1000.0:g} V; "
                            f"check raw_quote / original_unit"
                        )
                    else:
                        hard_issues.append(
                            f"Electrochemical window {v}V out of range "
                            f"{self._format_range((low, high))}"
                        )

            # Must have at least one numeric data point. Light extraction
            # (auxiliary literature) frequently reports no single numeric value
            # — a material composition alone is still a valid light record.
            has_data = (
                record.ionic_conductivity or record.electrochemical_window
                or record.sintering_temperature or record.test_temperature
                or (is_light and record.material_composition)
            )
            if not has_data:
                hard_issues.append("No numeric data extracted")

            if hard_issues:
                record.quality_status = "anomaly"
                record.quality_issues = hard_issues + soft_issues
                flagged.append(record)
            elif soft_issues:
                record.quality_status = "review"
                record.quality_issues = soft_issues
                flagged.append(record)
            else:
                record.quality_status = "verified"
                record.quality_issues = []
                verified.append(record)

        return verified, flagged

    def _parse_record(self, lit_id: str, result: dict | list, index: int) -> KnowledgeRecord | None:
        """Parse LLM JSON output into a KnowledgeRecord."""
        if isinstance(result, list):
            result = result[0] if result else {}
        if not isinstance(result, dict):
            return None

        def make_numeric(field: dict | None) -> NumericValue | None:
            if not field or not isinstance(field, dict):
                return None
            v = field.get("value")
            if v is None:
                return None
            # LLM may return a list for a scalar field, e.g. [1.2, 3.4]
            # Take the first element; skip empty lists.
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    return None
                v = v[0]
            value = self._parse_scalar_number(v)
            if value is None:
                return None
            # Unit may carry a ×10^n coefficient, e.g. {"value": 1.0, "unit": "×10⁻³ S/cm"}
            # → the true magnitude is 1e-3 S/cm. Apply it so plausibility checks
            # see the physical value, not the mantissa.
            coef = self._parse_unit_coefficient(field.get("unit", ""))
            if coef is not None:
                value = value * coef
            return NumericValue(
                value=value,
                unit=field.get("unit", ""),
                original_value=str(v),
                original_unit=field.get("unit", ""),
                temperature_K=field.get("temperature_K"),
                raw_quote=field.get("raw_quote", ""),
            )

        def make_temp(field: dict | None) -> NumericValue | None:
            """Numeric wrapper for temperature fields: °C → K normalization."""
            return self._normalize_temp_unit(make_numeric(field))

        def make_cond(field: dict | None) -> NumericValue | None:
            """Numeric wrapper for conductivity fields: unit → S/cm normalization."""
            return self._normalize_conductivity_unit(make_numeric(field))

        return KnowledgeRecord(
            id=f"{lit_id}_rec_{index:03d}",
            literature_id=lit_id,
            material_composition=result.get("material_composition"),
            crystal_structure=result.get("crystal_structure"),
            ionic_conductivity=make_cond(result.get("ionic_conductivity")),
            electrochemical_window=make_numeric(result.get("electrochemical_window")),
            synthesis_method=result.get("synthesis_method"),
            sintering_temperature=make_temp(result.get("sintering_temperature")),
            test_temperature=make_temp(result.get("test_temperature")),
            pressure=make_numeric(result.get("pressure")),
            simulation_method=result.get("simulation_method"),
            key_findings=result.get("key_findings"),
            raw_quotes=self._collect_quotes(result),
        )

    def _regex_extract(self, lit_id: str, text: str) -> KnowledgeRecord | None:
        """Fallback regex-based extraction (no LLM available)."""
        # Try to find chemical formulas (element patterns)
        formula_pattern = r'(Li[\d.]*[A-Z][a-z]?[\d.]*)+'
        formulas = re.findall(formula_pattern, text)

        # Try to find conductivity values
        cond_pattern = r'([\d.]+)\s*[×xX]\s*10[⁻⁽]?(\d+)[⁾]?\s*(S/cm|mS/cm|S/m)'
        cond_matches = re.findall(cond_pattern, text)

        if not formulas:
            return None

        cond_value = None
        if cond_matches:
            base, exp, unit = cond_matches[0]
            cond_value = NumericValue(
                value=float(base) * 10 ** int(exp),
                unit=unit,
                original_value=f"{base}×10^{exp}",
                raw_quote=cond_matches[0] if isinstance(cond_matches[0], str) else " ".join(cond_matches[0]),
            )

        return KnowledgeRecord(
            id=f"{lit_id}_rec_000",
            literature_id=lit_id,
            material_composition=formulas[0] if formulas else None,
            ionic_conductivity=cond_value,
        )

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML/XML tags and entity references from text.

        Sciverse /content responses often contain inline tags such as
        ``<sub>7</sub>``, ``<sup>3+</sup>``, ``<i>...</i>``, etc.
        Some LLM safety filters reject prompts containing HTML as potential
        XSS injection, resulting in empty responses.

        This method strips all tags and decodes common numeric entities
        so the LLM sees clean plain text.
        """
        import html as _html
        # Remove all XML/HTML tags: <tag>, <tag attr="val">, </tag>, <tag/>
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities like &#x2009;, &minus;, &plusmn;, etc.
        text = _html.unescape(text)
        # Collapse multiple blank lines caused by tag removal
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    # ── Upstream text cleaning (before chunking) ──

    @staticmethod
    def _clean_extraction_text(text: str) -> str:
        """Clean dirty PDF-extracted text BEFORE chunking & LLM extraction.

        Five levels, ordered by blast radius:

        1. Drop whole-line markdown image placeholders (MinerU artifacts)
           and stray page-number lines: ``![](dt=.../hash.jpg)`` — the LLM
           sometimes copies these into raw_quote / key_findings verbatim,
           which both pollutes records AND breaks JSON string escaping.
        2. LaTeX handling (B1 formula classification):
           - $-wrapped formula blocks are CLASSIFIED: real math (Greek
             letters, ``\\frac``/``\\sum``/``\\int``, ``\\mathbb``/calligraphic
             styles, operators) is PROTECTED (placeholder ``⟦MATH{i}⟧``) and
             restored verbatim at the very end — the LLM reads math structure
             natively.  Chemistry formulas (``Li7La3Zr2O12`` — the ~90%
             materials-science case) are converted to plain text on the
             spot: lossless AND ~60% fewer tokens than keeping LaTeX.
           - everything else is converted to plain text: bare multi-line
             environments (``\\begin{array}...\\end{array}`` without a ``$``
             wrapper — the MinerU fallback that used to reach the LLM raw),
             ``\\(...\\)``, stray ``\\%`` escapes, and residual bare
             commands that lost their delimiters (``\\mathrm {N a} _ {2}``).
             Unconverted backslashes inside JSON strings (``\\m`` is not a
             valid escape) are a direct cause of "non-JSON content" failures.
        3. Heuristically re-join column-merged table rows (insert `` | ``):
           the PDF text layer merges cells into one run-on line, e.g.
           ``...FAST1150°C, 10 min99.85.7``; splitting cells makes the data
           readable so the LLM quotes single cells instead of the whole row.
           Table-GLUE runs (MinerU flattening an entire table into one
           space-less alnum token, ``powderAl6.96.77--Al7.7...``) are
           undecodable AND a deterministic empty-response trigger — they
           are DROPPED (the chunk_lit_0004_102345 failure).
        4. Restore protected formula blocks verbatim (LAST — nothing above
           may touch them).

        Returns the cleaned text (same length class, all tokens preserved —
        no information is dropped, only reformatted / separated).
        """
        if not text:
            return text

        # 1) Handle markdown image placeholders + stray page numbers.
        # P3-16: retain the figure caption text instead of discarding the whole
        # line — the caption often carries the only legible description of a
        # figure/plot, and dropping it loses information the LLM could use.
        lines = []
        for ln in text.splitlines():
            s = ln.strip()
            m = re.fullmatch(r"!\[([^\]]*)\]\([^)]*\)", s)
            if m:
                caption = m.group(1).strip()
                if caption:
                    lines.append(f"图注: {caption}")
                continue  # rendering artifact / empty placeholder — no prose
            # MinerU image rows may carry a trailing metadata suffix after the
            # closing paren, e.g. "![](image)ult=success/type=image/dt=.../hash.jpg".
            # Any line that STARTS with an image placeholder is a rendering
            # artifact with no prose — drop it (normal prose never starts with "![").
            if s.startswith("![") and "]" in s and "(" in s and ")" in s:
                continue
            # Stray page-number lines ("77", "124") left by the PDF text layer
            # (usually with a journal-header line right above).  A bare 1-3
            # digit line is never prose; numbered lists always carry a dot.
            if re.fullmatch(r"\d{1,3}", s):
                continue
            lines.append(ln)
        text = "\n".join(lines)

        # 2) LaTeX handling — B1 formula classification:
        #    (a) $-wrapped blocks are CLASSIFIED: real math (\\frac, Greek
        #        letters, \\mathbb, operators...) is PROTECTED and restored
        #        verbatim — the LLM natively understands LaTeX math, so
        #        converting it to "a/b" loses semantics (\\frac{dσ}{dT} vs
        #        σ/T differ).  Chemistry formulas (\\mathrm{Li}_7... — the
        #        ~90% materials-science case) are converted to plain text
        #        INLINE by _protect_math_blocks: lossless + ~60% fewer
        #        tokens.  The P0 cleaning chain below runs on prose ONLY.
        #    (b) everything outside those blocks keeps the full P0 treatment:
        #        bare multi-line environments ($$-less MinerU fallback),
        #        \(...\), stray \% and residual bare commands.
        # 2a) Protect/convert $$...$$ FIRST (the $...$-regex would otherwise
        #     mis-eat the double-dollar delimiters and truncate the block to
        #     its subscript tail — the original empty-response bug).
        text, math_blocks = KnowledgeExtractionAgent._protect_math_blocks(text)
        # 2b) Bare multi-line environments WITHOUT a $ wrapper (MinerU
        #     fallback) — these used to survive untouched and reach the LLM,
        #     which then hit a prompt contradiction ("text is already cleaned"
        #     vs. full LaTeX) and returned empty responses (content filter).
        #     Only $$-less envs reach this pass: $-wrapped ones are already
        #     inside math_blocks placeholders.
        text = KnowledgeExtractionAgent._convert_bare_latex_envs(text)
        # 2c) Residual $-wrapped segments that were NOT protected (plain-text
        #     pseudo-blocks like "$10 per mol$" — no LaTeX commands, so not a
        #     formula) plus \(...\) wrappers: convert both to plain text.
        text = re.sub(
            r"\$([^$]*)\$",
            lambda m: KnowledgeExtractionAgent._latex_segment_to_text(m.group(1)),
            text,
        )
        text = re.sub(
            r"\\\((.+?)\\\)",
            lambda m: KnowledgeExtractionAgent._latex_segment_to_text(m.group(1)),
            text,
        )
        # 2c2) Every $ still alive after 2c is an ORPHAN: paired $$...$$ / $...$
        #      blocks were protected, converted, or flattened above (real math
        #      is now a ⟦MATH{i}⟧ placeholder, which contains no $).  Orphans
        #      come from MinerU table headers ("$SampleAtomic Ratio...") and
        #      are pure noise that confuses the LLM — drop them before restore.
        text = text.replace("$", "")
        # 2d) Stray \% escapes left in prose (e.g. "37\%" — \ followed by a
        #     non-letter is not a LaTeX command remnant, just PDF junk).
        text = text.replace("\\%", "%")
        # 2e) Safety net: strip residual bare LaTeX commands that lost their
        #     $ delimiters during PDF extraction (never touches normal prose).
        #     Runs while formula blocks are still placeholders, so it can
        #     never damage them.
        text = KnowledgeExtractionAgent._strip_bare_latex(text)

        # 3) Heuristic table-row repair (only for merged-table signature lines)
        out_lines = []
        for ln in text.splitlines():
            s = ln.strip()
            if (
                ("%" in s)
                and ("×10" in s or "x10" in s)
                and s.count("°C") >= 2
            ):
                s = KnowledgeExtractionAgent._repair_merged_table_row(s)
            out_lines.append(s if s else ln)
        text = "\n".join(out_lines)
        # 3b) Table-flattening glue runs (MinerU): an entire table collapsed
        #     into one space-less alnum token ("powderAl6.96.77...--Al7.7...")
        #     is undecodable AND a deterministic empty-response trigger (the
        #     chunk_lit_0004_102345 failure).  Runs with a table signature
        #     (-- or >=35% digits, 40+ chars) are dropped; prose is never
        #     touched.  Runs while formula blocks are still placeholders.
        text = KnowledgeExtractionAgent._strip_table_glue(text)

        # 4) Restore the protected formula blocks verbatim — this MUST run
        #    last so no cleaning pass ever touches them.  From here on the
        #    text may legitimately contain LaTeX inside $...$ / $$...$$.
        text = KnowledgeExtractionAgent._restore_math_blocks(text, math_blocks)
        return text

    # ── LaTeX math → plain text (wrapped AND bare-env forms) ──

    # Placeholder sentinels used to protect $-wrapped REAL-MATH formula
    # blocks while the P0 cleaning chain runs on prose only (A1 + B1).
    _MATH_PH = "⟦MATH{i}⟧"

    # Math-only LaTeX commands that mark a $...$ block as REAL math (B1
    # formula classification).  Materials-science papers are ~90% chemistry
    # formulas (Li7La3Zr2O12): converting those to plain text is lossless
    # AND ~60% cheaper in tokens than keeping the LaTeX.  Real math must
    # keep its LaTeX — the LLM reads it natively, and flattening e.g.
    # \sum_{i=1}^{n} to "i=1n" loses the operator entirely.
    #
    # Only STRUCTURE / FUNCTION commands live here: fractions, roots,
    # sums/integrals, letter styles, calculus, named functions.  Commands
    # that map losslessly to a single Unicode glyph (Greek letters, ± ≤ ≈,
    # \circ \times \mathrm, \left/\right, \xrightarrow, \tag, env
    # wrappers) live in _LATEX_SYMBOL_MAP or are handled inline — they do
    # NOT trigger protection.
    _MATH_CMDS = frozenset({
        # fractions / roots / sums / integrals / products / limits
        "frac", "dfrac", "tfrac", "sqrt", "sum", "prod", "int", "oint",
        "lim", "limsup", "liminf",
        # letter styles whose glyph carries meaning
        "mathbb", "mathcal", "mathscr", "mathfrak", "boldsymbol",
        # calculus & named functions (sin → "sin(x)" loses the operator)
        "partial", "nabla", "infty", "log", "ln", "exp",
        "sin", "cos", "tan", "cot", "sec", "csc",
        "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
        "max", "min", "sup", "inf", "det", "dim", "ker",
    })

    # Single-glyph LaTeX commands → lossless Unicode (B1).  Greek letters
    # (XRD 2θ, α/β phases, ΔG, μm) and relational/binary operators
    # (± ≤ ≈) are everyday tokens in materials-science prose; converting
    # them costs nothing and saves tokens, so they do NOT trigger math
    # protection.  Applied by both _latex_segment_to_text and
    # _strip_bare_latex.  Keys are full command names (\alpha ...) — no
    # prefix aliases like \le/\ge/\ne are included, so order is irrelevant.
    _LATEX_SYMBOL_MAP = {
        # Greek letters (lowercase)
        "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ",
        "\\epsilon": "ε", "\\varepsilon": "ε", "\\zeta": "ζ",
        "\\eta": "η", "\\theta": "θ", "\\vartheta": "ϑ",
        "\\iota": "ι", "\\kappa": "κ", "\\lambda": "λ", "\\mu": "μ",
        "\\nu": "ν", "\\xi": "ξ", "\\pi": "π", "\\rho": "ρ",
        "\\sigma": "σ", "\\tau": "τ", "\\upsilon": "υ",
        "\\phi": "φ", "\\varphi": "φ", "\\chi": "χ", "\\psi": "ψ",
        "\\omega": "ω",
        # Greek letters (uppercase)
        "\\Gamma": "Γ", "\\Delta": "Δ", "\\Theta": "Θ",
        "\\Lambda": "Λ", "\\Xi": "Ξ", "\\Pi": "Π", "\\Sigma": "Σ",
        "\\Upsilon": "Υ", "\\Phi": "Φ", "\\Psi": "Ψ", "\\Omega": "Ω",
        # relational / binary operators
        "\\pm": "±", "\\mp": "∓", "\\div": "÷", "\\ast": "*",
        "\\star": "⋆", "\\otimes": "⊗", "\\oplus": "⊕",
        "\\leq": "≤", "\\geq": "≥", "\\neq": "≠", "\\approx": "≈",
        "\\equiv": "≡", "\\propto": "∝", "\\sim": "∼",
        "\\ll": "≪", "\\gg": "≫",
    }

    @staticmethod
    def _apply_symbol_map(text: str) -> str:
        """Map lossless single-glyph LaTeX commands to Unicode."""
        for src, dst in KnowledgeExtractionAgent._LATEX_SYMBOL_MAP.items():
            text = text.replace(src, dst)
        return text

    @staticmethod
    def _is_math_formula(body: str) -> bool:
        """Classify a ``$...$`` block body: REAL math (keep LaTeX) or not.

        True when the body contains at least one math-only command from
        ``_MATH_CMDS`` (``\\frac``, ``\\sum``, ``\\mathbb``, operators,
        ...).  False for chemistry formulas (``\\mathrm{Li}_7...`` — the
        ~90% materials-science case), which only use commands like
        ``\\mathrm``/``\\circ``/``\\times`` and convert to plain text
        losslessly via ``_latex_segment_to_text``.
        """
        return bool(
            set(re.findall(r"\\([a-zA-Z]+)", body))
            & KnowledgeExtractionAgent._MATH_CMDS
        )

    @staticmethod
    def _protect_math_blocks(text: str) -> tuple[str, list[str]]:
        """Extract REAL-MATH formula blocks; convert chemistry formulas.

        B1 formula classification replaces A1's "protect every formula":
        each ``$$...$$`` / ``$...$`` block is inspected and

        - REAL math (see ``_is_math_formula``: ``\\frac``, Greek letters,
          ``\\mathbb``, operators, ...) is PROTECTED — swapped for
          ``⟦MATH{i}⟧`` so the P0 cleaning chain below never touches it,
          restored verbatim by ``_restore_math_blocks`` at the end (the
          LLM reads math LaTeX natively);
        - chemistry formulas (``\\mathrm{Li}_7\\mathrm{La}_3...`` — the
          ~90% case in materials-science papers) are CONVERTED to plain
          text INLINE right here via ``_latex_segment_to_text``: lossless
          and ~60% fewer tokens than keeping the LaTeX;
        - plain-text pseudo-blocks (``$10 per mol$`` — no LaTeX syntax at
          all) are left untouched for the normal ``$...$`` cleaner pass.

        Returns ``(text_with_placeholders, [math_block, ...])``.
        """
        blocks: list[str] = []

        def _repl(m: re.Match) -> str:
            body = m.group(1)
            # A formula block must contain LaTeX syntax: a command (\mathrm)
            # OR a sub/superscript marker (^ / _).  Plain-text pseudo-blocks
            # like "$10 per mol$" are NOT formulas — leave them for the
            # cleaner's $-pass below.
            if not re.search(r"\\[a-zA-Z]+|[_\^]", body):
                return m.group(0)
            if KnowledgeExtractionAgent._is_math_formula(body):
                idx = len(blocks)
                blocks.append(m.group(0))
                return KnowledgeExtractionAgent._MATH_PH.format(i=idx)
            # chemistry formula → lossless plain-text conversion (B1)
            return KnowledgeExtractionAgent._latex_segment_to_text(body)

        text = re.sub(r"\$\$(.+?)\$\$", _repl, text, flags=re.DOTALL)
        text = re.sub(r"\$([^$]*)\$", _repl, text)
        return text, blocks

    @staticmethod
    def _restore_math_blocks(text: str, blocks: list[str]) -> str:
        """Put protected formula blocks back verbatim (in extraction order)."""
        for i, blk in enumerate(blocks):
            text = text.replace(KnowledgeExtractionAgent._MATH_PH.format(i=i), blk)
        return text

    @staticmethod
    def _math_outside(text: str) -> str:
        """Text with $-wrapped formula blocks stripped — for fingerprint checks.

        A1/B1 keeps formula-internal LaTeX by design, so residue fingerprints
        must only count commands OUTSIDE formula blocks (real-math formula
        commands are legitimate; chemistry formulas are already plain text).  ``$$`` is stripped first (double-dollar delimiters).
        """
        t = re.sub(r"\$\$(.+?)\$\$", "", text, flags=re.DOTALL)
        return re.sub(r"\$([^$]*)\$", "", t)

    @staticmethod
    def _reclean_outside_math(text: str) -> str:
        """Re-run the bare-LaTeX safety net on prose only.

        Used by ``_latex_guard``: REAL-MATH formula blocks are re-protected
        so ``_strip_bare_latex`` never touches them (chemistry formulas are
        already plain text by this point — B1).
        """
        protected, blocks = KnowledgeExtractionAgent._protect_math_blocks(text)
        protected = KnowledgeExtractionAgent._strip_bare_latex(protected)
        return KnowledgeExtractionAgent._restore_math_blocks(protected, blocks)

    @staticmethod
    def _repair_math_spans(chunks: list[str]) -> list[str]:
        """Re-attach formula blocks split across chunk boundaries (B1).

        Sentence splitting (``_split_sentences``) can cut a ``$$...$$``
        block — formula bodies often contain ``. ``-like patterns, e.g.
        ``2. 75 H2O`` (a decimal torn apart by PDF extraction).  A chunk
        with an odd ``$$`` count then holds a truncated block, which the
        guard would strip/skip — losing the formula.  This walks the
        chunks, keeps complete ``$$`` pairs in place, and carries any
        dangling half-block into the next chunk so blocks stay whole.

        A dangling half-block at the very end of the text is dropped
        (never send truncated math to the LLM).
        """
        fixed: list[str] = []
        carry = ""
        for chunk in chunks:
            merged = carry + chunk
            parts = merged.split("$$")
            if len(parts) % 2 == 1:
                # balanced: text starts & ends outside a formula block
                fixed.append(merged)
                carry = ""
            else:
                # dangling: the last part is an unclosed formula fragment
                head = "$$".join(parts[:-1])
                if head:
                    fixed.append(head)
                carry = "$$" + parts[-1]
        if carry:
            # trailing half-block at EOF — drop it
            pass
        return fixed

    # Text commands whose body must keep internal spaces (prose inside math).
    _LATEX_PROSE_CMDS = {
        "text", "mbox", "textrm", "textit", "textnormal", "textbf",
    }

    @staticmethod
    def _collapse_math_body(body: str) -> str:
        """Collapse whitespace inside an atom/variable command body.

        ``\\mathrm {N a}`` → ``Na`` (chemistry atom), while prose commands
        (``\\text {some words}``) are handled by the caller separately.
        """
        return re.sub(r"\s+", "", body)

    @staticmethod
    def _latex_cmd_body(m: re.Match) -> str:
        """Collapse a ``\\cmd {body}`` match: prose commands keep spaces,
        atoms don't (``\\mathrm {N a}`` → ``Na``)."""
        cmd, body = m.group(1), m.group(2)
        if cmd in KnowledgeExtractionAgent._LATEX_PROSE_CMDS:
            return body  # prose: keep internal spaces
        return KnowledgeExtractionAgent._collapse_math_body(body)

    @staticmethod
    def _latex_segment_to_text(seg: str) -> str:
        """Convert one LaTeX math segment to readable plain text.

        Covers BOTH compact (``\\mathrm{Li}_7``) and spaced
        (``\\mathrm {N a} _ {2}``) command forms that MinerU / PDF text
        layers produce, plus the operators seen in extracted formula blocks:
        ``\\xrightarrow{...}`` → ``→``, ``\\frac{a}{b}`` → ``a/b``,
        ``\\left/\\right`` → dropped, ``\\%`` → ``%``, ``\\\\`` → ``; ``.
        Environment wrappers (``\\begin{array}``) and ``\\tag`` labels are
        stripped; sub/superscript bodies tolerate one nesting level
        (``_ {4 (\\mathrm {s})}`` → ``4(s)``).
        """
        if not seg:
            return ""
        # Strip environment wrappers & equation tags (display-math blocks)
        seg = re.sub(r"\\begin\{[a-zA-Z*]+\}(?:\{[lcr]\})?", "", seg)
        seg = re.sub(r"\\end\{[a-zA-Z*]+\}", "", seg)
        seg = re.sub(r"\\tag\s*\{[^}]*\}", "", seg)
        # Named glyphs: Greek letters + relational/binary operators map to
        # Unicode losslessly (B1); \circ/\degree/\times/\cdot inline here.
        seg = KnowledgeExtractionAgent._apply_symbol_map(seg)
        seg = seg.replace("\\circ", "°").replace("\\degree", "°")
        seg = seg.replace("\\times", "×").replace("\\cdot", "·")
        seg = seg.replace("\\rightarrow", "→").replace("\\to", "→")
        seg = seg.replace("\\left", "").replace("\\right", "")
        # \xrightarrow{label} / \xrightarrow {label} → → (label may contain
        # one nested brace level, e.g. {75 - 170 ^ {\circ} \mathrm {C}})
        seg = re.sub(
            r"\\xrightarrow\s*\{((?:[^{}]*|\{[^{}]*\})*)\}", "→", seg,
        )
        # \frac{a}{b} / \frac {a} {b} → a/b
        seg = re.sub(
            r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", seg,
        )
        # \cmd {body} (spaced OR compact) → body, collapsing spaces unless the
        # command is a prose command.  Runs BEFORE the _{...}/^{...} passes so
        # \mathrm {N a} becomes "Na" atom text.
        seg = re.sub(r"\\([a-zA-Z]+)\s*\{([^{}]*)\}", KnowledgeExtractionAgent._latex_cmd_body, seg)
        # _{x} / _ {x} → x ; ^{x} / ^ {x} → x (whitespace-tolerant; one
        # nesting level allowed, e.g. _ {4 (\mathrm {s})} → 4(s))
        seg = re.sub(r"_\s*\{((?:[^{}]*|\{[^{}]*\})*)\}", lambda m: KnowledgeExtractionAgent._collapse_math_body(m.group(1)), seg)
        seg = re.sub(r"\^\s*\{((?:[^{}]*|\{[^{}]*\})*)\}", lambda m: KnowledgeExtractionAgent._collapse_math_body(m.group(1)), seg)
        # bare _7 / ^3 / _ 7 / ^ 3 → 7 / 3
        seg = re.sub(r"[_\^]\s*(?=[A-Za-z0-9])", "", seg)
        # Residual \cmd{...} not caught above (belt & braces)
        seg = re.sub(r"\\[a-zA-Z]+\s*\{([^{}]*)\}", r"\1", seg)
        seg = re.sub(r"\\[a-zA-Z]+", "", seg)  # residual bare commands
        seg = seg.replace("\\%", "%")
        # line separators: "\  \\ " → ";" (leading whitespace consumed)
        seg = re.sub(r"\s*\\\\", ";", seg)
        seg = seg.replace("&", " ")
        # stray escapes of non-letter chars: \* → *, \# → # (author marks,
        # PDF-card numbers).  NEVER touches \cmd (lookahead excludes letters).
        seg = re.sub(r"\\(?=[^a-zA-Z])", "", seg)
        seg = seg.replace("{", "").replace("}", "").replace("$", "").strip()
        # Collapse intra-LINE spaces that separate atom fragments from their
        # subscripts only: "Na 2"→"Na2", "2 B"→"2B", "(OH) 4"→"(OH)4",
        # "7 5"→"75".  Real separations between different atoms stay spaced
        # ("→ C Na2" keeps its space — N is not a digit).  Line breaks are
        # NOT collapsed (a formula may span lines).
        seg = re.sub(r"(?<=[\w\)\]\}])\s+(?=\d)", "", seg)
        seg = re.sub(r"(?<=\d)\s+(?=[A-Z(])", "", seg)
        seg = re.sub(r"(?<=\d)\s+(?=\d)", "", seg)
        # Greek-letter atoms: "\Delta G" → "ΔG", "\mu m" → "μm" (B1 symbol
        # map output keeps the original inter-command space)
        seg = re.sub(
            r"(?<=[\u03b1-\u03c9\u0391-\u03a9])\s+(?=[A-Za-z\u4e00-\u9fff])", "", seg,
        )
        return seg.strip()

    @staticmethod
    def _convert_bare_latex_envs(text: str) -> str:
        """Convert bare ``\\begin{env}...\\end{env}`` math blocks to text.

        MinerU formula blocks are emitted as standalone multi-line LaTeX
        environments WITHOUT a ``$`` wrapper, e.g.::

            \\begin{array}{l}
            \\mathrm {N a} _ {2} \\mathrm {B} _ {4} ...
            \\end{array}

        The old cleaner only handled ``$...$`` / ``\\(...\\)``, so these
        blocks survived untouched and full LaTeX reached the LLM — the direct
        cause of the "empty response" content-filter rejections.  Envs:
        array/equation/align/gathered/split/cases/matrix/pmatrix/bmatrix.

        The non-greedy match consumes nested ``\\begin`` blocks
        innermost-first, so the substitution is iterated until stable.
        """
        pattern = re.compile(
            r"\\begin\{([a-zA-Z*]+)\}(?:\{[lcr]\})?(.*?)\\end\{\1\}",
            re.DOTALL,
        )
        for _ in range(8):
            new_text = pattern.sub(
                lambda m: KnowledgeExtractionAgent._latex_segment_to_text(m.group(2)),
                text,
            )
            if new_text == text:
                break
            text = new_text
        return text

    @staticmethod
    def _strip_bare_latex(text: str) -> str:
        """Remove residual LaTeX command sequences from arbitrary prose.

        The safety net behind ``_clean_extraction_text`` Level 2: isolated
        commands that lost their ``$`` / ``$$`` delimiters during PDF
        extraction (e.g. a lone ``\\mathrm {N a}`` in mid-sentence).  Targets
        ONLY backslash commands and their ``_``/``^`` markers — normal prose
        is never touched.  Used both as the final pass of the upstream
        cleaner AND as the ``_latex_guard`` re-clean step.
        """
        text = text.replace("\\circ", "°").replace("\\times", "×")
        # Named glyphs (Greek letters, ± ≤ ≈ ...) → Unicode (B1)
        text = KnowledgeExtractionAgent._apply_symbol_map(text)
        text = re.sub(r"\\begin\{[a-zA-Z*]+\}(?:\{[lcr]\})?", "", text)
        text = re.sub(r"\\end\{[a-zA-Z*]+\}", "", text)
        text = re.sub(r"\\tag\s*\{[^}]*\}", "", text)
        text = re.sub(r"\\([a-zA-Z]+)\s*\{([^{}]*)\}", KnowledgeExtractionAgent._latex_cmd_body, text)
        # sub/superscripts: consume the separator space so "Na _ {2}" → "Na2"
        text = re.sub(r"[ \t]*_\s*\{((?:[^{}]*|\{[^{}]*\})*)\}", r"\1", text)
        text = re.sub(r"[ \t]*\^\s*\{((?:[^{}]*|\{[^{}]*\})*)\}", r"\1", text)
        text = re.sub(r"[_\^]\s*(?=[A-Za-z0-9])", "", text)
        text = re.sub(r"\\[a-zA-Z]+(?:\s*\{[^{}]*\})?", "", text)
        text = text.replace("\\%", "%")
        text = re.sub(r"\s*\\\\", ";", text)          # line separators
        text = text.replace("&", " ")
        # stray escapes of non-letter chars: \* → *, \# → # (never \cmd)
        text = re.sub(r"\\(?=[^a-zA-Z])", "", text)
        return text

    @staticmethod
    @staticmethod
    def _strip_table_glue(text: str) -> str:
        """Strip MinerU table-flattening glue runs (table-glue fix).

        MinerU sometimes flattens an entire table into ONE run-on token
        with no spaces and no column separators, e.g.::

            ...(Sigma-Aldrich, $SampleAtomic RatioLiZrAlGaTaCalcined powder
            Al_6.96.771.860.31--Al_7.77.471.860.35--Al_8.47.901.850.34--...

        The ``powderAl_6.96.77...`` run is undecodable even by a human —
        the column boundaries are gone.  Feeding it to the LLM triggers
        the content filter / empty responses (the chunk_lit_0004_102345
        failure).  Normal prose NEVER contains a 40+ char token built
        ONLY from ``[A-Za-z0-9._,-]``; URLs / DOIs contain ``/`` or ``:``
        and are excluded by the charset.  The table itself is
        unrecoverable from the flattened run, so the RUN is dropped —
        the surrounding prose (e.g. the ``Li2CO3 (KOJUNDO, 99.99%)``
        reagent list on the same line) survives.

        A run qualifies as table glue when it is 40+ chars AND carries a
        table signature: ``--`` (MinerU's empty-cell placeholder) or a
        high digit density (>=35% digits — numeric table data, never
        prose).  Runs that fail the check are left untouched.
        """
        if not text:
            return text

        def _repl(m: re.Match) -> str:
            run = m.group(0)
            if "--" in run:
                return ""
            digits = sum(1 for c in run if c.isdigit())
            if digits / len(run) >= 0.35:
                return ""
            return run

        return re.sub(r"[A-Za-z0-9._,\-]{40,}", _repl, text)

    def _repair_merged_table_row(row: str) -> str:
        """Insert ' | ' separators into a column-merged table row.

        PDF text extraction merges table header cells with data cells into a
        single run-on line, e.g.::

            Synthesis techniqueTechnologyDensity (%)Li+conductivity at RT (×10-4S/cm)conventional solid-state reaction1230°C, 36 h96.03.6sol-gel method...

        Heuristics (best-effort; no tokens are dropped, only separated):
        - digit follows a letter           → cell boundary ("h96", "min99")
        - lower→upper letter transition    → camelCase boundary ("techniqueTechnology")
        - letter follows a digit           → numeric-cell end ("96.03.6sol")
        - letter follows ')'               → unit-cell end ("S/cm)conventional")

        Only applied to lines matching the merged-table signature (see
        ``_clean_extraction_text``), so normal prose is never touched.
        """
        parts: list[str] = []
        buf = ""
        prev = ""
        for ch in row:
            if buf and prev:
                boundary = (
                    (ch.isdigit() and prev.isalpha())
                    or (ch.isupper() and prev.islower())
                    or (ch.isalpha() and prev.isdigit())
                    or (ch.isalpha() and prev == ")")
                )
                if boundary:
                    parts.append(buf)
                    buf = ""
            buf += ch
            prev = ch
        if buf:
            parts.append(buf)
        return " | ".join(p for p in parts if p)

    # ── Semantic chunking: sentence-level split + section boundary detection ──

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into individual sentences on punctuation boundaries.

        Handles both Chinese (。！？) and English (.!?) sentence endings.
        Protects common abbreviations (e.g., Fig., et al.) from false splits.
        """
        if not text:
            return []

        # Step 1: protect abbreviations by replacing periods with placeholders
        protected = text
        placeholder_map: dict[str, str] = {}
        for i, abbr in enumerate(self._ABBREVIATIONS):
            placeholder = f"\x00ABBR{i}\x00"
            placeholder_map[placeholder] = abbr
            protected = protected.replace(abbr, placeholder)

        # Step 2: split on sentence-ending punctuation
        #   Chinese 。！？ — split right after (zero-width lookbehind, keep the punct)
        #   English .!? + whitespace + uppercase/numeral/Chinese char
        parts = re.split(
            r"(?<=[。！？])"
            r"|(?<=[.!?])\s+(?=[A-Z0-9\u4e00-\u9fff])",
            protected,
        )

        # Step 3: restore abbreviations
        sentences = []
        for part in parts:
            for placeholder, abbr in placeholder_map.items():
                part = part.replace(placeholder, abbr)
            part = part.strip()
            if part:
                sentences.append(part)

        return sentences

    def _is_section_header(self, text: str) -> bool:
        """Check whether a text span is an academic paper section/subsection header.

        Section headers are typically short (< 100 chars) and match known
        numbering or naming patterns.  Detecting them lets us force chunk
        boundaries at natural semantic transitions.
        """
        if not text:
            return False
        stripped = text.strip()
        if not stripped or len(stripped) > 100:
            return False
        return bool(self._SECTION_PAT.match(stripped))

    def _chunk_text(
        self,
        text: str,
        max_chars: int = 4000,
        overlap_sentences: int = 0,
    ) -> list[str]:
        """Semantic chunking: sentence-level split with section-boundary awareness.

        1. Split into individual sentences (。.!? boundaries).
        2. Group sentences into chunks ≤ max_chars, never splitting mid-sentence.
        3. Section/subsection headers force a fresh chunk (natural semantic boundary).
        4. Optional overlap: carry the last N sentences into the next chunk for
           context continuity when chunks are split due to max_chars within a section.

        Args:
            text: Full text to split.
            max_chars: Maximum characters per chunk.
            overlap_sentences: Number of sentences to overlap between adjacent chunks
                               (0 = no overlap, recommended for academic papers).

        Returns:
            List of chunk strings.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return [text] if text else []

        if len(sentences) <= 1 and len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        current_buf: list[str] = []
        current_len = 0

        def _flush(overlap: int = 0) -> None:
            """Commit current buffer as a chunk, optionally retaining tail for overlap."""
            nonlocal current_buf, current_len
            if not current_buf:
                return
            chunks.append(" ".join(current_buf))
            if overlap > 0 and len(current_buf) > overlap:
                current_buf = current_buf[-overlap:]
                current_len = sum(len(s) + 1 for s in current_buf) - 1
            else:
                current_buf = []
                current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            is_boundary = self._is_section_header(sent)

            # Section boundary: flush current chunk with NO overlap
            # (different semantic topic — overlap would be misleading)
            if is_boundary and current_buf:
                _flush(overlap=0)

            # Max-chars exceeded within same topic: flush with optional overlap
            if current_len + sent_len + (1 if current_buf else 0) > max_chars and current_buf:
                _flush(overlap=overlap_sentences)

            # Handle the rare case where a single sentence exceeds max_chars
            # (e.g., a giant paragraph with no sentence boundary).  Fall back
            # to hard-cut at max_chars so downstream code still gets chunks.
            if sent_len > max_chars and not current_buf:
                for i in range(0, sent_len, max_chars):
                    chunks.append(sent[i : i + max_chars])
                continue

            current_buf.append(sent)
            current_len += sent_len + (1 if len(current_buf) > 1 else 0)

        # Flush remaining
        _flush(overlap=0)

        # B1: sentence splitting may have cut a $$ formula block (formula
        # bodies contain ". "-like patterns, e.g. "2. 75 H2O").  Re-attach
        # dangling half-blocks so every chunk holds complete formulas only.
        chunks = KnowledgeExtractionAgent._repair_math_spans(chunks)

        return chunks if chunks else [text]

    def _collect_quotes(self, result: dict) -> list[str]:
        """Collect raw_quotes from all numeric fields."""
        quotes = []
        for key in ["ionic_conductivity", "electrochemical_window",
                     "sintering_temperature", "test_temperature", "pressure"]:
            field = result.get(key)
            if field and isinstance(field, dict) and field.get("raw_quote"):
                quotes.append(field["raw_quote"])
        return quotes
