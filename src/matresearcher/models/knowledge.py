"""Knowledge record data models."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class NumericValue(BaseModel):
    """A numeric value with unit and context."""
    value: float | None = None
    unit: str = ""
    original_value: Optional[str] = None  # raw text from paper
    original_unit: Optional[str] = None
    temperature_K: Optional[float] = None  # measurement temperature
    raw_quote: Optional[str] = None  # supporting quote from original text


class KnowledgeRecord(BaseModel):
    """Raw knowledge extracted from a single paper (Step 7 output)."""
    id: str
    literature_id: str
    doi: Optional[str] = None  # real DOI for evidence verification
    material_composition: Optional[str] = None  # e.g. "Li7La3Zr2O12"
    crystal_structure: Optional[str] = None      # e.g. "garnet, Ia-3d"
    ionic_conductivity: Optional[NumericValue] = None
    electrochemical_window: Optional[NumericValue] = None
    synthesis_method: Optional[str] = None
    sintering_temperature: Optional[NumericValue] = None
    test_temperature: Optional[NumericValue] = None
    pressure: Optional[NumericValue] = None
    simulation_method: Optional[str] = None     # DFT | MD | Monte Carlo | None
    key_findings: Optional[str] = None
    raw_quotes: list[str] = Field(default_factory=list)
    # Step 7a: quality verification status (three-way gate)
    # - verified: PASS — all hard & soft rules clean
    # - review:   REVIEW — plausible unit confusion / family outlier, kept for manual check
    # - anomaly:  FAIL — hard rule violation, excluded from fusion
    quality_status: str = "pending"  # pending | verified | review | anomaly
    quality_issues: list[str] = Field(default_factory=list)


class NormalizedRecord(BaseModel):
    """A knowledge record after entity normalization & unit conversion (Step 8 output)."""
    id: str
    literature_id: str
    doi: Optional[str] = None  # real DOI carried from KnowledgeRecord
    material_composition: Optional[str] = None    # normalized formula
    material_alias: Optional[str] = None          # e.g. "LLZO"
    crystal_structure: Optional[str] = None
    ionic_conductivity_S_cm: Optional[float] = None      # normalized to S/cm
    ionic_conductivity_temp_K: Optional[float] = None
    electrochemical_window_V: Optional[float] = None
    synthesis_method: Optional[str] = None
    sintering_temperature_K: Optional[float] = None       # normalized to K
    test_temperature_K: Optional[float] = None
    pressure_MPa: Optional[float] = None                 # normalized to MPa
    simulation_method: Optional[str] = None
    key_findings: Optional[str] = None
    raw_quotes: list[str] = Field(default_factory=list)
    quality_status: str = "verified"  # inherited from KnowledgeRecord


class MaterialSummary(BaseModel):
    """Aggregated summary for one material system (Step 10 output)."""
    material: str
    alias: Optional[str] = None
    records: list[NormalizedRecord] = Field(default_factory=list)
    conductivity_range: Optional[tuple[float, float]] = None  # (min, max) in S/cm
    mean_conductivity: Optional[float] = None
    n_papers: int = 0
    conflicts: list[dict] = Field(default_factory=list)
    data_gaps: list[dict] = Field(default_factory=list)


class FusedKnowledgeTable(BaseModel):
    """Cross-literature fusion result (Step 10-11 output)."""
    summaries: list[MaterialSummary] = Field(default_factory=list)
    conflict_list: list[dict] = Field(default_factory=list)
    missing_list: list[dict] = Field(default_factory=list)
    total_records: int = 0
    total_materials: int = 0
