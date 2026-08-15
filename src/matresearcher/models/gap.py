"""Research Gap data models."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GapSupportingRef(BaseModel):
    """A literature reference supporting a Research Gap.

    LLM 只输出 `title` + `finding`（+ 可选 `doi`）。`doc_id` / `year` /
    `journal` / `authors` 由工程层 `_enrich_gap_dois` 按 `lit.id` 回填，
    LLM 永不直接输出 doc_id（防止编造内部 ID）。
    """
    doi: Optional[str] = None
    title: str
    finding: str  # what this literature found that is relevant

    # 证据溯源锚点（工程层回填，DOI 缺失时的备选主键）
    doc_id: Optional[str] = None             # Sciverse 内部文档 ID（A-0 已验证跨会话稳定）
    year: Optional[int] = None
    journal: Optional[str] = None
    authors: list[str] = Field(default_factory=list)

    # Step 13 verification result — kept separate from `finding` so the
    # original claim text is never polluted with raw passage fragments.
    verified_passage: Optional[str] = None   # 核验命中的原文段落（独立字段）
    verification_score: Optional[float] = None  # 命中置信度

    @property
    def has_anchor(self) -> bool:
        """是否有可核验的溯源锚点（doc_id 或 DOI 有其一即可）。"""
        return bool(self.doc_id or self.doi)


class ResearchGap(BaseModel):
    """A structured Research Gap (Step 12 output).

    Each Gap contains exactly 7 elements per the proposal:
    1. 问题描述
    2. 支撑文献
    3. 证据缺失或冲突
    4. 新颖性
    5. 可操作性
    6. 可证伪假设
    7. 建议验证方法
    """
    gap_id: str
    description: str                         # 1. 问题描述
    supporting_literature: list[GapSupportingRef] = Field(default_factory=list)  # 2. 支撑文献
    evidence_gap_or_conflict: str            # 3. 证据缺失或冲突
    novelty: str                             # 4. 新颖性（新知vs已知）
    operability: str                         # 5. 可操作性
    falsifiable_hypothesis: str              # 6. 可证伪假设
    suggested_verification: str              # 7. 建议验证方法

    # Step 13: evidence verification
    verification_status: str = "pending"     # pending | passed | partial | failed | unverified
    verification_notes: Optional[str] = None
    correction_suggestions: list[str] = Field(default_factory=list)

    # Step 14: scoring
    score: Optional["GapScore"] = None


class GapScore(BaseModel):
    """Scoring for a Research Gap (Step 14 output)."""
    novelty_score: float = 0.0           # 0-1, higher = more novel
    operability_score: float = 0.0       # 0-1
    evidence_completeness: float = 0.0   # 0-1
    total_score: float = 0.0
    rank: int = 0

    def compute_total(self, weights: dict[str, float] | None = None) -> float:
        w = weights or {"novelty": 0.4, "operability": 0.3, "evidence_completeness": 0.3}
        self.total_score = (
            self.novelty_score * w.get("novelty", 0.4)
            + self.operability_score * w.get("operability", 0.3)
            + self.evidence_completeness * w.get("evidence_completeness", 0.3)
        )
        return self.total_score


class ConflictItem(BaseModel):
    """A detected conflict between literature records (Step 11 output)."""
    material: str
    field: str  # e.g. "ionic_conductivity"
    values: list[dict] = Field(default_factory=list)  # [{"value": ..., "doi": ..., "method": ...}]
    conflict_ratio: float = 0.0
    possible_causes: list[str] = Field(default_factory=list)


class MissingItem(BaseModel):
    """A detected missing knowledge connection (Step 11 output)."""
    material: str
    missing_field: str      # e.g. "conductivity_at_low_temperature"
    description: str
    existing_data_points: int = 0
