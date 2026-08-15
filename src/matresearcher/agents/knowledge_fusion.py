"""Knowledge Fusion Agent (Steps 8-11).

Responsibilities:
- Step 8: Unit normalization & chemical formula normalization
- Step 9: Store to vector DB + relational DB
- Step 10: Cross-literature knowledge fusion (per-material aggregation)
- Step 11: Conflict detection & missing connection detection
"""
from __future__ import annotations

from typing import Optional

from ..state import WorkflowState
from .base import BaseAgent
from ..models.knowledge import (
    KnowledgeRecord, NormalizedRecord, MaterialSummary, FusedKnowledgeTable
)
from ..rules.unit_conversion import UnitConverter
from ..rules.formula_normalizer import FormulaNormalizer
from ..rules.conflict_detector import ConflictDetector


class KnowledgeFusionAgent(BaseAgent):
    name = "knowledge_fusion"
    role = "知识融合 Agent"

    def __init__(self, llm=None, config=None,
                 vector_store=None, relational_store=None,
                 embedding_model=None, log_dir=None):
        super().__init__(llm, config, log_dir=log_dir)
        self.unit_converter = UnitConverter()
        self.conflict_detector = ConflictDetector(
            conflict_threshold=self.config.get("conflict_threshold", 0.5),
        )
        self.vector_store = vector_store
        self.relational_store = relational_store
        self.embedding_model = embedding_model

    async def run(self, state: WorkflowState) -> dict:
        filtered = state.get("filtered_literature", [])
        anomaly_records = state.get("anomaly_records", [])

        # Step 7a verdicts: lit.knowledge_records only carries PASS records,
        # so REVIEW/FAIL records are structurally excluded from fusion.
        # Surface the breakdown here so the separation is visible in the
        # fusion log, with a pointer to quality_report.json for adjudication.
        if anomaly_records:
            n_review = sum(
                1 for r in anomaly_records
                if getattr(r, "quality_status", "") == "review"
            )
            n_fail = len(anomaly_records) - n_review
            self.log(
                f"Quality gate: {n_review} REVIEW + {n_fail} FAIL records "
                f"excluded from fusion (details: quality_report.json)"
            )

        # Collect all verified knowledge records from parsed literature
        all_records: list[KnowledgeRecord] = []
        for lit in filtered:
            if lit.knowledge_records:
                all_records.extend(lit.knowledge_records)

        if not all_records:
            self.log("No knowledge records to fuse", "yellow")
            return {"normalized_records": [], "fused_table": FusedKnowledgeTable(),
                    "conflicts": [], "missing_items": []}

        self.log(f"Fusing {len(all_records)} knowledge records from {len(filtered)} papers")
        normalized = self._normalize_records(all_records)

        # Step 9: Store to databases
        self.log_step("9", "存储到向量库 + 关系库")
        self._store_records(normalized)
        self.log_step("9", f"{len(normalized)} 条记录已存储", "done")

        # Step 10: Cross-literature fusion
        self.log_step("10", "跨文献知识融合")
        fused_table = self._fuse_knowledge(normalized)
        self.log_step("10", f"{fused_table.total_materials} 种材料已融合", "done")

        # Step 11: Conflict & missing detection
        self.log_step("11", "冲突检测与缺失发现")
        conflicts = self.conflict_detector.detect_conflicts(normalized)
        missing_items = self.conflict_detector.detect_missing(normalized)
        self.log_step("11", f"{len(conflicts)} 条冲突, {len(missing_items)} 条缺失", "done")

        # Add results to fused table
        fused_table.conflict_list = conflicts
        fused_table.missing_list = missing_items

        self.log(f"Fusion complete: {fused_table.total_materials} materials, "
                 f"{len(conflicts)} conflicts, {len(missing_items)} missing")

        return {
            "normalized_records": normalized,
            "fused_table": fused_table,
            "conflicts": conflicts,
            "missing_items": missing_items,
        }

    def _normalize_records(self, records: list[KnowledgeRecord]) -> list[NormalizedRecord]:
        """Step 8: Normalize all records.

        - Convert units to standard (S/cm, K, MPa)
        - Normalize chemical formulas
        """
        normalized = []
        for record in records:
            # Unit conversion
            norm = self.unit_converter.normalize_record(record)

            # Chemical formula normalization
            if record.material_composition:
                formula, alias = FormulaNormalizer.normalize(record.material_composition)
                norm.material_composition = formula
                norm.material_alias = alias

            normalized.append(norm)
        return normalized

    # ── field name mapping: NormalizedRecord → KnowledgeRecordORM ──────

    _NORM_TO_ORM_MAP = {
        # NormalizedRecord field  →  KnowledgeRecordORM column
        "ionic_conductivity_S_cm": "ionic_conductivity",
        "ionic_conductivity_temp_K": "ionic_cond_temp",
        "electrochemical_window_V": "electrochemical_window",
        "sintering_temperature_K": "sintering_temperature",
        "test_temperature_K": "test_temperature",
        "pressure_MPa": "pressure",
    }

    # ORM columns (keys that KnowledgeRecordORM.__init__ accepts)
    _ORM_FIELDS = frozenset({
        "id", "literature_id", "material_composition", "material_alias",
        "crystal_structure", "ionic_conductivity", "ionic_cond_temp",
        "electrochemical_window", "synthesis_method", "sintering_temperature",
        "test_temperature", "pressure", "simulation_method",
        "key_findings", "raw_quotes", "quality_status",
    })

    @classmethod
    def _to_orm_dict(cls, record: NormalizedRecord) -> dict:
        """Convert a NormalizedRecord to a dict accepted by KnowledgeRecordORM."""
        raw = record.model_dump()
        mapped = {}
        for k, v in raw.items():
            if k in cls._ORM_FIELDS:
                mapped[k] = v
            elif k in cls._NORM_TO_ORM_MAP:
                mapped[cls._NORM_TO_ORM_MAP[k]] = v
        return mapped

    def _store_records(self, records: list[NormalizedRecord]):
        """Step 9: Store normalized records to vector DB and relational DB."""
        collection = self.config.get("vector_collection", "knowledge")

        # Store to vector DB for semantic retrieval
        if self.vector_store and self.embedding_model:
            texts = []
            valid_records = []
            for record in records:
                text = self._record_to_text(record)
                if text:
                    texts.append(text)
                    valid_records.append(record)

            if texts:
                try:
                    # Generate embeddings in batch
                    embeddings = self.embedding_model.embed_batch(texts)
                    emb_list = [e.tolist() for e in embeddings]

                    self.vector_store.add(
                        collection=collection,
                        ids=[r.id for r in valid_records],
                        documents=texts,
                        embeddings=emb_list,
                        metadatas=[{
                            "literature_id": r.literature_id,
                            "material": r.material_composition or "",
                            "alias": r.material_alias or "",
                            "quality": r.quality_status,
                        } for r in valid_records],
                    )
                    self.log(f"Stored {len(valid_records)} records to vector DB")
                except Exception as e:
                    self.log(f"Vector store error: {e}", "yellow")

        # Store to relational DB
        if self.relational_store:
            try:
                self.relational_store.add_knowledge_batch([
                    self._to_orm_dict(record) for record in records
                ])
                self.log(f"Stored {len(records)} records to relational DB (batch)")
            except Exception as e:
                self.log(f"Relational store error: {e}", "yellow")

    def _fuse_knowledge(self, records: list[NormalizedRecord]) -> FusedKnowledgeTable:
        """Step 10: Cross-literature knowledge fusion.

        Groups records by material and computes aggregated statistics.
        """
        # Group by material
        material_groups: dict[str, list[NormalizedRecord]] = {}
        for record in records:
            key = record.material_composition or "unknown"
            if key not in material_groups:
                material_groups[key] = []
            material_groups[key].append(record)

        summaries = []
        for material, mat_records in material_groups.items():
            # Extract conductivities
            conductivities = [
                r.ionic_conductivity_S_cm for r in mat_records
                if r.ionic_conductivity_S_cm is not None
            ]

            summary = MaterialSummary(
                material=material,
                alias=mat_records[0].material_alias if mat_records else None,
                records=mat_records,
                n_papers=len(set(r.literature_id for r in mat_records)),
                conductivity_range=(
                    (min(conductivities), max(conductivities))
                    if len(conductivities) >= 2 else None
                ),
                mean_conductivity=(
                    sum(conductivities) / len(conductivities)
                    if conductivities else None
                ),
            )
            summaries.append(summary)

        return FusedKnowledgeTable(
            summaries=summaries,
            total_records=len(records),
            total_materials=len(summaries),
        )

    @staticmethod
    def _record_to_text(record: NormalizedRecord) -> str:
        """Convert a normalized record to searchable text."""
        parts = []
        if record.material_composition:
            parts.append(f"Material: {record.material_composition}")
            if record.material_alias:
                parts.append(f"({record.material_alias})")
        if record.ionic_conductivity_S_cm is not None:
            parts.append(f"Conductivity: {record.ionic_conductivity_S_cm:.4e} S/cm")
        if record.ionic_conductivity_temp_K is not None:
            parts.append(f"at {record.ionic_conductivity_temp_K:.0f}K")
        if record.synthesis_method:
            parts.append(f"Synthesis: {record.synthesis_method}")
        if record.key_findings:
            parts.append(f"Findings: {record.key_findings}")
        return " ".join(parts)
