"""Conflict detection rules for cross-literature knowledge fusion.

Detects:
1. Value conflicts: same material, same property, significantly different values
2. Missing connections: data gaps for specific material/property/condition combinations
"""
from __future__ import annotations

from typing import Optional

from ..models.knowledge import NormalizedRecord, MaterialSummary, FusedKnowledgeTable
from .formula_normalizer import FormulaNormalizer


class ConflictDetector:
    """Detect conflicts and missing connections across literature records."""

    def __init__(self, conflict_threshold: float = 0.5):
        """
        Args:
            conflict_threshold: If values differ by more than this fraction
                (relative to the mean), they are flagged as conflict.
                E.g. 0.5 means values must differ by >50% of the mean.
        """
        self.conflict_threshold = conflict_threshold

    def detect_conflicts(self, records: list[NormalizedRecord]) -> list[dict]:
        """Detect value conflicts within a set of normalized records.

        Groups records by material, then checks for conflicting property values.
        Prefer detect_conflicts_from_fused() when FusedKnowledgeTable is available.
        """
        return self._detect_from_groups(self._group_by_material(records))

    def detect_conflicts_from_fused(self, fused_table: FusedKnowledgeTable) -> list[dict]:
        """Detect conflicts directly from a FusedKnowledgeTable, reusing pre-grouped summaries."""
        if not fused_table or not fused_table.summaries:
            return []
        groups = {}
        for summary in fused_table.summaries:
            groups[summary.material] = summary.records
        return self._detect_from_groups(groups)

    def detect_missing(self, records: list[NormalizedRecord]) -> list[dict]:
        """Detect missing data connections (knowledge gaps)."""
        return self._detect_missing_from_groups(self._group_by_material(records))

    def detect_missing_from_fused(self, fused_table: FusedKnowledgeTable) -> list[dict]:
        """Detect missing data from a FusedKnowledgeTable, reusing pre-grouped summaries."""
        if not fused_table or not fused_table.summaries:
            return []
        groups = {}
        for summary in fused_table.summaries:
            groups[summary.material] = summary.records
        return self._detect_missing_from_groups(groups)

    # ── Reusable core logic (works with pre-grouped records) ──

    def _detect_from_groups(self, material_groups: dict[str, list[NormalizedRecord]]) -> list[dict]:
        """Core conflict detection from pre-grouped records."""
        conflicts = []

        for material, mat_records in material_groups.items():
            # Check ionic conductivity conflicts
            cond_values = [
                {
                    "value": r.ionic_conductivity_S_cm,
                    "temp_K": r.ionic_conductivity_temp_K,
                    "doi": r.doi or r.literature_id,  # real DOI first, fallback to lit_id
                    "literature_id": r.literature_id,  # keep lit_id for gap enrichment lookup
                    "synthesis": r.synthesis_method,
                }
                for r in mat_records
                if r.ionic_conductivity_S_cm is not None
            ]
            if len(cond_values) >= 2:
                conflict = self._check_value_conflict(material, "ionic_conductivity", cond_values)
                if conflict:
                    conflicts.append(conflict)

            # Check electrochemical window conflicts
            ecw_values = [
                {
                    "value": r.electrochemical_window_V,
                    "doi": r.doi or r.literature_id,  # real DOI first, fallback to lit_id
                    "literature_id": r.literature_id,  # keep lit_id for gap enrichment lookup
                    "synthesis": r.synthesis_method,
                }
                for r in mat_records
                if r.electrochemical_window_V is not None
            ]
            if len(ecw_values) >= 2:
                conflict = self._check_value_conflict(material, "electrochemical_window", ecw_values)
                if conflict:
                    conflicts.append(conflict)

        return conflicts

    def _detect_missing_from_groups(self, material_groups: dict[str, list[NormalizedRecord]]) -> list[dict]:
        """Core missing-data detection from pre-grouped records."""
        missing = []

        for material, mat_records in material_groups.items():
            # Check: is there conductivity data at low temperature (< 300 K)?
            low_temp_records = [
                r for r in mat_records
                if r.ionic_conductivity_S_cm is not None
                and r.ionic_conductivity_temp_K is not None
                and r.ionic_conductivity_temp_K < 300
            ]
            if len(mat_records) >= 3 and len(low_temp_records) == 0:
                missing.append({
                    "material": material,
                    "missing_field": "conductivity_at_low_temperature",
                    "description": f"Material {material} has {len(mat_records)} records "
                                   f"but no conductivity data below 300 K (room temperature).",
                    "existing_data_points": len(mat_records),
                })

            # Check: is there pressure-dependent data?
            pressure_records = [r for r in mat_records if r.pressure_MPa is not None]
            if len(mat_records) >= 3 and len(pressure_records) == 0:
                missing.append({
                    "material": material,
                    "missing_field": "pressure_dependent_conductivity",
                    "description": f"Material {material} has {len(mat_records)} records "
                                   f"but no pressure-dependent conductivity data.",
                    "existing_data_points": len(mat_records),
                })

            # Check: is there simulation (DFT/MD) data?
            sim_records = [r for r in mat_records if r.simulation_method and r.simulation_method != "无"]
            if len(mat_records) >= 3 and len(sim_records) == 0:
                missing.append({
                    "material": material,
                    "missing_field": "simulation_data",
                    "description": f"Material {material} has {len(mat_records)} records "
                                   f"but no computational simulation (DFT/MD) data.",
                    "existing_data_points": len(mat_records),
                })

        return missing

    def _group_by_material(self, records: list[NormalizedRecord]) -> dict[str, list[NormalizedRecord]]:
        """Group records by normalized material composition."""
        groups: dict[str, list[NormalizedRecord]] = {}
        for r in records:
            if not r.material_composition:
                continue
            normalized, _ = FormulaNormalizer.normalize(r.material_composition)
            if not normalized:
                continue
            if normalized not in groups:
                groups[normalized] = []
            groups[normalized].append(r)
        return groups

    def _check_value_conflict(self, material: str, field: str, values: list[dict]) -> dict | None:
        """Check if values conflict (differ by > threshold)."""
        numeric_values = [v["value"] for v in values if v["value"] is not None]
        if len(numeric_values) < 2:
            return None

        mean_val = sum(numeric_values) / len(numeric_values)
        if mean_val == 0:
            return None

        max_val = max(numeric_values)
        min_val = min(numeric_values)
        range_ratio = (max_val - min_val) / abs(mean_val)

        if range_ratio > self.conflict_threshold:
            possible_causes = []
            # Check if different synthesis methods
            methods = set(v.get("synthesis") for v in values if v.get("synthesis"))
            if len(methods) > 1:
                possible_causes.append("不同合成方法导致性能差异")
            # Check if different test temperatures
            temps = [v.get("temp_K") for v in values if v.get("temp_K")]
            if temps and max(temps) - min(temps) > 50:
                possible_causes.append("不同测试温度导致性能差异")
            if not possible_causes:
                possible_causes.append("原因待查，可能涉及制备工艺或表征方法差异")

            return {
                "material": material,
                "field": field,
                "values": values,
                "conflict_ratio": round(range_ratio, 2),
                "possible_causes": possible_causes,
            }
        return None
