"""Unit conversion rules for materials science data.

Handles: ionic conductivity, temperature, pressure, etc.
Standard units:
  - ionic_conductivity -> S/cm
  - temperature        -> K
  - pressure           -> MPa
"""
from __future__ import annotations

import re
from typing import Optional

from ..models.knowledge import KnowledgeRecord, NormalizedRecord


class UnitConverter:
    """Rule-based unit conversion for material property values."""

    # Conversion factors: value_in_standard = value * factor[from_unit]
    CONDUCTIVITY_FACTORS = {
        "S/cm": 1.0,
        "s/cm": 1.0,
        "mS/cm": 1e-3,
        "ms/cm": 1e-3,
        "uS/cm": 1e-6,
        "us/cm": 1e-6,
        "S/m": 1e-2,     # S/m -> S/cm
        "s/m": 1e-2,
    }

    TEMPERATURE_OFFSETS = {
        # K = (C + 273.15),  K = (F - 32) * 5/9 + 273.15
        "K": lambda v: v,
        "k": lambda v: v,
        "C": lambda v: v + 273.15,
        "c": lambda v: v + 273.15,
        "°C": lambda v: v + 273.15,
        "℃": lambda v: v + 273.15,
        "F": lambda v: (v - 32) * 5 / 9 + 273.15,
        "f": lambda v: (v - 32) * 5 / 9 + 273.15,
    }

    PRESSURE_FACTORS = {
        "MPa": 1.0,
        "mpa": 1.0,
        "GPa": 1e3,
        "gpa": 1e3,
        "kPa": 1e-3,
        "kpa": 1e-3,
        "Pa": 1e-6,
        "pa": 1e-6,
        "atm": 0.101325,
    }

    def convert_conductivity(self, value: float, unit: str) -> tuple[float, str]:
        """Convert ionic conductivity to S/cm."""
        factor = self.CONDUCTIVITY_FACTORS.get(unit.strip())
        if factor is None:
            # Try to parse compound units like "1.0×10⁻³ S/cm"
            return value, unit  # return as-is if unknown
        return value * factor, "S/cm"

    def convert_temperature(self, value: float, unit: str) -> tuple[float, str]:
        """Convert temperature to Kelvin."""
        converter = self.TEMPERATURE_OFFSETS.get(unit.strip())
        if converter is None:
            return value, unit
        return converter(value), "K"

    def convert_pressure(self, value: float, unit: str) -> tuple[float, str]:
        """Convert pressure to MPa."""
        factor = self.PRESSURE_FACTORS.get(unit.strip())
        if factor is None:
            return value, unit
        return value * factor, "MPa"

    def normalize_record(self, record: KnowledgeRecord) -> NormalizedRecord:
        """Convert a KnowledgeRecord to a NormalizedRecord with standard units."""
        # Conductivity
        cond_s_cm = None
        cond_temp_k = None
        if record.ionic_conductivity and record.ionic_conductivity.value is not None:
            cond_val, _ = self.convert_conductivity(
                record.ionic_conductivity.value,
                record.ionic_conductivity.unit or "S/cm",
            )
            cond_s_cm = cond_val
            if record.ionic_conductivity.temperature_K:
                cond_temp_k = record.ionic_conductivity.temperature_K
            elif record.test_temperature and record.test_temperature.value is not None:
                temp_k, _ = self.convert_temperature(
                    record.test_temperature.value,
                    record.test_temperature.unit or "K",
                )
                cond_temp_k = temp_k

        # Electrochemical window (already in V)
        ecw_v = None
        if record.electrochemical_window and record.electrochemical_window.value is not None:
            ecw_v = record.electrochemical_window.value

        # Sintering temperature
        sinter_k = None
        if record.sintering_temperature and record.sintering_temperature.value is not None:
            sinter_k, _ = self.convert_temperature(
                record.sintering_temperature.value,
                record.sintering_temperature.unit or "K",
            )

        # Test temperature
        test_k = None
        if record.test_temperature and record.test_temperature.value is not None:
            test_k, _ = self.convert_temperature(
                record.test_temperature.value,
                record.test_temperature.unit or "K",
            )

        # Pressure
        press_mpa = None
        if record.pressure and record.pressure.value is not None:
            press_mpa, _ = self.convert_pressure(
                record.pressure.value,
                record.pressure.unit or "MPa",
            )

        return NormalizedRecord(
            id=record.id,
            literature_id=record.literature_id,
            doi=record.doi,
            material_composition=record.material_composition,
            crystal_structure=record.crystal_structure,
            ionic_conductivity_S_cm=cond_s_cm,
            ionic_conductivity_temp_K=cond_temp_k,
            electrochemical_window_V=ecw_v,
            synthesis_method=record.synthesis_method,
            sintering_temperature_K=sinter_k,
            test_temperature_K=test_k,
            pressure_MPa=press_mpa,
            simulation_method=record.simulation_method,
            key_findings=record.key_findings,
            raw_quotes=record.raw_quotes,
            quality_status=record.quality_status,
        )
