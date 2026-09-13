"""Unit tests for the deterministic rule layer.

These modules are pure functions/classes — no LLM, no network — so they are the
cheapest place to get real regression coverage. (Before this, the only automated
checks were the 39 zero-token smoke cases in scripts/smoke_quality_gate.py;
none of the 10 agents had tests.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from matresearcher.rules.unit_conversion import UnitConverter  # noqa: E402
from matresearcher.rules.formula_normalizer import FormulaNormalizer  # noqa: E402
from matresearcher.rules.conflict_detector import ConflictDetector  # noqa: E402
from matresearcher.models.knowledge import NormalizedRecord  # noqa: E402


# ── Unit conversion ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "value,unit,expected_value",
    [
        (1.0, "S/cm", 1.0),
        (1.0, "mS/cm", 1e-3),      # the classic S/cm <-> mS/cm confusion
        (1.0, "uS/cm", 1e-6),
        (1.0, "S/m", 1e-2),
    ],
)
def test_conductivity_normalised_to_S_per_cm(value, unit, expected_value):
    got_value, got_unit = UnitConverter().convert_conductivity(value, unit)
    assert got_unit == "S/cm"
    assert got_value == pytest.approx(expected_value)


@pytest.mark.parametrize(
    "value,unit,expected",
    [(25.0, "C", 298.15), (25.0, "℃", 298.15), (298.15, "K", 298.15)],
)
def test_temperature_normalised_to_kelvin(value, unit, expected):
    got_value, got_unit = UnitConverter().convert_temperature(value, unit)
    assert got_unit == "K"
    assert got_value == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    "value,unit,expected",
    [(1.0, "GPa", 1000.0), (500.0, "MPa", 500.0), (1.0, "kPa", 1e-3)],
)
def test_pressure_normalised_to_mpa(value, unit, expected):
    got_value, got_unit = UnitConverter().convert_pressure(value, unit)
    assert got_unit == "MPa"
    assert got_value == pytest.approx(expected)


def test_unknown_unit_is_passed_through_not_silently_rescaled():
    # Guards the "only warn, never silently fix" policy of rule R6.
    value, unit = UnitConverter().convert_conductivity(3.0, "furlongs/fortnight")
    assert (value, unit) == (3.0, "furlongs/fortnight")


# ── Formula normalisation ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw", ["Li7La3Zr2O12", "Li₇La₃Zr₂O₁₂", "Li7 La3 Zr2 O12"]
)
def test_llzo_variants_collapse_to_one_form(raw):
    normalized, _alias = FormulaNormalizer.normalize(raw)
    assert normalized == "Li7La3Zr2O12"
    assert FormulaNormalizer.are_same_material(raw, "Li7La3Zr2O12")


def test_alias_in_parentheses_is_recognised():
    _normalized, alias = FormulaNormalizer.normalize("Li7La3Zr2O12 (LLZO)")
    assert alias == "LLZO"


def test_doped_garnet_is_not_the_same_material_as_undoped():
    assert not FormulaNormalizer.are_same_material(
        "Li6.25Ga0.25La3Zr2O12", "Li7La3Zr2O12"
    )


def test_garbage_input_does_not_raise():
    assert FormulaNormalizer.normalize(None) == (None, None)
    assert FormulaNormalizer.normalize("") == (None, None)
    assert FormulaNormalizer.are_same_material(None, None) is False
    # Unknown garbage must not crash the pipeline either:
    assert FormulaNormalizer.normalize("!!!")[0] == "!!!"


# ── Conflict detection ───────────────────────────────────────────────────────

def _record(material: str, value: float, doi: str = "10.1000/x",
            temp_k: float = 298.15) -> NormalizedRecord:
    return NormalizedRecord(
        id=f"{material}-{value}-{doi}",
        literature_id="lit-1",
        material_composition=material,
        ionic_conductivity_S_cm=value,
        ionic_conductivity_temp_K=temp_k,
        doi=doi,
    )


def test_two_orders_of_magnitude_gap_is_a_conflict():
    # Mirrors the real LLZO spread reported in the proposal (1.9e-5 ~ 1.49e-3).
    recs = [_record("Li7La3Zr2O12", 1.9e-5, "doi-a"),
            _record("Li7La3Zr2O12", 1.49e-3, "doi-b")]
    conflicts = ConflictDetector(conflict_threshold=0.5).detect_conflicts(recs)
    assert conflicts, "a ~78x spread on one material must be flagged as a conflict"


def test_near_identical_values_are_not_a_conflict():
    recs = [_record("Li7La3Zr2O12", 1.0e-3, "doi-a"),
            _record("Li7La3Zr2O12", 1.02e-3, "doi-b")]
    assert ConflictDetector(conflict_threshold=0.5).detect_conflicts(recs) == []


def test_different_materials_are_never_compared():
    recs = [_record("Li7La3Zr2O12", 1.0e-3), _record("Li3PS4", 1.0e-5)]
    assert ConflictDetector(conflict_threshold=0.5).detect_conflicts(recs) == []


def test_missing_field_detection_needs_three_records():
    """detect_missing only reports once a material has >= 3 records."""
    one = ConflictDetector().detect_missing([_record("Li7La3Zr2O12", 1.0e-3)])
    assert one == []

    # 3 records, all measured at/above 300 K, none with pressure or simulation
    # data → low-temperature, pressure-dependent and simulation gaps are raised.
    recs = [_record("Li7La3Zr2O12", v, f"doi-{i}", temp_k=350.0) for i, v in
            enumerate([1.0e-3, 1.1e-3, 1.2e-3])]
    missing = ConflictDetector().detect_missing(recs)
    fields = {m["missing_field"] for m in missing}
    assert "conductivity_at_low_temperature" in fields
    assert "pressure_dependent_conductivity" in fields
    assert "simulation_data" in fields
