"""Loader for the domain configuration (config/materials.yaml).

The rule engine and the structure-property analysis read every domain-specific
constant from here instead of hardcoding it, so retargeting the pipeline to
another material domain is a config edit rather than a code change.

Loading never raises: a missing or malformed file falls back to the built-in
defaults so the pipeline keeps running (a config error must not be able to
take down a run).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "materials.yaml"
)

FALLBACK: dict[str, Any] = {
    "domain": {
        "name": "solid-state-battery-electrolytes",
        "property_under_study": "ionic_conductivity",
        "standard_unit": "S/cm",
    },
    "family_keywords": {
        "polymer": ["peo", "pvdf", "ptmc", "polymer", "poly(", "poly-"],
        "borohydride": ["libh4", "bh4", "borohydride"],
        "sulfide": ["lgps", "argyrodite", "li6ps5cl", "lpsc", "li7p3s11",
                    "thio", "sulfide", "sulphide"],
        "halide": ["chloride", "bromide", "iodide", "fluoride", "halide",
                   "li3incl6", "li3ycl6"],
        "oxide": ["llzo", "llzto", "garnet", "nasicon", "lisicon", "latp",
                  "lagp", "llto", "lipon", "perovskite", "oxide"],
    },
    "conductivity_windows": {
        "oxide": {"plausible_range": [1e-9, 1e-1], "warn_above": 1e-2,
                  "label": "oxide (garnet/NASICON/perovskite)"},
        "sulfide": {"plausible_range": [1e-8, 1e-1], "warn_above": 5e-2,
                    "label": "sulfide (argyrodite/LGPS/thio-LISICON)"},
        "halide": {"plausible_range": [1e-9, 1e-2], "warn_above": 5e-3,
                   "label": "halide (Li3InCl6/Li3YCl6 family)"},
        "polymer": {"plausible_range": [1e-10, 1e-2], "warn_above": 1e-2,
                    "label": "polymer (PEO/PVDF-based)"},
        "borohydride": {"plausible_range": [1e-9, 1e-2], "warn_above": 1e-2,
                        "label": "borohydride (LiBH4-based)"},
    },
    "value_ranges": {
        "ionic_conductivity_S_cm": [1e-10, 1.0],
        "temperature_K": [0, 3000],
        "electrochemical_window_V": [0, 10],
        "pressure_MPa": [0, 10000],
    },
    "unit_confusion_factors": [1000.0, 0.001],
    "structure_property": {
        "min_samples_for_correlation": 5,
        "min_abs_correlation": 0.3,
        "host_formulas": {"oxide": "Li7La3Zr2O12", "sulfide": "Li7P3S11"},
        "dopant_elements": ["Al", "Ga", "Nb", "Ta", "In", "B", "Y", "Zr",
                            "Ti", "Ge", "Sb", "W", "Mo", "Ca", "Sr", "Ba",
                            "Fe", "Zn", "Mg"],
    },
}

_cache: dict[str, Any] | None = None


def load_material_config(path: str | Path | None = None, reload: bool = False) -> dict:
    """Return the merged domain config (file values over fallback defaults)."""
    global _cache
    if _cache is not None and not reload and path is None:
        return _cache

    config: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v)
                              for k, v in FALLBACK.items()}
    target = Path(path) if path else CONFIG_PATH
    try:
        import yaml

        with open(target, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                # Shallow-merge one level so a partial YAML file still works.
                if isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key].update(value)
                else:
                    config[key] = value
    except Exception:  # noqa: BLE001 - config problems must never break a run
        pass

    if path is None:
        _cache = config
    return config


def conductivity_window(family: str, config: dict | None = None) -> tuple[float, float]:
    """Plausible conductivity range for a family, in S/cm."""
    cfg = config or load_material_config()
    entry = cfg.get("conductivity_windows", {}).get(family)
    if not entry:
        return 1e-12, 1.0
    lo, hi = entry.get("plausible_range", [1e-12, 1.0])
    return float(lo), float(hi)


def family_keywords(config: dict | None = None) -> list[tuple[str, tuple[str, ...]]]:
    """Ordered (family, keywords) pairs, mirroring the old class attribute."""
    cfg = config or load_material_config()
    order = ["polymer", "borohydride", "sulfide", "halide", "oxide"]
    raw = cfg.get("family_keywords", {})
    ordered = [(f, tuple(raw[f])) for f in order if f in raw]
    ordered += [(f, tuple(kw)) for f, kw in raw.items() if f not in order]
    return ordered
