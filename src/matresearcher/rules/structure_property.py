"""Quantitative structure-property (构效关系) analysis.

This module turns the extracted, normalized knowledge records into statistical
structure-property insights instead of leaving them as散文-style observations.

It is deliberately dependency-light on the pipeline itself:
  - pure-Python element parser (no chemistry library required)
  - Pearson / Spearman correlation via numpy + scipy (both already in the env)
  - everything configurable through config/materials.yaml

The output is a list of *significant* correlations (n >= min_samples_for_correlation
and |r| >= min_abs_correlation) so the report only shows claims it can defend
with the data actually collected — addressing the audit finding that the original
pipeline had "zero code" for quantitative structure-property modeling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .material_config import load_material_config

_ELEM_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_formula(formula: str | None) -> dict[str, float]:
    """Parse a chemical formula into {element: count}.

    Handles fractional stoichiometry (e.g. "Li6.4La3Zr1.4Ta0.6O12") and is
    tolerant of parenthetical substituents / phase labels (they are stripped).
    Returns {} for empty / unparseable input.
    """
    if not formula:
        return {}
    cleaned = re.sub(r"\([^)]*\)", "", str(formula))
    cleaned = re.sub(r"[^A-Za-z0-9.\-]", "", cleaned)
    counts: dict[str, float] = {}
    for elem, num in _ELEM_RE.findall(cleaned):
        # element symbol must not be part of a larger lowercase run (e.g. 'latp')
        if not elem or not elem[0].isupper():
            continue
        try:
            value = float(num) if num not in ("", ".") else 1.0
        except ValueError:
            value = 1.0
        counts[elem] = counts.get(elem, 0.0) + value
    return counts


def dopant_fraction(composition: str | None, config: dict | None = None) -> float | None:
    """Fraction of atoms that are dopant species, relative to total atoms.

    Dopant elements are taken from config (structure_property.dopant_elements).
    Returns None when the formula can't be parsed or no dopant is present.
    """
    if not composition:
        return None
    cfg = config or load_material_config()
    dopants: set[str] = set(
        cfg.get("structure_property", {}).get("dopant_elements", [])
    )
    counts = parse_formula(composition)
    if not counts:
        return None
    total = sum(counts.values())
    if total <= 0:
        return None
    doped = sum(v for el, v in counts.items() if el in dopants)
    if doped <= 0:
        return 0.0
    return doped / total


@dataclass
class CorrelationResult:
    feature: str
    n: int
    pearson_r: float
    spearman_r: float | None
    p_value: float
    direction: str            # "positive" | "negative" | "none"
    strength: str             # "weak" | "moderate" | "strong"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "n": self.n,
            "pearson_r": round(self.pearson_r, 3),
            "spearman_r": round(self.spearman_r, 3) if self.spearman_r is not None else None,
            "p_value": round(self.p_value, 4),
            "direction": self.direction,
            "strength": self.strength,
            "note": self.note,
        }


@dataclass
class StructurePropertyReport:
    n_records: int = 0
    correlations: list[CorrelationResult] = field(default_factory=list)
    dopant_findings: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "correlations": [c.to_dict() for c in self.correlations],
            "dopant_findings": self.dopant_findings,
            "warnings": self.warnings,
        }


def _strength(abs_r: float) -> str:
    if abs_r >= 0.7:
        return "strong"
    if abs_r >= 0.4:
        return "moderate"
    return "weak"


def _correlate(x: np.ndarray, y_log: np.ndarray) -> CorrelationResult | None:
    """Compute Pearson + Spearman with p-value for one feature vs log conductivity."""
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y_log) == 0:
        return None
    r = float(np.corrcoef(x, y_log)[0, 1])
    # two-sided p-value via t-distribution (n-2 dof)
    try:
        from scipy import stats
        t = r * np.sqrt((n - 2) / max(1e-12, (1 - r * r)))
        p = float(2 * (1 - stats.t.cdf(abs(t), n - 2)))
        try:
            rho, p_s = stats.spearmanr(x, y_log)
            spearman = float(rho)
        except Exception:
            spearman = None
    except Exception:
        p = float("nan")
        spearman = None
    direction = "positive" if r > 0 else ("negative" if r < 0 else "none")
    return CorrelationResult(
        feature="",
        n=n,
        pearson_r=r,
        spearman_r=spearman,
        p_value=p,
        direction=direction,
        strength=_strength(abs(r)),
    )


_FEATURES = [
    ("sintering_temperature_K", "烧结温度"),
    ("test_temperature_K", "测试温度"),
    ("pressure_MPa", "压力"),
    ("electrochemical_window_V", "电化学窗口"),
]


def analyze_structure_property(records: list[Any], config: dict | None = None) -> StructurePropertyReport:
    """Compute structure-property correlations across the normalized records.

    Args:
        records: list of NormalizedRecord (Pydantic models) or dicts with the
                 same fields (ionic_conductivity_S_cm, *_K, *_V, *_MPa,
                 material_composition).
        config: optional merged material config (defaults to materials.yaml).

    Returns a StructurePropertyReport with only statistically significant,
    config-thresholded correlations.
    """
    cfg = config or load_material_config()
    sp_cfg = cfg.get("structure_property", {})
    min_n = int(sp_cfg.get("min_samples_for_correlation", 5))
    min_abs_r = float(sp_cfg.get("min_abs_correlation", 0.3))

    report = StructurePropertyReport()
    y: list[float] = []
    feats: dict[str, list[float]] = {k: [] for k, _ in _FEATURES}
    comps: list[str] = []

    for rec in records:
        cond = getattr(rec, "ionic_conductivity_S_cm", None)
        if cond is None or cond <= 0:
            continue
        y.append(np.log10(cond))
        for key, _ in _FEATURES:
            feats[key].append(getattr(rec, key, None))
        comps.append(getattr(rec, "material_composition", None))
    report.n_records = len(y)

    if len(y) < min_n:
        report.warnings.append(
            f"可用记录仅 {len(y)} 条，低于最小样本阈值 {min_n}，未输出构效相关性结论"
        )
        return report

    y_arr = np.array(y, dtype=float)

    for key, label in _FEATURES:
        xs = [v for v in feats[key] if v is not None]
        # align y to non-None feature values
        paired = [(getattr(rec, key, None), getattr(rec, "ionic_conductivity_S_cm", None))
                  for rec in records
                  if getattr(rec, "ionic_conductivity_S_cm", None) is not None
                  and getattr(rec, key, None) is not None]
        if len(paired) < min_n:
            continue
        x_arr = np.array([p[0] for p in paired], dtype=float)
        yy = np.array([np.log10(p[1]) for p in paired], dtype=float)
        res = _correlate(x_arr, yy)
        if res is None or abs(res.pearson_r) < min_abs_r:
            continue
        res.feature = f"{label} ({key})"
        if res.p_value != res.p_value or res.p_value > 0.05:
            res.note = "相关性未达显著水平 (p>0.05)，仅作趋势参考"
        report.correlations.append(res)

    # Dopant fraction vs conductivity
    dopant_pairs = [
        (dopant_fraction(c, cfg), np.log10(getattr(rec, "ionic_conductivity_S_cm")))
        for c, rec in zip(comps, records)
        if getattr(rec, "ionic_conductivity_S_cm", None) is not None
        and dopant_fraction(c, cfg) is not None
    ]
    valid_dopant = [(d, l) for d, l in dopant_pairs if d is not None]
    if len(valid_dopant) >= min_n:
        d_arr = np.array([d for d, _ in valid_dopant], dtype=float)
        l_arr = np.array([l for _, l in valid_dopant], dtype=float)
        res = _correlate(d_arr, l_arr)
        if res is not None and abs(res.pearson_r) >= min_abs_r:
            res.feature = "掺杂比例 (dopant fraction)"
            report.correlations.append(res)
            # rank materials by dopant fraction for a concrete narrative
            ranked = sorted(
                (
                    (c, d, getattr(rec, "ionic_conductivity_S_cm"))
                    for c, rec, (d, _) in zip(comps, records, dopant_pairs)
                    if d is not None and getattr(rec, "ionic_conductivity_S_cm", None) is not None
                ),
                key=lambda t: t[1],
            )
            report.dopant_findings.append({
                "direction": res.direction,
                "pearson_r": round(res.pearson_r, 3),
                "lowest_dopant": ranked[0][:2] if ranked else None,
                "highest_dopant": ranked[-1][:2] if ranked else None,
            })

    # Strongest correlation first
    report.correlations.sort(key=lambda c: abs(c.pearson_r), reverse=True)
    return report


def format_structure_property_md(report: StructurePropertyReport) -> str:
    """Render the structure-property analysis as a markdown section."""
    if report.n_records == 0:
        return ""
    lines = ["## 构效关系定量分析 (Structure-Property Analysis)", ""]
    lines.append(
        f"基于 {report.n_records} 条归一化材料记录的统计相关性分析"
        f"（仅列出达到样本量与相关系数阈值的显著关系）。"
    )
    lines.append("")

    if report.warnings:
        for w in report.warnings:
            lines.append(f"> ⚠️ {w}")
        lines.append("")

    if not report.correlations:
        lines.append("当前数据量/覆盖度不足以支撑稳健的构效相关性结论；"
                     "该分析已在后续复盘中列为重点扩展方向。")
        return "\n".join(lines)

    lines.append("| 特征 | 样本数 n | Pearson r | Spearman ρ | 方向 | 强度 | 显著性 |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in report.correlations:
        p = f"{c.p_value:.3f}" if c.p_value == c.p_value else "n/a"
        sig = "显著" if (c.p_value == c.p_value and c.p_value <= 0.05) else "趋势"
        lines.append(
            f"| {c.feature} | {c.n} | {c.pearson_r:+.3f} | "
            f"{c.spearman_r:+.3f} | {c.direction} | {c.strength} | {sig} |"
        )
    lines.append("")

    if report.dopant_findings:
        for f in report.dopant_findings:
            lines.append(
                f"- 掺杂比例与离子电导率的相关系数为 {f['pearson_r']:+.3f}"
                f"（{f['direction']}，最高掺杂样本与最低掺杂样本的电导率差异见上表）。"
            )
            lines.append("")

    return "\n".join(lines)
