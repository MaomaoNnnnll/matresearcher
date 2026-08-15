# -*- coding: utf-8 -*-
"""Smoke test for the three-way quality gate + R5/R6/R9/R10 rules.

Zero LLM calls — constructs KnowledgeRecords directly and runs
KnowledgeExtractionAgent._quality_check + _detect_material_family
+ _arrhenius_check + _statistical_outlier_check.
"""
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from matresearcher.agents.knowledge_extraction import KnowledgeExtractionAgent
from matresearcher.models.knowledge import KnowledgeRecord, NumericValue

PASS, FAIL = 0, 0
_SEQ = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail and not cond else ""))


def rec(comp, cond=None, raw_quote="q", **kw):
    """Build a KnowledgeRecord with optional test_temperature via kw."""
    global _SEQ
    _SEQ += 1
    return KnowledgeRecord(
        id=f"lit_test_rec_{_SEQ:03d}",
        literature_id="lit_test",
        material_composition=comp,
        ionic_conductivity=NumericValue(
            value=cond, unit="S/cm", raw_quote=raw_quote,
        ) if cond is not None else None,
        **kw,
    )


def rec_temp(comp, cond, temp_k, raw_quote="q"):
    """Build a record with ionic_conductivity.temperature_K set."""
    global _SEQ
    _SEQ += 1
    return KnowledgeRecord(
        id=f"lit_test_rec_{_SEQ:03d}",
        literature_id="lit_test",
        material_composition=comp,
        ionic_conductivity=NumericValue(
            value=cond, unit="S/cm", raw_quote=raw_quote,
            temperature_K=temp_k,
        ),
    )


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        agent = KnowledgeExtractionAgent(config={}, log_dir=td)

        # ── R5: material family detection ──
        print("== R5: material family detection ==")
        cases = [
            ("Li7La3Zr2O12", "oxide"), ("LLZO", "oxide"), ("LATP", "oxide"),
            ("Li10GeP2S12", "sulfide"), ("LGPS", "sulfide"),
            ("argyrodite Li6PS5Cl", "sulfide"),
            ("Li3InCl6", "halide"), ("Li3YCl6 (chloride)", "halide"),
            ("PEO-LiTFSI", "polymer"), ("PVDF-HFP", "polymer"),
            ("LiBH4", "borohydride"), ("unknown-material", "unknown"),
            (None, "unknown"),
            # Edge case: variable x in formula + parenthetical note
            ("Li7-xLa3-xZr2-xBxO12 (x = 0.15, 0.20, 0.30)", "oxide"),
        ]
        for comp, expected in cases:
            got = agent._detect_material_family(comp)
            check(f"family({comp!r}) == {expected}", got == expected, f"got {got}")

        # ── Three-way gate (R1-R6 per-record) ──
        print("== Three-way gate (R1-R6) ==")
        records = [
            rec("Li7La3Zr2O12", cond=5e-4),           # PASS
            rec("Li7La3Zr2O12", cond=0.5),            # REVIEW (unit confusion)
            rec("Li7La3Zr2O12", cond=5.0),            # FAIL (>2 orders)
            rec("LLZO", cond=0.2),                    # REVIEW (soft outlier)
            rec("Li6PS5Cl", cond=1e-2),               # PASS
            rec("Li6PS5Cl", cond=1e-3, raw_quote=""), # FAIL (no raw_quote)
            rec("La2Zr2O7"),                          # FAIL (no numeric)
        ]
        verified, flagged = agent._quality_check(records, depth="deep")

        by_id = {r.id: r for r in verified + flagged}
        ids = [r.id for r in records]
        st = [by_id[i].quality_status for i in ids]

        check("oxide 5e-4 -> verified", st[0] == "verified", st[0])
        check("LLZO 0.5 S/cm -> review (unit confusion)",
              st[1] == "review", st[1])
        check("unit-confusion hint present",
              "unit confusion" in " ".join(by_id[ids[1]].quality_issues),
              by_id[ids[1]].quality_issues)
        check("corrected value mentioned in hint",
              "5.00e-04" in " ".join(by_id[ids[1]].quality_issues),
              by_id[ids[1]].quality_issues)
        check("oxide 5 S/cm -> anomaly (>2 orders)", st[2] == "anomaly", st[2])
        check("LLZO 0.2 S/cm -> review (soft outlier)", st[3] == "review", st[3])
        check("sulfide 1e-2 -> verified", st[4] == "verified", st[4])
        check("missing raw_quote -> anomaly", st[5] == "anomaly", st[5])
        check("no numeric data -> anomaly",
              "No numeric data" in " ".join(by_id[ids[6]].quality_issues))
        check("verified records carry no issues",
              all(not r.quality_issues for r in verified))

        # ── ECW mV confusion ──
        print("== ECW mV confusion ==")
        r_ecw = KnowledgeRecord(
            id="ecw1", literature_id="lit_test",
            material_composition="Li6PS5Cl",
            electrochemical_window=NumericValue(value=1700.0, unit="V"),
        )
        v2, f2 = agent._quality_check([r_ecw], depth="deep")
        check("ECW 1700 V -> review with mV hint",
              r_ecw.quality_status == "review"
              and "mV/V confusion" in " ".join(r_ecw.quality_issues),
              r_ecw.quality_issues)

        # ── R10: statistical outlier detection ──
        print("== R10: statistical outlier ==")

        # 6 oxide records, 5 normal + 1 outlier (4 orders below median)
        r10_records = [
            rec("Li7La3Zr2O12", cond=5e-4),
            rec("Li7La3Zr2O12", cond=4e-4),
            rec("Li7La3Zr2O12", cond=6e-4),
            rec("Li7La3Zr2O12", cond=3e-4),
            rec("Li7La3Zr2O12", cond=5e-4),
            rec("Li7La3Zr2O12", cond=1e-8),   # outlier
        ]
        r10_issues = KnowledgeExtractionAgent._statistical_outlier_check(r10_records)
        outlier_id = r10_records[-1].id
        normal_ids = [r.id for r in r10_records[:-1]]
        check("R10 flags the outlier", outlier_id in r10_issues,
              f"issues={list(r10_issues.keys())}")
        check("R10 does not flag normal records",
              all(i not in r10_issues for i in normal_ids),
              f"flagged normals={[i for i in normal_ids if i in r10_issues]}")
        check("R10 issue mentions 'statistical outlier'",
              "statistical outlier" in " ".join(r10_issues.get(outlier_id, [])))

        # R10 with too few records (<5) → skip
        r10_few = [rec("Li7La3Zr2O12", cond=5e-4) for _ in range(3)]
        check("R10 skips groups with <5 records",
              len(KnowledgeExtractionAgent._statistical_outlier_check(r10_few)) == 0)

        # R10 with tight cluster (all same value) → no outliers
        r10_tight = [rec("Li7La3Zr2O12", cond=5e-4) for _ in range(6)]
        check("R10 no false positives on tight cluster",
              len(KnowledgeExtractionAgent._statistical_outlier_check(r10_tight)) == 0)

        # ── R9: Arrhenius check ──
        print("== R9: Arrhenius check ==")

        # Normal Arrhenius points (Ea ~ 0.3 eV) → no issues
        r9_normal = [
            rec_temp("Li7La3Zr2O12", 8.4e-4, 298),
            rec_temp("Li7La3Zr2O12", 2.1e-3, 323),
            rec_temp("Li7La3Zr2O12", 4.5e-3, 348),
            rec_temp("Li7La3Zr2O12", 8.9e-3, 373),
            rec_temp("Li7La3Zr2O12", 1.6e-2, 398),
        ]
        r9_a = KnowledgeExtractionAgent._arrhenius_check(r9_normal)
        check("R9: normal Arrhenius -> no issues", len(r9_a) == 0,
              f"issues={list(r9_a.keys())}")

        # With an outlier (1e-6 at 348K, ~1000x below trend) → outlier flagged
        r9_outlier = list(r9_normal) + [rec_temp("Li7La3Zr2O12", 1e-6, 348)]
        r9_b = KnowledgeExtractionAgent._arrhenius_check(r9_outlier)
        r9_outlier_id = r9_outlier[-1].id
        check("R9: outlier flagged", r9_outlier_id in r9_b,
              f"issues={list(r9_b.keys())}")
        check("R9 issue mentions 'Arrhenius'",
              "Arrhenius" in " ".join(r9_b.get(r9_outlier_id, [])))

        # R9 with <4 points → skip
        r9_few = r9_normal[:3]
        check("R9 skips groups with <4 points",
              len(KnowledgeExtractionAgent._arrhenius_check(r9_few)) == 0)

        # R9 with <50K spread → skip (all at 298K)
        r9_nospread = [
            rec_temp("Li7La3Zr2O12", 5e-4, 298),
            rec_temp("Li7La3Zr2O12", 1e-3, 298),
            rec_temp("Li7La3Zr2O12", 2e-3, 300),
            rec_temp("Li7La3Zr2O12", 8e-4, 299),
        ]
        check("R9 skips when temperature spread <50K",
              len(KnowledgeExtractionAgent._arrhenius_check(r9_nospread)) == 0)

        # ── Quality report file ──
        print("== Quality report file ==")
        report = agent._write_quality_report(verified, flagged)
        check("report path returned", report is not None)
        data = json.loads(Path(report).read_text(encoding="utf-8"))
        check("summary counts match",
              data["summary"]["pass"] == len(verified)
              and data["summary"]["fail"] == sum(
                  1 for r in flagged if r.quality_status == "anomaly")
              and data["summary"]["review"] == sum(
                  1 for r in flagged if r.quality_status == "review"),
              data["summary"])
        check("report entries carry family",
              all("family" in e for e in data["review_records"] + data["fail_records"]))
        check("report located in log dir",
              Path(report).name == "quality_report.json")

        # Windows: close the log handle before the temp dir is removed
        agent.close_log()

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
