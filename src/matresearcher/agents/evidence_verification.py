"""Evidence Verification Agent .

Part 1: Coverage check — verify search recall is sufficient, loop back to Step 3 if not.
Part 2: Data quality check — verify extracted data integrity (handled in knowledge_extraction.py).
Part 3: Evidence verification — backtrack claims to original full-text passages.
Part 4: Final report fact-checking — verify all factual claims in the draft report.

This agent is the "gatekeeper" ensuring quality at multiple checkpoints.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from ..state import WorkflowState
from .base import BaseAgent, PROMPTS_DIR
from ..tools.sciverse import SciverseClient


class EvidenceVerificationAgent(BaseAgent):
    name = "evidence_verification"
    role = "证据核查 Agent"

    def __init__(self, llm=None, config=None,
                 sciverse: Optional[SciverseClient] = None, log_dir=None):
        super().__init__(llm, config, log_dir=log_dir)
        self.sciverse = sciverse
        self.max_verification_rounds = self.config.get("max_verification_rounds", 2)
        # ── Verification thresholds (configurable, no longer hardcoded) ──
        # min_match_score       : per-reference passage match score to count as verified
        # min_verified_coverage : fraction of anchor-bearing refs that must verify
        #   before the whole Gap is marked "passed".
        # Old logic required verified == anchor_total (100%), so one weak
        # reference downgraded an otherwise well-supported Gap to "partial" —
        # that is why an earlier run reported "0/3 gaps passed".
        self.pass_score = float(self.config.get("min_match_score", 0.5))
        self.min_coverage = float(self.config.get("min_verified_coverage", 0.8))

    async def run(self, state: WorkflowState) -> dict:
        """This is a dispatch agent — the workflow engine calls specific methods
        for each step rather than calling run() directly.
        """
        return {}

    # --- Coverage Check ---

    async def check_coverage(self, state: dict) -> dict:
        """Check if the current literature search has sufficient coverage.

        Checks:
        - Number of candidates meets minimum threshold
        - All subtask dimensions have representative papers
        - Year distribution is appropriate

        If coverage is insufficient, flags for retry (max 2 retries).
        """
        candidates = state.get("candidate_literature", [])
        subtasks = state.get("subtasks", [])
        retry_count = state.get("coverage_retry_count", 0)

        min_candidates = self.config.get("coverage_min_candidates", 10)
        min_per_subtask = self.config.get("coverage_min_per_subtask", 2)

        self.log(f"  Coverage check (retry {retry_count}/2)")
        self.log(f"  Candidates: {len(candidates)}, Subtasks: {len(subtasks)}")

        issues = []

        # Check 1: Minimum candidate count
        if len(candidates) < min_candidates:
            issues.append({
                "type": "insufficient_candidates",
                "detail": f"Only {len(candidates)} candidates (minimum {min_candidates})",
            })

        # Check 2: Each subtask dimension has coverage
        if self.llm and subtasks and len(candidates) >= 3:
            try:
                dim_coverage = await self._check_subtask_coverage(candidates, subtasks)
                for dim in dim_coverage:
                    if dim["count"] < min_per_subtask:
                        issues.append({
                            "type": "low_subtask_coverage",
                            "dimension": dim["dimension"],
                            "detail": f"Only {dim['count']} papers for dimension '{dim['dimension']}'",
                        })
            except Exception:
                pass  # Skip LLM check on error

        # Check 3: Year distribution
        years = [
            lit.metadata.year for lit in candidates
            if lit.metadata.year is not None
        ]
        if years:
            recent = sum(1 for y in years if y >= 2020)
            recent_ratio = recent / len(years) if years else 0
            if recent_ratio < 0.3:
                issues.append({
                    "type": "poor_year_distribution",
                    "detail": f"Only {recent_ratio:.0%} papers from 2020+",
                })

        passed = len(issues) == 0

        result = {
            "coverage_check_result": {
                "passed": passed,
                "issues": issues,
                "candidate_count": len(candidates),
                "subtask_count": len(subtasks),
                "retry_count": retry_count,
            },
            "coverage_retry_count": retry_count + 1 if not passed else retry_count,
        }

        if not passed:
            self.log(f"Coverage check FAILED: {len(issues)} issue(s)", "red")
            for issue in issues:
                self.log(f"  - {issue['type']}: {issue.get('detail', '')}", "yellow")
        else:
            self.log("Coverage check PASSED", "green")

        return result

    async def _check_subtask_coverage(
        self, candidates: list, subtasks: list
    ) -> list[dict]:
        """Use LLM to check if each subtask dimension has sufficient paper coverage."""
        abstracts = []
        for lit in candidates[:20]:  # Sample for efficiency
            abstract = lit.metadata.abstract or lit.metadata.title
            abstracts.append(f"- [{lit.id}] {abstract}")

        dims = "\n".join(
            f"- {s.get('dimension', s.get('id', ''))}: {s.get('semantic_query', '')}"
            for s in subtasks
        )

        try:
            prompt_path = PROMPTS_DIR / "coverage_check.txt"
            template = Path(prompt_path).read_text(encoding="utf-8")
            prompt = template.format(dims=dims, papers="\n".join(abstracts))
            result = await self.llm.complete_json(
                "You are a materials science literature review expert. "
                "Evaluate coverage of each research subtask dimension by candidate papers. "
                "Return ONLY a valid JSON array.",
                prompt,
                stage="coverage_check",  # pipeline-stage label for token accounting
            )
            return result if isinstance(result, list) else []
        except Exception:
            return []

    # --- Evidence Verification (Gap Backtracking) ---

    async def verify_gaps(self, state: dict) -> dict:
        """Verify Research Gaps by backtracking claims to original full-text passages.

        Uses concurrent Sciverse locate_passage calls (Semaphore=5) for performance.

        Concurrency safety: tasks return (gap_id, ok, score, reason, detail) tuples;
        gap status/counters are aggregated AFTER asyncio.gather, so no shared
        counter is ever mutated from inside the concurrent workers.
        """
        gaps = state.get("scored_gaps", state.get("gaps", []))

        if not gaps:
            self.log("No gaps to verify", "yellow")
            return {}

        if not self.sciverse:
            self.log("Sciverse API not available, skipping evidence verification", "yellow")
            return {"scored_gaps": gaps}

        self.log(f"  Verifying evidence for {len(gaps)} gaps (concurrent)")

        # Flatten all (gap, ref) pairs for concurrent execution
        sem = asyncio.Semaphore(5)
        tasks = []
        skipped_no_anchor = 0
        total_refs = sum(len(g.supporting_literature) for g in gaps)

        for gap in gaps:
            for ref in gap.supporting_literature:
                # A-1 dual-anchor: doc_id OR doi is enough to verify.
                if not ref.has_anchor:
                    skipped_no_anchor += 1
                    self.log(
                        f"  SKIP: ref '{ref.title[:50]}...' in {gap.gap_id} "
                        f"has no DOI nor doc_id",
                        "yellow",
                    )
                    continue
                tasks.append(self._verify_ref(sem, gap.gap_id, ref))

        if skipped_no_anchor:
            self.log(
                f"  {skipped_no_anchor}/{total_refs} supporting refs lack "
                f"traceability anchors (DOI/doc_id) — these refs cannot be "
                f"verified via Sciverse",
                "yellow" if skipped_no_anchor == total_refs else None,
            )

        # ── Aggregate results AFTER gather (no shared counter mutation) ──
        from collections import defaultdict
        per_gap: dict[str, list] = defaultdict(list)
        if tasks:
            self.log(f"Dispatching {len(tasks)} verification tasks...")
            results = await asyncio.gather(*tasks)
            for gap_id, doi, ok, score, reason, detail in results:
                per_gap[gap_id].append(
                    {"doi": doi, "ok": ok, "score": score, "reason": reason, "detail": detail}
                )
        else:
            self.log(
                "WARNING: No verification tasks to dispatch — all refs lack "
                "traceability anchors (DOI/doc_id). Check that gap_identification "
                "enriched refs from filtered_literature.",
                "red",
            )

        for gap in gaps:
            items = per_gap.get(gap.gap_id, [])
            verified = sum(1 for it in items if it["ok"])
            failed = len(items) - verified
            total = len(gap.supporting_literature)
            # 无锚点 (DOI/doc_id) 的 ref 无法核验，不应计入通过/失败判定 —
            # 否则「可核验部分全过」的 gap 会被误判为 partial。
            anchor_total = sum(1 for ref in gap.supporting_literature if ref.has_anchor)

            corrections = [it["detail"] for it in items if not it["ok"]]
            if corrections:
                gap.correction_suggestions = corrections

            # ── Coverage-based judgement (replaces all-or-nothing) ──
            # A Gap passes when at least `min_verified_coverage` of its
            # anchor-bearing references verify; the failing refs stay listed in
            # correction_suggestions so the report can disclose them.
            coverage = (verified / anchor_total) if anchor_total else 0.0
            gap.verification_coverage = round(coverage, 3)

            if anchor_total > 0 and coverage >= self.min_coverage:
                gap.verification_status = "passed"
            elif verified > 0:
                gap.verification_status = "partial"
            elif anchor_total > 0:
                gap.verification_status = "failed"
            # else: anchor_total == 0 → 保持 pending，由下方 unverified 循环接管

            gap.verification_notes = (
                f"Verified: {verified}, Failed: {failed}, Total refs: {total}, "
                f"Anchor refs: {anchor_total}, Coverage: {coverage:.0%} "
                f"(pass threshold {self.min_coverage:.0%})"
            )

        # ── Mark gaps that could not be verified at all (all refs skipped) ──
        # A gap stuck on "pending" with refs means every ref was skipped for
        # lack of an anchor (DOI/doc_id) — expose that explicitly.
        for gap in gaps:
            if gap.verification_status == "pending":
                total = len(gap.supporting_literature)
                gap.verification_status = "unverified"
                gap.verification_notes = (
                    f"No supporting literature to verify"
                    if total == 0
                    else f"All {total} supporting ref(s) lack traceability "
                         f"anchors (DOI/doc_id) — cannot verify via Sciverse"
                )

        passed = sum(1 for g in gaps if g.verification_status == "passed")
        partial = sum(1 for g in gaps if g.verification_status == "partial")
        failed = sum(1 for g in gaps if g.verification_status == "failed")
        unverified = sum(1 for g in gaps if g.verification_status == "unverified")
        self.log(
            f"Evidence verification: {passed} passed, {partial} partial, "
            f"{failed} failed, {unverified} unverified"
        )

        # Per-gap detail — include failure reasons so they are visible in logs
        for g in gaps:
            self.log(
                f"  {g.gap_id}: status={g.verification_status}, "
                f"refs={len(g.supporting_literature)}, "
                f"notes={g.verification_notes or 'N/A'}"
            )
            for s in g.correction_suggestions:
                self.log(f"    - {s}", "yellow")

        return {"scored_gaps": gaps}

    async def _verify_ref(
        self,
        sem: asyncio.Semaphore,
        gap_id: str,
        ref,
    ) -> tuple:
        """Verify a single reference. Returns (gap_id, doi, ok, score, reason, detail).

        Never mutates shared gap state — the caller aggregates after gather.
        Failure reasons are logged per-ref so they are visible in the console.

        A-1 dual-anchor: locate_passage is called with BOTH doi and doc_id;
        SciverseClient tries the doc_id-first path, then falls back to DOI.
        """
        anchor = ref.doi or ref.doc_id or "no-anchor"
        async with sem:
            try:
                result = await self.sciverse.locate_passage(
                    doi=ref.doi or "",
                    claim=ref.finding,
                    doc_id=ref.doc_id,
                )
                score = result.get("match_score", 0.0)
                passage = result.get("passage", "") or ""
                reason = result.get("reason", "matched")

                if passage and score >= self.pass_score:
                    # Verification result goes into dedicated fields —
                    # ref.finding keeps the original claim text unmodified.
                    ref.verified_passage = passage[:500]
                    ref.verification_score = score
                    self.log(
                        f"  VERIFIED: {anchor} score={score:.2f} reason={reason} "
                        f"(in {gap_id})",
                        "green",
                    )
                    return (gap_id, anchor, True, score, reason, "")
                else:
                    detail = (
                        f"Claim about {anchor} could not be verified in full text "
                        f"(match_score={score:.2f}, reason={reason})"
                    )
                    self.log(
                        f"  FAILED: {anchor} score={score:.2f} reason={reason} "
                        f"(in {gap_id})",
                        "red",
                    )
                    return (gap_id, anchor, False, score, reason, detail)
            except Exception as e:
                detail = f"Verification failed for {anchor}: {type(e).__name__}: {e}"
                self.log(f"  ERROR: {anchor} ({gap_id}): {e}", "red")
                return (gap_id, anchor, False, 0.0, f"error: {e}", detail)

    # --- Step 12: Final Report Fact-Check ---

    async def fact_check_report(self, state: dict) -> dict:
        """Verify all factual claims in the draft report (Step 12).

        Ensures:
        - Every factual claim has a literature source
        - Every numeric value has a raw_quote backing
        - No hallucinated data exists
        - Cross-literature inferences are marked as [跨文献推论]
        - Unverified hypotheses are marked as [待验证假设]
        """
        draft = state.get("draft_report", "")
        if not draft:
            self.log("No draft report to check", "yellow")
            return {}

        self.log("Step 12: Fact-checking draft report...")

        # Check 1: Report section markers
        has_fact = "[文献事实]" in draft
        has_inference = "[跨文献推论]" in draft
        has_hypothesis = "[待验证假设]" in draft

        checks = {
            "has_literature_facts": has_fact,
            "has_cross_inferences": has_inference,
            "has_verification_hypotheses": has_hypothesis,
        }

        issues = []
        if not has_inference:
            issues.append("报告缺少 [跨文献推论] 标记 — 跨文献推理内容需明确标注")
        if not has_hypothesis:
            issues.append("报告缺少 [待验证假设] 标记 — 未验证的推断性结论需明确标注")

        # Check 2: Use LLM for deeper fact-checking if available
        if self.llm:
            try:
                fact_check_result = await self._llm_fact_check(draft)
                issues.extend(fact_check_result.get("issues", []))
            except Exception as e:
                self.log(f"LLM fact-check failed: {e}", "yellow")

        # Check 3: Scan for suspicious patterns (hallucination indicators)
        # 只扫正文部分 — 附录 A 证据清单是工程生成的表格（含高精度数值与
        # 溯源锚点列），启发式"数值缺引用"扫描会对其误报。
        body_text = draft.split("\n## 附录")[0]
        suspicious_patterns = self._scan_suspicious_patterns(body_text)
        issues.extend(suspicious_patterns)

        # Check 4 (A-2): Evidence traceability — untraceable / unverified / failed refs
        # 数据源优先用 state 里的结构化 gaps；若拿不到（state 未传），
        # 降级为扫描 draft 中附录 A 的「不可追溯」标记。
        gaps = state.get("scored_gaps", state.get("gaps", []))
        if gaps:
            trace_issues = self._scan_untraceable_evidence(gaps)
            issues.extend(trace_issues)
            untraceable = sum(
                1 for g in gaps for r in g.supporting_literature if not r.has_anchor
            )
            self.log(
                f"Evidence traceability: {len(gaps)} gaps, "
                f"{untraceable} untraceable ref(s)",
                "yellow" if untraceable else "green",
            )
        else:
            manifest_issues = self._scan_manifest_in_draft(draft)
            issues.extend(manifest_issues)

        # ── Closed loop: regenerate instead of merely logging ──
        # Old behaviour: `final_report = draft + issue log` — the report shipped
        # with its problems still in the body. Now we ask the graph to send the
        # draft back to report_generation (bounded by max_revisions) and only
        # fall back to appending the log when the budget is exhausted.
        revision = int(state.get("fact_check_revision", 0) or 0) + 1
        max_revisions = int(self.config.get("max_revisions", 2))
        needs_revision = bool(issues) and revision <= max_revisions

        corrections = [f"[修正建议] {issue}" for issue in issues]
        final_report = draft

        if needs_revision:
            status = "revising"
            self.log(
                f"Found {len(issues)} issue(s) — sending draft back for "
                f"revision {revision}/{max_revisions}",
                "yellow",
            )
        elif issues:
            status = "unresolved"
            corrections_section = "\n\n---\n## 事实核查记录 (Fact-Check Log)\n\n"
            corrections_section += (
                f"> 已达最大修订轮次 ({max_revisions})，以下问题未能自动修正，"
                f"请人工复核后再使用该报告。\n\n"
            )
            corrections_section += "\n".join(f"- {c}" for c in corrections)
            final_report = draft + corrections_section
            self.log(
                f"{len(issues)} issue(s) remain after {max_revisions} revision(s); "
                f"appended to report for human review",
                "red",
            )
        else:
            status = "revised" if revision > 1 else "clean"
            self.log("All factual claims verified", "green")

        return {
            "final_report": final_report,
            "fact_check_checks": checks,
            "fact_check_issues": issues,
            "fact_check_corrections": corrections,
            "fact_check_revision": revision,
            "fact_check_status": status,
            "needs_revision": needs_revision,
        }

    async def _llm_fact_check(self, draft: str) -> dict:
        """Use LLM to perform deep fact-checking."""
        prompt = f"""You are a rigorous fact-checker for a materials science literature survey report.

Review this draft report and identify:
1. Any factual claims that are presented as facts but appear to be cross-literature inferences
2. Any numeric values that lack source attribution
3. Any overgeneralizations
4. Any internally contradictory statements

Report format:
{{
  "issues": ["issue1", "issue2", ...],
  "overall_confidence": "high/medium/low"
}}

Report to check:
{draft[:8000]}"""

        result = await self.llm.complete_json(
            "You are a rigorous scientific fact-checker.",
            prompt,
            stage="fact_check",  # pipeline-stage label for token accounting
        )
        return result if isinstance(result, dict) else {"issues": []}

    def _scan_suspicious_patterns(self, text: str) -> list[str]:
        """Scan for patterns that suggest hallucination."""
        import re
        issues = []

        # Pattern 1: Very precise numbers without citations
        # e.g., "the conductivity was 1.2345e-4 S/cm" without a reference
        precise_numbers = re.findall(
            r'([\d]+\.[\d]{4,}\s*(?:[×xX]?\s*10[⁻⁽]?\d+[⁾]?\s*)?(?:S/cm|mS/cm|V|MPa|GPa|K))',
            text,
        )
        if precise_numbers:
            # Check if these numbers are immediately followed by a citation
            for num in precise_numbers:
                context = text[max(0, text.find(num) - 30):text.find(num) + len(num) + 30]
                if not re.search(r'\[[\d,]+\]|DOI:|et al\.|\(\d{4}\)', context):
                    issues.append(
                        f"精确数值 '{num}' 缺少文献引用：请标注来源或降级为 [跨文献推论]"
                    )

        # Pattern 2: Absolute claims
        if re.search(r'(?:is the best|highest ever|first time|never before)', text, re.IGNORECASE):
            issues.append("检测到绝对化表述 (best/highest/first/never) — 请核实是否有充分证据")

        # Pattern 3: Claims without any citation nearby
        sentences = re.split(r'(?<=[。.])', text)
        assertion_count = 0
        for sent in sentences:
            if len(sent.strip()) > 30 and not re.search(r'\[[\d,]+\]|et al|DOI:', sent):
                assertion_count += 1
        if assertion_count > 5:
            issues.append(
                f"报告中有 {assertion_count} 个陈述句未附带引用 — 建议补充文献来源"
            )

        return issues

    # --- A-2: Evidence traceability checks ---

    def _scan_untraceable_evidence(self, gaps: list) -> list[str]:
        """Check gap supporting refs for traceability (A-2).

        对每条支撑文献检查三类问题：
        - 不可追溯：无 doc_id/DOI 锚点 → 结论无法回溯到原文
        - 未核验：有锚点但 verification_score 为空（未执行核验或核验跳过）
        - 核验失败：有锚点但分数 ≤ 0.5（Sciverse 全文库未找到支持段落）

        返回可读的 issue 列表，由 fact_check_report 汇总进「事实核查记录」。
        """
        issues = []
        for gap in gaps:
            for ref in gap.supporting_literature:
                anchor = self._short_anchor(ref)
                title = (ref.title or "N/A")[:60]
                if not ref.has_anchor:
                    issues.append(
                        f"不可追溯证据: 缺口 {gap.gap_id} 的支撑文献 "
                        f"'{title}' 无 doc_id/DOI 溯源锚点 — 该结论无法回溯到原文"
                    )
                elif ref.verification_score is None:
                    issues.append(
                        f"未核验证据: 缺口 {gap.gap_id} 的支撑文献 "
                        f"'{title}' (锚点 {anchor}) 未经证据核验，"
                        f"结论可信度待确认"
                    )
                elif ref.verification_score is not None and ref.verification_score < self.pass_score:
                    issues.append(
                        f"核验失败: 缺口 {gap.gap_id} 的支撑文献 "
                        f"'{title}' (锚点 {anchor}) 在 Sciverse 全文库中"
                        f"未找到支持段落 (score={ref.verification_score:.2f})"
                    )
        return issues

    def _scan_manifest_in_draft(self, draft: str) -> list[str]:
        """Fallback: scan the report's 附录 A for traceability markers.

        当 state 未携带结构化 gaps 时使用 — 从证据清单文本中统计
        「⚠️ 不可追溯」条目，保证 fact_check 不会因缺数据而静默放行。
        """
        import re
        issues = []
        if "证据溯源清单" not in draft and "证据溯源" not in draft:
            issues.append(
                "报告缺少证据溯源清单（附录 A）— 结论无法逐条回溯到原文，"
                "建议补充结构化溯源信息"
            )
            return issues
        untraceable = len(re.findall(r"⚠️ 不可追溯", draft))
        if untraceable:
            issues.append(
                f"证据清单中存在 {untraceable} 条「不可追溯」证据 "
                f"— 对应结论无 doc_id/DOI 锚点，可信度存疑"
            )
        return issues

    @staticmethod
    def _short_anchor(ref) -> str:
        """Short anchor label for log messages (doc_id 前 16 位 / DOI)."""
        if ref.doc_id:
            d = ref.doc_id
            return f"doc_id:{d[:16]}…" if len(d) > 16 else f"doc_id:{d}"
        if ref.doi:
            return f"DOI:{ref.doi}"
        return "无锚点"
