"""Report Generation Agent (Step 11).

Generates a structured literature survey report with:
- [文献事实] — facts directly from literature
- [跨文献推论] — inferences drawn across multiple papers
- [待验证假设] — unverified hypotheses requiring validation

Report structure:
1. 研究问题概述
2. 检索方法与文献覆盖
3. 知识体系图 (material-property matrix)
4. 数据一致性分析 (conflicts)
5. 研究缺口发现 (ranked gaps)
6. 构效关系定量分析 (Structure-Property Analysis)
7. 方法局限性
参考文献清单
附录 A. 证据溯源清单
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..state import WorkflowState
from .base import BaseAgent, PROMPTS_DIR


def _load_report_template() -> str:
    path = PROMPTS_DIR / "report_generation.txt"
    if path.exists():
        return Path(path).read_text(encoding="utf-8")
    return ""   # fallback to empty string (will fail gracefully at format time)


def _load_competition_template() -> str:
    path = PROMPTS_DIR / "competition_submission.txt"
    if path.exists():
        return Path(path).read_text(encoding="utf-8")
    return ""


def _extract_section(text: str, title_pattern: str):
    """Remove the first `## heading` section whose title matches ``title_pattern``.

    Returns ``(remaining_text, section_text_or_None)``. The captured section runs
    from its heading line up to (but not including) the next ``## `` heading, or
    to EOF. If no matching heading exists, returns ``text`` unchanged with ``None``
    — making callers idempotent across fact-check revision rounds.
    """
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        m = re.match(r"^##\s+(.*)$", line)
        if m and re.search(title_pattern, m.group(1)):
            start = i
            break
    if start is None:
        return text, None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^##\s+", lines[j]):
            end = j
            break
    section = "\n".join(lines[start:end]).strip()
    remaining = "\n".join(lines[:start] + lines[end:]).rstrip()
    return remaining, section


REPORT_TEMPLATE = _load_report_template()
COMPETITION_TEMPLATE = _load_competition_template()

# Revision prompt: edit the EXISTING draft against the fact-check issue list
# instead of regenerating from scratch. Used by _revise_with_llm on revision
# rounds so that flagged (unsupported) claims are truly deleted/demoted.
REVISION_TEMPLATE = """# 报告修订任务（基于事实核查问题清单就地改写）

## 原始研究问题
{question}

## 上一版草稿（请在此基础上做最小化修订，不要凭空重写）
{previous_draft}

## 事实核查问题清单（必须逐条处理）
{issues}

## 修订要求
1. 逐条处理上面的问题清单：对每条「没有文献支撑的精确数值或论断」，要么直接从正文中**删除**，要么**降级**标注为 [跨文献推论]。
2. 不得保留任何无法追溯出处的结论，也不得新增任何未经证据支持的数字或论断。
3. **必须完整保留**草稿的全部章节结构与其余内容，特别是以下章节原样保留、不要改动：
   - 「## 构效关系定量分析 (Structure-Property Analysis)」
   - 「## 假设外部/离线交叉核验 (Hypothesis Cross-Check)」
   - 「## 附录 A. 证据溯源清单」
4. 只做必要的删除/降级，返回修订后的**完整 markdown 草稿**。

生成时间：{generate_time}
"""


class ReportGenerationAgent(BaseAgent):
    name = "report_generation"
    role = "报告生成 Agent"

    def __init__(self, llm=None, config=None, log_dir=None):
        super().__init__(llm, config, log_dir=log_dir)
        self.output_mode = self.config.get("output_mode", "survey")  # "survey" | "submission"

    def _assemble_survey_report(
        self, draft: str, sp_md: str | None, xc_md: str | None, evidence_md: str | None
    ) -> str:
        """Reorder the survey report into the canonical section layout.

        Canonical order (after chapters 1-5):
          ## 6. 构效关系定量分析 (Structure-Property Analysis)
          ## 7. 方法局限性
          ## 参考文献清单
          ## 附录 A. 证据溯源清单

        ``draft`` comes from the LLM/template and contains chapters 1-7 (with
        ``## 6. 参考文献清单`` and ``## 7. 方法局限性``). The structure-property
        (``sp_md``) and cross-check (``xc_md``) sections are engineered and may
        already be embedded (revision rounds), so they are stripped first to keep
        this function idempotent, then re-inserted in canonical position.

        The evidence appendix (``## 附录 A``) is NOT stripped: it is appended last
        and may carry trailing subsections (e.g. ``### 修订说明`` added by the
        revision LLM). We only ensure its presence, so any trailing content after
        it survives revision rounds.
        """
        # 1. Strip the relocatable / engineered sections (idempotent on revisions).
        #    Keep 附录 A in place — see note above.
        body, refs_section = _extract_section(draft, r"参考文献")
        body, _ = _extract_section(body, r"构效关系定量")
        body, _ = _extract_section(body, r"假设外部|交叉核验")

        # 2. References: relabel to an unnumbered "## 参考文献清单".
        refs_block = ""
        if refs_section:
            refs_block = re.sub(
                r"^##\s*(?:\d+\.\s*)?参考文献清单",
                "## 参考文献清单",
                refs_section.strip(),
                count=1,
                flags=re.M,
            )

        # 3. Structure-property: force the canonical "## 6. 构效关系定量分析" heading.
        sp_block = (sp_md or "").strip()
        if sp_block:
            sp_block = re.sub(
                r"^##\s*(?:\d+\.\s*)?构效关系定量分析",
                "## 6. 构效关系定量分析",
                sp_block,
                count=1,
                flags=re.M,
            )

        # 4. Insert 6. 构效关系 (+ cross-checks) immediately before 7. 方法局限性.
        insert_block = sp_block
        if xc_md and xc_md.strip():
            xc_block = re.sub(
                r"^##\s*(?:\d+\.\s*)?假设外部/离线交叉核验",
                "## 假设外部/离线交叉核验",
                xc_md.strip(),
                count=1,
                flags=re.M,
            )
            insert_block = (insert_block + "\n\n" + xc_block) if insert_block else xc_block

        method_re = re.compile(r"(?m)^##\s*(?:\d+\.\s*)?方法局限性\b")
        if method_re.search(body):
            body = method_re.sub(
                lambda m: (insert_block + "\n\n" + m.group(0)) if insert_block else m.group(0),
                body,
                count=1,
            )
        elif insert_block:
            body = body.rstrip() + "\n\n" + insert_block

        # 5. Re-insert references right after 方法局限性 (before any following 附录 A).
        if refs_block:
            method_section_re = re.compile(
                r"^##\s*(?:\d+\.\s*)?方法局限性\b.*?(?=^##\s+|\Z)", re.S | re.M
            )
            if method_section_re.search(body):
                body = method_section_re.sub(
                    lambda m: m.group(0).rstrip() + "\n\n" + refs_block, body, count=1
                )
            else:
                body = body.rstrip() + "\n\n" + refs_block

        # 6. Ensure the evidence appendix is present (append once; never strip existing).
        if evidence_md and evidence_md.strip() and "附录 A. 证据溯源清单" not in body:
            body = body.rstrip() + "\n\n" + evidence_md.strip()

        return body

    async def run(self, state: WorkflowState) -> dict:
        question = state.get("raw_question", "No question provided")
        candidate = state.get("candidate_literature", [])
        filtered = state.get("filtered_literature", [])
        fused_table = state.get("fused_table")
        conflicts = state.get("conflicts", [])
        missing_items = state.get("missing_items", [])
        scored_gaps = state.get("scored_gaps", state.get("gaps", []))

        # ── Fact-check closed loop (Step 12 → Step 11) ──
        # The fact-check node returns a list of concrete issues. Previously we
        # only re-prompted a from-scratch regeneration with the issue list as soft
        # instructions — fixes were never guaranteed to land (audit A2). Now, on a
        # revision round we REWRITE the EXISTING draft in place against the issue
        # list (delete/demote unsupported claims), so corrections truly propagate.
        prior_issues = state.get("fact_check_issues") or []
        revision = int(state.get("fact_check_revision", 0) or 0)
        previous_draft = state.get("draft_report", "")
        is_revision = revision > 0 and bool(prior_issues) and bool(previous_draft)

        if not is_revision and prior_issues and revision > 0:
            # Fallback (no prior draft available, e.g. template/no-LLM path):
            # inject the issues as hard prompt requirements.
            fix_block = (
                f"\n\n【事实核查修正要求 — 第 {revision} 轮重写】\n"
                "上一版报告存在以下问题，必须逐条修正后才能交付：\n"
            )
            fix_block += "\n".join(f"{i}. {x}" for i, x in enumerate(prior_issues, 1))
            fix_block += (
                "\n处理原则：无法找到文献支撑的精确数值一律删除，或降级标注为 "
                "[跨文献推论]；不得保留无法追溯的结论，也不得新增未经证据支持的数字。\n"
            )
            question = f"{question}\n{fix_block}"
            self.log(
                f"Revision pass {revision}: injecting {len(prior_issues)} "
                f"fact-check issue(s) into the prompt (template fallback)"
            )
        elif is_revision:
            self.log(
                f"Revision pass {revision}: rewriting previous draft in place "
                f"against {len(prior_issues)} fact-check issue(s)"
            )

        # Compute statistics
        parsed_count = sum(1 for lit in filtered if lit.is_parsed)
        record_count = fused_table.total_records if fused_table else 0
        material_count = fused_table.total_materials if fused_table else 0

        # Build reference map: lit_id → citation info for all filtered literature
        reference_map: dict[str, dict] = {}
        for idx, lit in enumerate(sorted(filtered, key=lambda x: x.id), 1):
            meta = lit.metadata
            reference_map[lit.id] = {
                "ref_num": idx,
                "id": lit.id,
                "title": meta.title if meta else "N/A",
                "authors": ", ".join(meta.authors[:3]) if meta and meta.authors else "N/A",
                "doi": meta.doi if meta else "N/A",
                "doc_id": meta.doc_id if meta else None,  # A-2: doc_id 溯源锚点（DOI 缺失时用于文献#N 映射）
                "year": meta.year if meta else "N/A",
                "journal": meta.journal if meta else "N/A",
                "abstract": meta.abstract[:300] + "..." if meta and meta.abstract and len(meta.abstract) > 300 else (meta.abstract if meta else "N/A"),
                "keywords": ", ".join(meta.keywords[:5]) if meta and meta.keywords else "N/A",
            }

        # Format references summary for prompt
        references = self._format_references(reference_map)

        # Build knowledge table (material-property matrix)
        knowledge_table = self._build_knowledge_table(fused_table, reference_map)

        # Format conflicts
        conflicts_detail = self._format_conflicts(conflicts)

        # Format gaps (ranked, with supporting_literature references)
        gaps_detail = self._format_gaps(scored_gaps, reference_map)
        generate_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.log(f"Generating report: {material_count} materials, "
                 f"{len(scored_gaps)} gaps")

        if self.llm and is_revision:
            # ① True in-place rewrite: edit the existing draft against the
            # fact-check issue list instead of regenerating from scored_gaps.
            draft = await self._revise_with_llm(
                previous_draft=previous_draft,
                issues=prior_issues,
                question=question,
                generate_time=generate_time,
            )
        elif self.llm:
            draft = await self._generate_with_llm(
                question=question,
                candidate_count=len(candidate),
                filtered_count=len(filtered),
                parsed_count=parsed_count,
                material_count=material_count,
                record_count=record_count,
                conflict_count=len(conflicts),
                missing_count=len(missing_items),
                gap_count=len(scored_gaps),
                knowledge_table=knowledge_table,
                conflicts_detail=conflicts_detail,
                gaps_detail=gaps_detail,
                references=references,
                generate_time=generate_time,
            )
        else:
            # Fallback: template-based report
            draft = self._generate_template_based(
                question=question,
                candidate_count=len(candidate),
                filtered_count=len(filtered),
                parsed_count=parsed_count,
                material_count=material_count,
                conflict_count=len(conflicts),
                gap_count=len(scored_gaps),
                knowledge_table=knowledge_table,
                conflicts_detail=conflicts_detail,
                gaps_detail=gaps_detail,
                references=references,
                generate_time=generate_time,
            )

        self.log(f"Report generated: {len(draft)} characters")

        # ── A-2: Build structured evidence manifest (工程生成，保证可追溯) ──
        # 无论 LLM / 模板路径，证据清单都由工程层从 scored_gaps 渲染，
        # 确保每条缺口结论都能回溯到文献与原文段落。
        evidence_manifest = self._build_evidence_manifest(scored_gaps)

        # ── P3-14/P3-15: 扩展分析章节组装（固定章节顺序）──
        # 章节固定顺序：… ch1-5, 6. 构效关系定量分析, 7. 方法局限性,
        # 参考文献清单, 附录 A. 证据溯源清单。
        # 构效关系(sp_md) 与 交叉核验(xc_md) 为工程层生成，证据清单(evidence)
        # 亦为工程层生成；参考文献清单/方法局限性 来自 LLM 草稿。为保证顺序稳定
        # 且修订轮幂等，统一在 _assemble_survey_report 中重排（extract → 重插）。
        sp_md = state.get("_structure_property_md")
        xchecks = state.get("hypothesis_cross_checks")
        xc_md = self._format_cross_checks(xchecks) if xchecks else ""

        survey_report = self._assemble_survey_report(
            draft, sp_md=sp_md, xc_md=xc_md, evidence_md=evidence_manifest
        )
        if sp_md:
            self.log(f"Structure-property section assembled: {len(sp_md)} chars")
        if xc_md:
            self.log(f"Hypothesis cross-check section assembled: {len(xc_md)} chars")

        # ── Survey report 调研报告本体落盘（独立于参赛方案） ──
        # 该文件始终保存 survey 版本（含构效关系 + 参考文献清单 + 证据溯源），
        # 不被下面的 submission 改写覆盖。
        if self._log_path:
            survey_path = Path(self._log_path).parent / "survey_report.md"
            survey_path.write_text(survey_report, encoding="utf-8")
            self.log(f"Survey report persisted: {survey_path.name} ({len(survey_report)} chars)")

        # ── P2: Optional competition submission wrapper ──
        # CRITICAL: do NOT mutate `survey_report` into the submission document.
        # `survey_report` must stay the SURVEY report (with the canonical section
        # order) so that:
        #   (a) survey_report.md (written above) always holds the survey version
        #       instead of being clobbered by the submission doc on revision rounds;
        #   (b) the fact-check revision loop reads state["draft_report"] as
        #       previous_draft and rewrites the SURVEY report, not the submission.
        # The submission is produced into a separate variable and exposed via the
        # state key "submission_report" (consumed by main.py / engine for report.md).
        submission_report = ""
        if self.output_mode == "submission" and self.llm and survey_report:
            submission_report = await self._generate_submission(survey_report)
            self.log(f"Competition submission generated: {len(submission_report)} characters")
            if evidence_manifest and "附录 A. 证据溯源清单" not in submission_report:
                submission_report = submission_report.rstrip() + "\n\n" + evidence_manifest
                self.log("Evidence manifest appended to submission report")

        return {"draft_report": survey_report, "submission_report": submission_report}

    async def _generate_with_llm(self, **kwargs) -> str:
        """Generate report using LLM, with post-hoc §6/§7 completeness guard."""
        prompt = REPORT_TEMPLATE.format(**kwargs)
        try:
            result = await self.llm.complete(
                "你是一位材料科学文献综述专家，擅长撰写结构化、可查证的科研调研报告。"
                "请务必生成包含全部7个章节的完整报告，不要遗漏任何一节。",
                prompt,
                max_tokens=16384,
                stage=self.name,
                # 16384-token generations on reasoning models run 90-150 s;
                # the 120 s client default is too tight (2026-08-14 submission
                # timeout: 3 x 120 s SDK retries = 6 min before failing).
                timeout=300,
            )
        except Exception as e:
            self.log(f"LLM report generation failed: {e}, using template", "yellow")
            return self._generate_template_based(**kwargs)

        # ── Post-generation completeness check ──
        result = self._ensure_required_sections(result, **kwargs)
        return result

    async def _revise_with_llm(
        self,
        previous_draft: str,
        issues: list[str],
        question: str,
        generate_time: str,
    ) -> str:
        """Revision pass: rewrite the EXISTING draft against the fact-check issue list.

        Unlike _generate_with_llm (which rebuilds a fresh draft from scored_gaps and
        merely re-injects the issues as a soft prompt), this edits the prior draft
        in place so that unsupported claims are genuinely deleted/demoted — fixing
        the closed-loop semantics (audit A2): the report can no longer ship with the
        very claims fact_check just flagged.
        """
        issue_block = "\n".join(f"{i}. {x}" for i, x in enumerate(issues, 1))
        prompt = REVISION_TEMPLATE.format(
            question=question,
            previous_draft=previous_draft,
            issues=issue_block,
            generate_time=generate_time,
        )
        try:
            result = await self.llm.complete(
                "你是一位严谨的材料科学文献综述编辑。请基于事实核查问题清单，对已有草稿"
                "做最小化、精准的修订：删除或降级无文献支撑的论断，其余内容（含构效关系、"
                "交叉核验、证据溯源清单等章节）原样保留，不要凭空重写。",
                prompt,
                max_tokens=16384,
                stage=self.name,
                timeout=300,
            )
        except Exception as e:
            self.log(f"LLM revision failed: {e}, keeping previous draft", "yellow")
            return previous_draft
        # Keep §6/§7 guarantees consistent with the generation path.
        result = self._ensure_required_sections(result)
        self.log(f"Revision draft generated: {len(result)} characters")
        return result

    def _ensure_required_sections(self, report: str, **kwargs) -> str:
        """Ensure §6 参考文献清单 and §7 方法局限性 are present.
        
        If LLM output is truncated or omits these sections, append default content.
        """
        has_refs = "## 6." in report or "参考文献清单" in report or "参考文献" in report.split("## 5.")[-1] if "## 5." in report else True
        has_limits = "## 7." in report or "方法局限性" in report

        if not has_refs or not has_limits:
            self.log("Missing required sections in LLM output, appending defaults", "yellow")

        if not has_refs:
            references_text = kwargs.get("references", "（无可用文献数据）")
            report += f"\n\n## 6. 参考文献清单\n\n{references_text}\n"

        if not has_limits:
            report += """\n\n## 7. 方法局限性

- **检索覆盖范围**: 依赖 Sciverse API 进行文献检索，可能存在数据库收录不全、非英文文献覆盖不足的问题。调研范围限定在 LLZO 基固态电解质，其他石榴石体系（如 LLTO、LGLZO）未充分覆盖。
- **知识提取精度**: LLM 提取结构化数据时可能存在误提取或遗漏，尤其是复杂化学式（如共掺杂体系）和隐含数值（如图表中的数据）的提取准确率受限。轻量抽取模式下仅提取 4 个字段，辅助文献的数据完整性低于核心文献。
- **分析方法局限**: 冲突检测基于数值比较，未考虑材料微观结构（致密度、晶界相、第二相）的定量影响；知识融合采用材料化学式归一化分组，未区分同一材料在不同掺杂浓度下的性能差异。
- **Agent 流水线局限**: 整个调研由 10 个专业化 Agent 协作的 LangGraph 流水线自动完成（含默认开启的构效关系定量分析与可选的 Materials Project 交叉核验扩展节点），未引入人工专家审核环节；检索策略的覆盖度核验依赖 LLM 自评，可能存在盲区。\n"""
        return report

    def _build_knowledge_table(self, fused_table, reference_map: dict[str, dict] | None = None) -> str:
        """Build a material-property knowledge matrix as markdown table.

        Args:
            fused_table: FusedKnowledgeTable with material summaries.
            reference_map: lit_id → {{ref_num, ...}}  mapping for citation lookup.
        """
        if not fused_table or not fused_table.summaries:
            return "（无数据）"

        rows = []
        rows.append(
            "| 材料体系 | 缩写 | 离子电导率 (S/cm) | 电化学窗口 (V) | "
            "合成方法 | 文献数 | 引用文献 |"
        )
        rows.append(
            "|----------|------|-------------------|----------------|"
            "----------|--------|----------|"
        )

        for summary in fused_table.summaries:
            material = summary.material or "unknown"
            alias = summary.alias or "-"

            # Conductivity range — flag implausible values
            if summary.conductivity_range:
                low, high = summary.conductivity_range
                if low == high:
                    cond = f"{low:.2e}"
                else:
                    cond = f"{low:.2e} ~ {high:.2e}"
                # Flag anomalously high conductivity for solid electrolytes (> 0.1 S/cm)
                if high is not None and high > 0.1:
                    cond += " [⚠️ 待核实]"
            elif summary.mean_conductivity:
                cond = f"{summary.mean_conductivity:.2e}"
                if summary.mean_conductivity > 0.1:
                    cond += " [⚠️ 待核实]"
            else:
                cond = "-"

            # Electrochemical window — detect contradictions (max/min > 2x)
            windows = [
                (r.electrochemical_window_V, r.literature_id)
                for r in summary.records
                if r.electrochemical_window_V is not None
            ]
            if len(windows) >= 2:
                win_vals = [w[0] for w in windows]
                wmin, wmax = min(win_vals), max(win_vals)
                if wmax / wmin > 2.0 and wmax > 4.0:
                    # Large discrepancy → mark as disputed
                    win = f"争议: {wmin:.1f}~{wmax:.1f}V"
                else:
                    win = f"{wmin:.1f}~{wmax:.1f}"
            elif windows:
                win = f"{windows[0][0]:.1f}"
            else:
                win = "-"

            # Synthesis methods
            methods = set(
                r.synthesis_method for r in summary.records
                if r.synthesis_method
            )
            method_str = ", ".join(list(methods)[:2]) if methods else "-"

            # ── Build reference column: collect unique literature IDs → ref numbers ──
            ref_str = "-"
            if reference_map:
                lit_ids = set(
                    r.literature_id for r in summary.records
                    if r.literature_id
                )
                ref_nums = []
                for lid in lit_ids:
                    info = reference_map.get(lid)
                    if info:
                        ref_nums.append(f"文献#{info['ref_num']}")
                if ref_nums:
                    ref_str = ", ".join(sorted(ref_nums, key=lambda x: int(x.replace("文献#", ""))))

            rows.append(
                f"| {material} | {alias} | {cond} | {win} | {method_str} | "
                f"{summary.n_papers} | {ref_str} |"
            )

        return "\n".join(rows)

    def _format_conflicts(self, conflicts: list[dict]) -> str:
        """Format conflicts as readable text."""
        if not conflicts:
            return "未检测到显著数据冲突。"

        lines = []
        for i, c in enumerate(conflicts):
            material = c.get("material", "?")
            field = c.get("field", "?")
            ratio = c.get("conflict_ratio", 0)
            causes = c.get("possible_causes", [])
            values = c.get("values", [])

            lines.append(f"**冲突 {i+1}**: {material} 的 {field}")
            for v in values[:3]:
                lines.append(
                    f"  - {v.get('value')} (DOI: {v.get('doi', 'N/A')}, "
                    f"合成方法: {v.get('synthesis', 'N/A')})"
                )
            lines.append(f"  - 冲突比: {ratio:.1%}")
            if causes:
                lines.append(f"  - 可能原因: {'; '.join(causes)}")
            lines.append("")

        return "\n".join(lines)

    def _format_gaps(self, gaps: list, reference_map: dict[str, dict] | None = None) -> str:
        """Format research gaps as readable text (ranked by score).

        When reference_map is provided, translates supporting_literature DOIs into
        numbered references (文献#N).
        """
        if not gaps:
            return "未发现研究缺口。"

        lines = []
        for i, gap in enumerate(gaps):
            # Get gap attributes (handle both ResearchGap objects and dicts)
            if hasattr(gap, "gap_id"):
                gid = gap.gap_id
                desc = gap.description
                evidence = gap.evidence_gap_or_conflict
                novelty = gap.novelty
                hypothesis = gap.falsifiable_hypothesis
                verify = gap.suggested_verification
                status = gap.verification_status
                score = gap.score
                support_refs = getattr(gap, "supporting_literature", [])
            else:
                gid = gap.get("gap_id", f"gap_{i}")
                desc = gap.get("description", "")
                evidence = gap.get("evidence_gap_or_conflict", "")
                novelty = gap.get("novelty", "")
                hypothesis = gap.get("falsifiable_hypothesis", "")
                verify = gap.get("suggested_verification", "")
                status = gap.get("verification_status", "pending")
                score = gap.get("score")
                support_refs = gap.get("supporting_literature", [])

            total = score.total_score if hasattr(score, "total_score") else (
                score.get("total_score", 0) if isinstance(score, dict) else 0
            )

            # Map supporting literature to reference numbers.
            # A-2: match by DOI first, then by doc_id (doc_id 是 A-1 后的主锚，
            # 大量 ref 只有 doc_id 没有 DOI — 只按 DOI 匹配会漏).
            supporting_str = ""
            if reference_map and support_refs:
                ref_nums = []
                for ref in support_refs:
                    if hasattr(ref, "doi"):
                        ref_doi = ref.doi
                        ref_docid = ref.doc_id
                    elif isinstance(ref, dict):
                        ref_doi = ref.get("doi", "")
                        ref_docid = ref.get("doc_id", "")
                    else:
                        ref_doi = str(ref)
                        ref_docid = None
                    for lit_id, info in reference_map.items():
                        if (info.get("doi") and info.get("doi") != "N/A"
                                and info.get("doi") == ref_doi):
                            ref_nums.append(f"文献#{info['ref_num']}")
                            break
                        if (ref_docid and info.get("doc_id")
                                and info.get("doc_id") == ref_docid):
                            ref_nums.append(f"文献#{info['ref_num']}")
                            break
                if ref_nums:
                    supporting_str = f"\n- **支撑文献**: {', '.join(ref_nums[:5])}"

            lines.append(f"### 缺口 {i+1}: {desc[:100]}...")
            lines.append(f"- **评分**: {total:.3f} (验证状态: {status})")
            lines.append(f"- **证据状态**: {evidence}")
            lines.append(f"- **新颖性**: {novelty}")
            lines.append(f"- **[待验证假设]**: {hypothesis}")
            lines.append(f"- **建议验证**: {verify}")
            if supporting_str:
                lines.append(supporting_str)
            lines.append("")

        return "\n".join(lines)

    def _format_references(self, reference_map: dict[str, dict]) -> str:
        """Format reference map as a numbered reference list for the prompt."""
        if not reference_map:
            return "（无可用文献数据）"

        lines = []
        for lit_id, info in sorted(reference_map.items(), key=lambda x: x[1]["ref_num"]):
            ref_num = info["ref_num"]
            authors = info["authors"]
            title = info["title"]
            year = info["year"]
            journal = info["journal"]
            doi = info["doi"]
            keywords = info.get("keywords", "N/A")
            abstract = info.get("abstract", "N/A")
            lines.append(
                f"文献#{ref_num}: \"{title}\" ({authors}, {year}, {journal}, DOI: {doi})"
            )
            if keywords and keywords != "N/A":
                lines.append(f"    关键词: {keywords}")
            if abstract and abstract != "N/A":
                lines.append(f"    摘要: {abstract}")
        return "\n".join(lines)

    # ── A-2: Structured evidence manifest (工程生成，非 LLM 输出) ──

    @staticmethod
    def _anchor_label(ref) -> str:
        """Short human-readable anchor for a supporting ref (doc_id 优先，DOI 兜底)."""
        if ref.doc_id:
            d = ref.doc_id
            return f"doc_id: {d[:16]}…" if len(d) > 16 else f"doc_id: {d}"
        if ref.doi:
            return f"DOI: {ref.doi}"
        return "—"

    @classmethod
    def _ref_verification_status(cls, ref) -> tuple[str, str]:
        """Map a supporting ref's verification state to (emoji_label, raw_status).

        raw_status ∈ {verified, failed, unverified, untraceable}
        """
        if not ref.has_anchor:
            return ("⚠️ 不可追溯", "untraceable")
        score = ref.verification_score
        if score is None:
            return ("⏳ 未核验", "unverified")
        if score > 0.5:
            return ("✅ 已核验", "verified")
        return ("❌ 未命中", "failed")

    def _format_cross_checks(self, checks: list[dict]) -> str:
        """Render the hypothesis cross-check results (Step 10.5) as a markdown table."""
        if not checks:
            return ""
        lines = [
            "## 假设外部/离线交叉核验 (Hypothesis Cross-Check)",
            "",
            "对每个研究缺口的「可证伪假设」进行 Materials Project 在线数据"
            "与 Sci-Base 本地离线语料交叉核验，作为科学性闭环的一部分。",
            "",
            "| 缺口 | 假设中识别的材料 | 核验来源 | 结论 | 说明 |",
            "|------|----------------|---------|------|------|",
        ]
        verdict_label = {
            "corroborated": "已交叉验证",
            "conflict": "发现冲突",
            "no_data": "无可用数据",
            "skipped": "跳过",
        }
        for c in checks:
            src = "Materials Project" if c.get("mp") else ("Sci-Base" if c.get("scibase") else "—")
            lines.append(
                f"| {c.get('gap_id') or '—'} | {c.get('formula') or '—'} | {src} | "
                f"{verdict_label.get(c.get('verdict', 'skipped'), c.get('verdict'))} | {c.get('note', '')} |"
            )
        return "\n".join(lines)

    def _build_evidence_manifest(self, gaps: list) -> str:
        """Build a structured evidence traceability manifest (附录 A).

        每条研究结论 → 支撑文献 → 溯源锚点(doc_id/DOI) → 核验状态 →
        命中原文段落。由工程层从 scored_gaps 结构化数据渲染，不依赖 LLM，
        保证报告结论可逐条回溯到原文。
        """
        if not gaps:
            return ""
        total_refs = sum(len(g.supporting_literature) for g in gaps)
        if total_refs == 0:
            return ""

        lines = [
            "## 附录 A. 证据溯源清单",
            "",
            "> 本清单由系统基于证据核验结果自动生成（非 LLM 输出）。"
            "每条研究结论均可通过溯源锚点（doc_id / DOI）回溯到原文；"
            "「⚠️ 不可追溯」表示该支撑文献无任何溯源锚点，结论可信度存疑。",
            "",
            "| 缺口 | 结论主张 | 支撑文献 | 溯源锚点 | 核验状态 | 分数 | 命中原文段落 |",
            "|------|---------|---------|---------|---------|------|-------------|",
        ]

        for gap in gaps:
            for ref in gap.supporting_literature:
                label, status = self._ref_verification_status(ref)
                title = (ref.title or "N/A").replace("|", "\\|")
                if len(title) > 60:
                    title = title[:57] + "…"
                finding = (ref.finding or "").replace("|", "\\|")
                if len(finding) > 80:
                    finding = finding[:77] + "…"
                score = (
                    f"{ref.verification_score:.2f}"
                    if ref.verification_score is not None else "—"
                )
                passage = (ref.verified_passage or "").replace("|", "\\|").replace("\n", " ")
                if len(passage) > 90:
                    passage = passage[:87] + "…"
                if not passage:
                    passage = "—"
                lines.append(
                    f"| {gap.gap_id} | {finding} | {title} | "
                    f"{self._anchor_label(ref)} | {label} | {score} | {passage} |"
                )

        # Summary line
        n_untraceable = sum(
            1 for g in gaps for r in g.supporting_literature if not r.has_anchor
        )
        n_verified = sum(
            1 for g in gaps for r in g.supporting_literature
            if r.has_anchor and r.verification_score is not None and r.verification_score > 0.5
        )
        summary = (
            f"\n**溯源统计**: 共 {total_refs} 条支撑文献，"
            f"{n_verified} 条已回溯到原文，"
            f"{n_untraceable} 条不可追溯。"
        )
        if n_untraceable:
            summary += " 不可追溯条目需人工核实或补充溯源锚点。"
        lines.append(summary)
        return "\n".join(lines)

    async def _generate_submission(self, survey_report: str) -> str:
        """Generate competition submission document from survey report."""
        if not COMPETITION_TEMPLATE:
            self.log("Competition template not found, returning survey report", "yellow")
            return survey_report

        pipeline_summary = self._build_pipeline_summary()
        tools_list = self._build_tools_list()
        agent_architecture = self._build_agent_architecture()

        prompt = COMPETITION_TEMPLATE.format(
            project_name=self.config.get("project_name", "材料科学文献驱动的科学发现智能体"),
            agent_architecture=agent_architecture,
            technical_pipeline=pipeline_summary,
            tools_list=tools_list,
            pipeline_summary=pipeline_summary,
            verification_metrics="知识提取准确率、冲突检测召回率、缺口识别新颖性评分、文献溯源完整率",
            survey_report=survey_report,
        )

        try:
            result = await self.llm.complete(
                "你是一位 AI for Research 算法赛参赛方案撰写专家。"
                "请严格按照模板生成完整的六章参赛方案文档。",
                prompt,
                max_tokens=16384,
                stage=self.name,
                # Submission is a template-driven reformatting of the survey
                # report — deep reasoning is unnecessary and the default
                # MiniMax <think> chain made the call exceed the 120 s timeout
                # (2026-08-14: "Request timed out." after 3 x 120 s SDK
                # retries, ~6 min). Disabling thinking + a 600 s per-request
                # timeout lets the full budget go to the document itself.
                timeout=600,
                disable_thinking=True,
            )
            return result
        except Exception as e:
            self.log(f"Submission generation failed: {e}, returning survey report", "yellow")
            return survey_report

    @staticmethod
    def _build_agent_architecture() -> str:
        """Build agent architecture description."""
        return """MatResearcher 采用基于 LangGraph 的多智能体协作架构，由 10 个专业化 Agent 协同完成从研究问题到调研报告的端到端自动化流程：

**Agent 角色矩阵（按流水线顺序）：**
1. **任务规划 Agent** — 科学问题理解、任务拆解、检索策略生成
2. **文献检索 Agent** — Sciverse API 多轮语义检索 + 查询扩展
3. **LLM 三分类预筛 Agent** — 基于标题+摘要的 relevant/partial/irrelevant 三分类预筛选
4. **文献筛选 Agent** — 去重 + BGE-Reranker 精排 + 阈值过滤
5. **PDF 解析 Agent** — MinerU 全文结构化解析（正文段落 + 表格 + 图注入库）+ Sciverse 全文 fallback 链
6. **知识抽取 Agent** — 分层抽取（核心文献 9 字段深度 / 辅助文献 4 字段轻量）
7. **跨文献融合 Agent** — 实体规范化、单位标准化、冲突检测
8. **Gap 识别 Agent** — 缺失检测、矛盾归纳、Research Gap 生成与评分
9. **证据核验 Agent** — 原文回溯、引用验证、事实核查图级回退重写闭环
10. **报告生成 Agent** — 三类标记的结构化文献调研报告（文献事实/跨文献推论/待验证假设）

**关键技术决策：**
- 检索覆盖度核验回退机制：自动重试 + 策略精炼（max 2 轮）
- PDF 解析 fallback 链：MinerU → Sciverse /content → chunk → abstract
- 数据质量核验：异常数据分离存储，不参与融合
- 分层抽取策略：Reranker Top-8 深度抽取，剩余轻量抽取
- 单位换算规则引擎（R1 单位规范化）+ 构效关系定量分析（数值重算、掺杂比例计算）"""

    @staticmethod
    def _build_pipeline_summary() -> str:
        """Build technical pipeline description."""
        return """12 步核心技术路线（+ 2 个可选扩展节点）：

**阶段一：任务理解与检索（步骤 1-3）**
1. 研究问题理解与分解（任务规划 Agent）
2. 多轮文献检索（Sciverse agentic-search + meta-search + 语义扩展）
3. 检索覆盖度核验（条件回退机制，max 2 retry）

**阶段二：文献筛选与解析（步骤 4-6）**
4. LLM 三分类预筛（relevant/partial/irrelevant）
5. 文献去重 + BGE-Reranker 精排（Top-25 保留）
6. PDF 解析（MinerU 全文结构化，含表格/图注入库）

**阶段三：知识提取与融合（步骤 7-8.5）**
7. 分层知识抽取（核心深度 + 辅助轻量）
7a. 数据质量核验（异常分离）
8. 实体归一化与单位标准化 + 跨文献知识融合 + 冲突检测
8.5. 构效关系定量分析（默认开启：元素解析 + Pearson/Spearman 相关性 + 掺杂比例）

**阶段四：缺口发现、核验与报告（步骤 9-12）**
9. Research Gap 生成（7 元素结构）与评分排序
10. 证据溯源验证（Sciverse locate_passage 回溯全文）
10.5. 假设交叉核验（可选：Materials Project 在线 + Sci-Base 离线兜底，默认关闭）
11. 结构化报告生成（三类标记）
12. 最终事实核查（图级回退重写闭环，max 2 轮）"""

    @staticmethod
    def _build_tools_list() -> str:
        """Build tools and dependencies list."""
        return """- **Sciverse API**：语义检索（agentic-search）+ 元数据检索（meta-search）+ 全文定位（locate_passage）
- **BGE-Reranker**：BAAI/bge-reranker-v2-m3 跨编码器精排模型
- **MinerU**：深度 PDF 解析引擎，支持表格和图片提取
- **LangGraph**：有向图工作流编排，12 核心节点 + 可选扩展节点，2 个条件分支
- **LLM**：通义千问 qwen-3.7-plus（任务规划/知识提取/报告生成）
- **Pydantic**：结构化数据模型
- **SQLite**：知识记录持久化存储"""

    def _generate_template_based(self, **kwargs) -> str:
        """Fallback: generate a basic template-based report."""
        return f"""# 材料科学文献调研报告

## 1. 研究问题概述

**研究问题**: {kwargs.get('question', 'N/A')}

**调研范围**: 固态电池材料体系的离子电导率、电化学窗口、合成方法相关研究。

## 2. 检索方法与文献覆盖

- 检索候选文献: {kwargs.get('candidate_count', 0)} 篇
- 筛选后保留: {kwargs.get('filtered_count', 0)} 篇
- 成功解析: {kwargs.get('parsed_count', 0)} 篇

## 3. 知识体系图

**[文献事实]** 以下材料-性能数据均提取自已解析文献的原文，每条数据可经附录 A 溯源锚点回溯：

{kwargs.get('knowledge_table', '（无数据）')}

## 4. 数据一致性分析

**[跨文献推论]** 通过对 {kwargs.get('material_count', 0)} 种材料的文献数据进行交叉对比分析：

{kwargs.get('conflicts_detail', '未检测到显著数据冲突。')}

## 5. 研究缺口发现

共发现 {kwargs.get('gap_count', 0)} 个研究缺口：

{kwargs.get('gaps_detail', '未发现研究缺口。')}

**[待验证假设]** 上述研究缺口中包含的可证伪假设，建议通过以下方式验证：
1. 控制变量实验设计
2. 标准化表征协议
3. 独立复现实验

## 6. 参考文献清单

{kwargs.get('references', '（无可用文献数据）')}

## 7. 方法局限性

- 检索覆盖: 依赖Sciverse数据库，可能存在未收录的高质量文献
- 知识提取: LLM提取精度受模型能力限制，复杂数据可能存在误提取
- 分析方法: 冲突检测基于数值比较，未考虑材料微观结构的定量影响
- 研究缺口: 基于已有文献的缺失分析，可能遗漏全新的研究方向

---
*报告由 MatResearcher 文献调研Agent自动生成*
*生成时间: {kwargs.get('generate_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}*
"""
