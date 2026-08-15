"""Gap Identification Agent (Step 9) + Scoring.

Step 9: Generate Research Gaps from conflicts and missing connections,
then score by novelty, operability, evidence completeness.

Each Research Gap follows the 7-element structure:
1. 问题描述
2. 支撑文献
3. 证据缺失或冲突
4. 新颖性
5. 可操作性
6. 可证伪假设
7. 建议验证方法
"""
from __future__ import annotations

import re
from typing import Optional

from ..state import WorkflowState
from .base import BaseAgent
from ..models.gap import ResearchGap, GapSupportingRef, GapScore, ConflictItem, MissingItem


GAP_GENERATION_PROMPT = """You are a materials science researcher specializing in solid-state batteries.
Based on the following conflicts and missing data connections found in the literature, generate Research Gaps.

For each gap, provide a JSON object with these fields:

1. description: A clear statement of the research problem/question
2. supporting_literature: [{{"doi": "...", "title": "...", "finding": "..."}}]
3. evidence_gap_or_conflict: What evidence is missing or contradictory
4. novelty: Why this gap represents new knowledge (vs what is already known)
5. operability: How feasible it is to experimentally investigate
6. falsifiable_hypothesis: A testable hypothesis (must be falsifiable)
7. suggested_verification: Proposed experimental or computational methods to verify

Gap Scoring Criteria:
- novelty_score (0-1): 1.0 = completely unexplored area; 0.0 = well-established topic
- operability_score (0-1): 1.0 = can be tested with standard equipment; 0.0 = requires unavailable facilities
- evidence_completeness (0-1): 1.0 = strong supporting evidence; 0.0 = speculation without evidence

Return ONLY a valid JSON array of gap objects, with scores included.

Conflicts:
{conflicts}

Missing Data Connections:
{missing_items}

Materials in scope:
{materials}

Available literature (cite ONLY these in supporting_literature — never invent titles or DOIs):
{filtered_literature}
"""


class GapIdentificationAgent(BaseAgent):
    name = "gap_identification"
    role = "研究缺口识别 Agent"

    _SYSTEM_GAP_GENERATION = (
        "你是一位固态电池材料研究专家，专精于从文献数据中识别研究缺口（Research Gap）。\n\n"
        "## 领域知识\n"
        "- 固态电解质：氧化物（LLZO/LATP/LLTO）、硫化物（LGPS/Li6PS5Cl/Li3PS4）、"
        "聚合物（PEO-LiTFSI）、卤化物（Li3YCl6/Li2ZrCl6）\n"
        "- 关键性能：离子电导率（S/cm）、电化学窗口（V）、对锂稳定性、"
        "空气稳定性、机械强度\n"
        "- 制备工艺：固相反应、溶胶凝胶、球磨、SPS烧结、ALD/CVD涂层\n"
        "- 表征方法：EIS、XRD、SEM/TEM、XPS、NMR、拉曼\n\n"
        "## Gap 识别方法论\n"
        "1. **数据冲突型 Gap**：同一材料体系、同一性能参数在不同文献中差异显著"
        "（>50%），可能源于合成工艺、测试条件、表征方法不同\n"
        "2. **数据缺失型 Gap**：某材料体系缺少特定性能数据（如卤化物电解质的高温电导率、"
        "硫化物电解质的电化学窗口实测数据）\n"
        "3. **方法论型 Gap**：现有研究以实验为主，缺少 DFT/MD 等计算模拟的交叉验证\n"
        "4. **应用型 Gap**：材料在实验室表现优异但缺乏全电池/实际工况下的性能数据\n\n"
        "生成 Gap 时必须包含可证伪假设（falsifiable_hypothesis），"
        "评分时考虑新颖性（novelty_score）、可操作性（operability_score）、"
        "证据完整度（evidence_completeness）。\n\n"
        "## 引用纪律（强制，违反将导致输出被丢弃）\n"
        "- supporting_literature 中每一条必须真实存在于下方「Available literature」列表中\n"
        "- DOI 必须与列表中的 doi 字段逐字一致，禁止编造、改写或拼接 DOI\n"
        "- 如果列表中没有与论点直接相关的文献，宁可不列该条，也不得虚构标题或 DOI\n"
        "- finding 只能概括该文献摘要/关键数据中实际存在的内容，禁止编造数据\n"
        "- finding 必须聚焦该文献「实际报告了什么」（如性能数值、合成方法、结论），"
        "禁止描述该文献「缺少/未研究」什么（如 'lacks investigation into X'、"
        "'without addressing Y'）—— 缺失点只属于 evidence_gap_or_conflict 字段"
    )

    def __init__(self, llm=None, config=None, log_dir=None):
        super().__init__(llm, config, log_dir=log_dir)
        self.scoring_weights = self.config.get("scoring_weights", {
            "novelty": 0.4,
            "operability": 0.3,
            "evidence_completeness": 0.3,
        })
        self.min_supporting_papers = self.config.get("min_supporting_papers", 2)

    async def run(self, state: WorkflowState) -> dict:
        conflicts = state.get("conflicts", [])
        missing_items = state.get("missing_items", [])
        fused_table = state.get("fused_table")

        # Collect material names
        materials = []
        if fused_table and hasattr(fused_table, "summaries"):
            for s in fused_table.summaries:
                name = s.material
                if s.alias:
                    name += f" ({s.alias})"
                materials.append(name)

        self.log(f"Generating Research Gaps from {len(conflicts)} conflicts "
                 f"and {len(missing_items)} missing items")

        filtered_lit = state.get("filtered_literature", [])

        # Step 12: Generate gaps
        if self.llm:
            gaps = await self._generate_gaps_with_llm(
                conflicts, missing_items, materials, filtered_lit
            )
        else:
            gaps = self._generate_gaps_rule_based(conflicts, missing_items)

        self.log(f"Generated {len(gaps)} Research Gaps")

        # Step 14: Score gaps
        scored = self._score_gaps(gaps)

        # Sort by total score descending
        scored.sort(key=lambda g: g.score.total_score if g.score else 0, reverse=True)
        for i, g in enumerate(scored):
            if g.score:
                g.score.rank = i + 1

        self.log(f"Scored {len(scored)} gaps, top score: "
                 f"{scored[0].score.total_score:.3f}" if scored else "N/A")

        # ── Step 12a: Enrich gap DOIs from filtered_literature ──
        # LLM-generated gaps often have hallucinated or missing DOIs.
        # Backfill from the actual filtered_literature metadata so that
        # Step 13 evidence verification has real DOIs to work with.
        # Refs that cannot be matched to a real paper are DOWNGRADED to
        # doi=None so the verifier SKIPs them explicitly instead of
        # querying a fabricated DOI.
        if filtered_lit:
            self._enrich_gap_dois(scored, filtered_lit)

        total_refs = sum(len(g.supporting_literature) for g in scored)
        total_with_doi = sum(
            1 for g in scored for r in g.supporting_literature if r.doi
        )
        total_with_docid = sum(
            1 for g in scored for r in g.supporting_literature if r.doc_id
        )
        total_with_anchor = sum(
            1 for g in scored for r in g.supporting_literature if r.has_anchor
        )
        self.log(
            f"Anchor enrichment: {total_with_anchor}/{total_refs} supporting refs "
            f"traceable (doc_id={total_with_docid}, doi={total_with_doi})"
        )

        return {
            "gaps": gaps,
            "scored_gaps": scored,
        }

    async def _generate_gaps_with_llm(
        self,
        conflicts: list[dict],
        missing: list[dict],
        materials: list[str],
        filtered_literature: list | None = None,
    ) -> list[ResearchGap]:
        """Use LLM to generate Research Gaps (Step 12)."""
        try:
            conf_text = str(conflicts) if conflicts else "None found"
            miss_text = str(missing) if missing else "None found"
            mat_text = ", ".join(materials) if materials else "various materials"

            # Whitelist of real, filtered literature the LLM may cite.
            # This is the ONLY source for supporting_literature entries.
            # When a paper has no DOI (common for Sciverse doc_id-only results),
            # instruct the LLM to cite it by literature id — the engineering
            # layer resolves "lit_xxxx" back to a real doc_id anchor via
            # _enrich_gap_dois Strategy 0. Without this, the LLM is stuck:
            # the citation discipline forbids inventing DOIs, so with all-DOI-None
            # lists it returns EMPTY supporting_literature and the whole gap
            # becomes untraceable (evidence verification 0/0, unverified).
            lit_entries = []
            for lit in (filtered_literature or []):
                meta = lit.metadata
                if meta.doi:
                    ref_hint = f"DOI: {meta.doi}"
                else:
                    ref_hint = (
                        f"lit_id: {lit.id} (cite as doi=\"{lit.id}\", "
                        f"e.g. {{\"doi\": \"{lit.id}\", ...}})"
                    )
                lit_entries.append(
                    f"- [{lit.id}] {meta.title} | {ref_hint}"
                )
            if lit_entries:
                lit_text = "\n".join(lit_entries)
            else:
                lit_text = (
                    "(no literature list provided — if you cannot name a real "
                    "DOI, leave doi empty; never invent one)"
                )

            # Use .replace() instead of .format() because conflict text
            # may contain curly braces from chemical formulas (e.g. Li{7}).
            prompt_text = (
                GAP_GENERATION_PROMPT
                .replace("{conflicts}", conf_text)
                .replace("{missing_items}", miss_text)
                .replace("{materials}", mat_text)
                .replace("{filtered_literature}", lit_text)
            )
            result = await self.llm.complete_json(
                self._SYSTEM_GAP_GENERATION,
                prompt_text,
                max_tokens=16384,
                stage=self.name,
                # Verified 2026-08-14: gap generation with thinking took
                # ~110 s — already at the 120 s default limit. Give it room
                # so a longer think chain doesn't hit the timeout.
                timeout=300,
            )

            gaps = []
            # Normalize LLM JSON: accept either a bare list of gap objects or a
            # wrapped dict like {"gaps": [...]} / {"research_gaps": [...]} —
            # LLMs occasionally wrap the array under a key (observed 2026-08-14:
            # 4.5-min call returned a wrapper, old code dropped it silently).
            result_list = result
            if isinstance(result, dict):
                for key in ("gaps", "research_gaps", "research_gap", "gap_list"):
                    if isinstance(result.get(key), list):
                        result_list = result[key]
                        break
                else:
                    # Single wrapped object: {"gap_000": {...}} etc.
                    result_list = [v for v in result.values()
                                   if isinstance(v, dict)]
            if isinstance(result_list, list):
                for i, item in enumerate(result_list):
                    if isinstance(item, dict):
                        gap = self._parse_gap_from_dict(i, item)
                        if gap:
                            gaps.append(gap)
            return gaps
        except Exception as e:
            self.log(f"LLM gap generation failed: {e}, using rule-based fallback", "yellow")
            return self._generate_gaps_rule_based(conflicts, missing)

    def _generate_gaps_rule_based(
        self, conflicts: list[dict], missing: list[dict]
    ) -> list[ResearchGap]:
        """Rule-based gap generation (fallback without LLM)."""
        gaps = []
        idx = 0

        # Generate gaps from conflicts
        for conflict in conflicts:
            material = conflict.get("material", "unknown")
            field = conflict.get("field", "property")
            causes = conflict.get("possible_causes", [])
            cause_text = "; ".join(causes) if causes else "原因待查"

            gap = ResearchGap(
                gap_id=f"gap_{idx:03d}",
                description=(
                    f"材料 {material} 的 {field} 性能数据在不同文献中存在显著差异"
                    f"(冲突比 {conflict.get('conflict_ratio', 0):.1%})，"
                    f"导致无法确定该材料的最佳性能参数范围"
                ),
                supporting_literature=[
                    GapSupportingRef(
                        doi=v.get("doi"),
                        title=f"Record from {v.get('doi', 'unknown')}",
                        finding=f"Reported {field}={v.get('value')} "
                                f"(method: {v.get('synthesis', 'unknown')})",
                    )
                    for v in conflict.get("values", [])[:3]
                ],
                evidence_gap_or_conflict=(
                    f"文献间数据冲突 ({field})，差异超过{conflict.get('conflict_ratio', 0):.0%}。"
                    f"可能原因：{cause_text}"
                ),
                novelty=(
                    f"现有文献未系统研究{material}的{field}性能与制备参数的关系，"
                    f"存在知识空白。通过系统变量控制可明确性能差异来源"
                ),
                operability=(
                    f"可通过控制单一变量实验（合成方法、温度、前驱体等）进行验证，"
                    f"实验条件在常规材料合成设施可达范围内"
                ),
                falsifiable_hypothesis=(
                    f"若统一合成条件和测试温度，{material}的{field}值差异"
                    f"将收敛至测量误差范围之内（±20%）"
                ),
                suggested_verification=(
                    f"1) 设计控制变量实验：固定{field}测试条件；"
                    f"2) 标准化表征协议；"
                    f"3) 进行独立复现实验"
                ),
            )
            idx += 1
            gaps.append(gap)

        # Generate gaps from missing items
        for missing_item in missing:
            material = missing_item.get("material", "unknown")
            field = missing_item.get("missing_field", "data")
            desc = missing_item.get("description", "")

            gap = ResearchGap(
                gap_id=f"gap_{idx:03d}",
                description=(
                    f"材料 {material} 缺乏 {field} 相关数据。{desc}"
                    if desc else f"材料 {material} 缺乏 {field} 数据"
                ),
                supporting_literature=[],
                evidence_gap_or_conflict=f"现有文献数据({missing_item.get('existing_data_points', 0)}条)缺少 {field} 信息",
                novelty=(
                    f"填补{material}的{field}数据空白将为理解其在实际工况下的性能"
                    f"提供新的实验依据"
                ),
                operability=(
                    f"可通过现有实验手段获取{field}数据，"
                    f"无需特殊设备或极端条件"
                ),
                falsifiable_hypothesis=(
                    f"若补充{material}的{field}数据，预测其性能将显著不同于室温条件下的已知值"
                ),
                suggested_verification=(
                    f"1) 设计{field}测试实验；"
                    f"2) 对比已有文献数据建立完整性能画像；"
                    f"3) 发表系统数据集填补空白"
                ),
            )
            idx += 1
            gaps.append(gap)

        return gaps

    def _score_gaps(self, gaps: list[ResearchGap]) -> list[ResearchGap]:
        """Step 14: Score each Research Gap.

        If the gap already has an LLM-generated score (from _parse_gap_from_dict),
        preserve it. Otherwise, fall back to rule-based scoring.

        Scoring dimensions:
        - Novelty: more unique conflicts/missing → higher
        - Operability: simpler verification → higher
        - Evidence completeness: more supporting refs → higher
        """
        for gap in gaps:
            if gap.score is not None:
                # LLM already scored this gap — preserve and skip rules
                continue

            # Rule-based fallback scoring
            novelty = min(1.0, 0.5 + 0.1 * len(gap.supporting_literature))

            operability = 0.8
            if "特殊设备" in gap.suggested_verification or "极端条件" in gap.suggested_verification:
                operability = 0.4
            elif "标准化" in gap.suggested_verification:
                operability = 0.9

            evidence = min(1.0, 0.3 + 0.15 * len(gap.supporting_literature))

            score = GapScore(
                novelty_score=round(novelty, 2),
                operability_score=round(operability, 2),
                evidence_completeness=round(evidence, 2),
            )
            score.compute_total(self.scoring_weights)
            gap.score = score

        return gaps

    @staticmethod
    def _unwrap_gap_item(item: dict) -> dict:
        """Unwrap a nested LLM gap object.

        LLMs occasionally wrap the gap object one level deeper than the schema,
        e.g. {"gap_000": {...}}, {"gap": {...}}, {"research_gap": {...}}.
        Detect: if the top level has NONE of the expected fields but one of its
        values is a dict that DOES, unwrap that value (recursively).
        """
        expected = (
            "description", "supporting_literature", "evidence_gap_or_conflict",
            "novelty", "operability", "falsifiable_hypothesis",
            "suggested_verification", "novelty_score", "operability_score",
            "evidence_completeness",
        )
        # Walk down while the current level is a pure wrapper
        current = item
        seen = 0
        while seen < 4:  # guard against pathological nesting
            if any(k in current for k in expected):
                return current
            inner = None
            for v in current.values():
                if isinstance(v, dict):
                    inner = v
                    break
            if inner is None:
                return current
            current = inner
            seen += 1
        return current

    def _parse_gap_from_dict(self, idx: int, item: dict) -> ResearchGap | None:
        """Parse LLM output into a ResearchGap object.

        Tries to parse all fields with per-field fallbacks so that a single
        malformed field won't silently discard the entire Gap.
        """
        # Unwrap nested wrapper objects first (e.g. {"gap_000": {...}}).
        item = self._unwrap_gap_item(item)
        if not item:
            return None

        # ── supporting_literature (with per-ref fallback) ──
        refs = []
        raw_refs = item.get("supporting_literature", [])

        def _unwrap_ref(ref: dict) -> dict:
            """Unwrap {"ref1": {...}} / {"ref": {...}} style wrappers."""
            expected = ("doi", "title", "finding")
            current = ref
            depth = 0
            while depth < 3:
                if any(k in current for k in expected):
                    return current
                inner = next((v for v in current.values()
                              if isinstance(v, dict)), None)
                if inner is None:
                    return current
                current = inner
                depth += 1
            return current

        if isinstance(raw_refs, list):
            for ref in raw_refs:
                try:
                    if isinstance(ref, dict):
                        ref = _unwrap_ref(ref)
                        refs.append(GapSupportingRef(
                            doi=ref.get("doi"),
                            title=ref.get("title", "Unknown"),
                            finding=ref.get("finding", ""),
                        ))
                except Exception:
                    pass  # skip malformed individual ref
        elif isinstance(raw_refs, dict):
            # Common LLM mistake: {"ref1": {...}} instead of [{...}]
            self.log(f"gap_{idx:03d}: supporting_literature is a dict, not a list — trying to unwrap", "yellow")
            try:
                raw_refs = _unwrap_ref(raw_refs)
                refs.append(GapSupportingRef(
                    doi=raw_refs.get("doi"),
                    title=raw_refs.get("title", "Unknown"),
                    finding=raw_refs.get("finding", ""),
                ))
            except Exception:
                pass

        if not refs and raw_refs:
            self.log(f"gap_{idx:03d}: failed to parse any supporting_literature refs", "yellow")

        # ── scores (per-field try/except) ──
        score = None
        try:
            if any(k in item for k in ("novelty_score", "operability_score", "evidence_completeness")):
                score = GapScore(
                    novelty_score=self._safe_float(item, "novelty_score", 0.5),
                    operability_score=self._safe_float(item, "operability_score", 0.5),
                    evidence_completeness=self._safe_float(item, "evidence_completeness", 0.5),
                )
                score.compute_total(self.scoring_weights)
        except Exception as e:
            self.log(f"gap_{idx:03d}: failed to parse LLM scores ({e}), will use rule-based fallback", "yellow")

        # ── build gap with field-level defaults ──
        try:
            return ResearchGap(
                gap_id=f"gap_{idx:03d}",
                description=str(item.get("description", "")),
                supporting_literature=refs,
                evidence_gap_or_conflict=str(item.get("evidence_gap_or_conflict", "")),
                novelty=str(item.get("novelty", "")),
                operability=str(item.get("operability", "")),
                falsifiable_hypothesis=str(item.get("falsifiable_hypothesis", "")),
                suggested_verification=str(item.get("suggested_verification", "")),
                score=score,
            )
        except Exception as e:
            self.log(f"gap_{idx:03d}: failed to construct ResearchGap: {e}", "red")
            return None

    @staticmethod
    def _safe_float(item: dict, key: str, default: float = 0.5) -> float:
        """Extract a float value, handling string representations."""
        try:
            return float(item.get(key, default))
        except (TypeError, ValueError):
            return default

    def _enrich_gap_dois(
        self, gaps: list[ResearchGap], filtered_literature: list
    ) -> list[ResearchGap]:
        """Backfill full evidence objects into gap supporting_literature.

        LLM-generated gaps may have hallucinated or empty DOIs, and LLMs must
        NEVER output doc_id directly (they would fabricate it). This method
        matches each ref to a real Literature from filtered_literature and
        backfills the complete traceability anchor:

            doi / doc_id / year / journal / authors / title

        Matching strategies (in order):
        0. literature_id (rule-based gaps use "lit_xxxx" as pseudo-DOI)
        1. DOI exact match
        2. title substring match
        3. material-name match from the gap description
        """
        # Build lookup maps
        doi_map: dict[str, object] = {}
        title_map: dict[str, object] = {}
        lit_id_map: dict[str, object] = {}  # literature_id → lit

        for lit in filtered_literature:
            meta = lit.metadata
            # ID lookup (rule-based gaps use literature_id as pseudo-DOI)
            if lit.id:
                lit_id_map[lit.id] = lit
            # DOI lookup (normalized)
            if meta.doi:
                doi_map[meta.doi.lower().strip()] = lit
            # Title lookup (normalized: lowercase, strip punctuation)
            if meta.title:
                key = re.sub(r"[^\w\s]", "", meta.title.lower()).strip()
                title_map[key] = lit
                # Also index shorter tokens (first 8 words of title)
                short = " ".join(key.split()[:8])
                if short not in title_map:
                    title_map[short] = lit

        enriched_count = 0
        degraded_count = 0
        for gap in gaps:
            for ref in gap.supporting_literature:
                # Skip if already has a valid-looking DOI
                if ref.doi and len(ref.doi) > 5 and "/" in ref.doi:
                    # Check if this DOI exists in our filtered literature
                    if ref.doi.lower().strip() in doi_map:
                        continue  # Already valid
                    # Otherwise it might be hallucinated — try to fix

                matched = None

                # Strategy 0: Resolve literature_id → real DOI
                # (rule-based gaps use "lit_00123" as pseudo-DOI)
                if not matched and ref.doi and ref.doi.startswith("lit_"):
                    if ref.doi in lit_id_map:
                        matched = lit_id_map[ref.doi]

                # Strategy 1: Try DOI from ref (even if partial)
                if ref.doi and len(ref.doi) > 3:
                    norm_doi = ref.doi.lower().strip()
                    if norm_doi in doi_map:
                        matched = doi_map[norm_doi]

                # Strategy 2: Try title substring matching
                if not matched and ref.title and ref.title != "Unknown":
                    ref_title = re.sub(r"[^\w\s]", "", ref.title.lower()).strip()
                    # Exact match
                    if ref_title in title_map:
                        matched = title_map[ref_title]
                    else:
                        # Substring match: check if ref_title is contained in any key
                        for key, lit in title_map.items():
                            if len(ref_title) > 20 and ref_title[:40] in key:
                                matched = lit
                                break
                            if len(key) > 20 and key[:40] in ref_title:
                                matched = lit
                                break

                # Strategy 3: Match by material name from gap description (strict).
                # Requires BOTH the lit title AND the ref title to contain the
                # material — otherwise a fabricated ref title (e.g. "novel study
                # of LLZO") could be wrongly bound to a real paper, making a
                # hallucinated citation look verified. Title-less refs are exempt
                # (only material name is available to match on).
                if not matched:
                    material = self._extract_material_from_desc(gap.description)
                    if material:
                        m_low = material.lower()
                        ref_title_low = (ref.title or "").lower()
                        ref_has_title = bool(ref.title) and ref.title != "Unknown"
                        for lit in filtered_literature:
                            lit_title = (lit.metadata.title or "").lower()
                            if m_low in lit_title and (
                                not ref_has_title or m_low in ref_title_low
                            ):
                                matched = lit
                                break

                if matched:
                    ref.doi = matched.metadata.doi
                    ref.doc_id = matched.metadata.doc_id
                    ref.year = matched.metadata.year
                    ref.journal = matched.metadata.journal
                    ref.authors = matched.metadata.authors or []
                    if not ref.title or ref.title == "Unknown":
                        ref.title = matched.metadata.title
                    enriched_count += 1
                elif ref.doi and ref.doi.lower().strip() not in doi_map:
                    # Downgrade: cannot be matched to a real paper in our
                    # filtered literature → strip the (likely fabricated) DOI
                    # so Step 13 SKIPs it explicitly instead of querying garbage.
                    self.log(
                        f"  DEGRADE: ref '{ref.title[:60]}...' DOI '{ref.doi}' "
                        f"not found in filtered_literature — doi set to None",
                        "yellow",
                    )
                    ref.doi = None
                    degraded_count += 1

        if enriched_count or degraded_count:
            self.log(
                f"  Enriched {enriched_count} supporting ref(s) with real DOIs, "
                f"downgraded {degraded_count} unmatched ref(s)"
            )
        else:
            self.log(
                f"  WARNING: Could not enrich any DOIs — "
                f"evidence verification will likely fail for all gaps",
                "yellow",
            )

        return gaps

    @staticmethod
    def _extract_material_from_desc(description: str) -> str | None:
        """Extract a material formula/name from a gap description."""
        # Try explicit chemical formulas: Li7La3Zr2O12, Li6PS5Cl, etc.
        m = re.search(r"(Li[\d.]*[A-Z][a-z]?[\d.]*)+", description)
        if m:
            return m.group()
        # Try common abbreviations
        for abbr in ["LLZO", "LLZTO", "LGPS", "LATP", "LLTO", "NASICON", "PEO"]:
            if abbr.lower() in description.lower():
                return abbr
        return None
