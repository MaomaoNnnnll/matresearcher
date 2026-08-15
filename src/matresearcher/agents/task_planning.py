"""Task Planning Agent (Steps 1).

Responsibilities:
- Phase 1: Query reformulation — rewrite original query into semantically equivalent queries
- Phase 2: Keyword extraction from all queries (original + reformulated)
- Phase 3: Strategy generation — decompose into subtasks, build search strategy

Two-step LLM pipeline:
  1. query_reformulation.txt → reformulated_queries + extracted_keywords
  2. task_planning.txt     → understood_question + subtasks + search_strategy
"""
from __future__ import annotations
import re
from pathlib import Path
from ..state import WorkflowState
from .base import BaseAgent, PROMPTS_DIR


# English stop words to filter out during keyword extraction
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "and", "of", "in", "on", "for", "to", "a", "an", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "can", "shall", "with", "by", "at", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "about", "up", "out", "over", "its", "it", "or", "but", "this",
    "that", "these", "those", "which", "who", "whom", "what", "when",
    "where", "how", "why", "if", "also", "however", "therefore",
    "advances", "recent", "progress", "research", "review", "study",
    "investigation", "analysis", "development", "approach", "method",
    "novel", "new", "based", "using", "toward", "towards",
})


class TaskPlanningAgent(BaseAgent):
    name = "task_planning"
    role = "任务规划 Agent"
    prompt_file = "task_planning.txt"
    reformulation_prompt_file = "query_reformulation.txt"

    # ═══════════════════════════════════════════════════════════════
    # System prompts with solid-state battery materials domain knowledge
    # ═══════════════════════════════════════════════════════════════

    _DOMAIN_KNOWLEDGE = (
        "固态电池材料研究覆盖以下核心体系：\n"
        "- 固态电解质：氧化物（LLZO/Garnet、LLTO/Perovskite、LATP/NASICON、LAGP）、"
        "硫化物（LGPS/thio-LISICON、Argyrodite/Li6PS5X、Li2S-P2S5玻璃陶瓷）、"
        "聚合物（PEO-LiTFSI、PVDF基）、卤化物（Li3YCl6、Li3InCl6）\n"
        "- 电极材料：锂金属负极、Si/C负极、NCM/NCA/LFP/LCO正极、硫正极（Li-S体系）\n"
        "- 界面层/缓冲层：LiNbO3、Li3PO4、Al2O3、ALD涂层\n"
        "- 掺杂元素：Ta、Al、Ga（LLZO掺杂）、Nb、W、Mo（稳定化掺杂）\n"
        "- 制备工艺：固相烧结、溶胶凝胶、SPS、ALD/CVD涂层\n"
        "- 表征技术：EIS、XRD、SEM/TEM、固态核磁（ssNMR）、DFT/MD计算\n"
        "- 关键性能：离子电导率、电子电导率、界面阻抗、电化学窗口、临界电流密度（CCD）\n"
    )

    _SYSTEM_REFORMULATION = (
        "你是固态电池材料领域的文献检索专家，专精于学术查询的语义改写与关键词抽取。"
        "你必须严格按照用户指令输出，只返回合法JSON，不输出任何解释、问候语或自我介绍。\n\n"
        "## 领域知识\n" + _DOMAIN_KNOWLEDGE + "\n"
        "## 改写策略\n"
        "1. 缩写与全称互换：LLZO ↔ Li7La3Zr2O12，LGPS ↔ Li10GeP2S12\n"
        "2. 命名体系变换：Li garnet electrolyte ↔ cubic Li7La3Zr2O12 ↔ Li-stuffed garnet\n"
        "3. 性能描述变换：ionic conductivity ↔ Li+ transport ↔ Li-ion diffusivity ↔ charge transfer\n"
        "4. 工艺术语变换：sintering ↔ densification ↔ hot-pressing ↔ spark plasma sintering\n"
        "5. 表征方法变换：EIS ↔ electrochemical impedance ↔ AC impedance spectroscopy\n"
        "6. 每条改写查询应为语义完整的英文句子或学术短语，适合Sciverse语义检索\n\n"
        "## 关键词标准\n"
        "- 优先提取：材料化学式/名称、性能指标（ionic conductivity、electrochemical window）、"
        "工艺方法（sintering、doping）、表征手段（EIS、XRD、SEM、TEM、DFT）\n"
        "- 避免泛化虚词（method、approach、study、review、advances等）\n"
        "- 每个关键词应为可独立用于学术检索的检索词\n"
        "- 保证缩写和全称同时包含\n"
    )

    _SYSTEM_STRATEGY = (
        "你是固态电池材料领域的研究策略专家，专精于将研究问题拆解为系统性的文献调研方案。"
        "你必须严格按照用户指令输出，只返回合法JSON，不输出任何解释、问候语或自我介绍。\n\n"
        "## 领域知识\n" + _DOMAIN_KNOWLEDGE + "\n"
        "## 子任务拆解原则\n"
        "1. 每个子任务应覆盖一个独立的研究维度，维度之间不重叠\n"
        "2. 常见维度：材料体系（电解质类型）、性能指标、工艺方法、界面/掺杂改性、表征/模拟\n"
        "3. 子任务之间可以有交叉但不应重复，例如：\n"
        "   - T1: 聚焦氧化物电解质的离子电导率提升策略\n"
        "   - T2: 聚焦硫化物电解质的界面稳定性问题\n"
        "   - T3: 聚焦不同掺杂元素对LLZO性能影响的对比\n"
        "4. 优先级1为最核心的维度，通常是对应原始问题最直接的查询方向\n\n"
        "## 检索策略设计原则\n"
        "1. Sciverse的agentic-search为语义检索，semantic_query宜使用完整英文句子而非关键词堆砌\n"
        "2. primary_keywords应整合已提供的关键词，并补充遗漏的学科专有名词\n"
        "3. semantic_queries应基于已有的改写查询做进一步调整，针对不同子任务定制\n"
        "4. filters可设置学科领域、发表年份（建议2015-2025覆盖近十年研究）、语言（en）\n"
        "5. 平衡广度（覆盖多种电解质体系）与精度（聚焦具体问题），避免检索策略过于泛化\n"
    )

    # ── Keyword extraction helpers ──

    @staticmethod
    def _tokenize_keywords(text: str) -> list[str]:
        """Extract English keywords from a text string, filtering stop words."""
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())
        return [w for w in words if w not in _STOP_WORDS]

    @staticmethod
    def _enrich_keywords(
        existing_kw: list[str],
        original_queries: list[str],
        reformulated_queries: list[str],
    ) -> list[str]:
        """Extract keywords from all queries and merge with existing ones.
        Flow: original queries -> reformulated queries -> extract from ALL -> deduplicate.
        """
        seen: dict[str, bool] = {}
        for kw in existing_kw:
            seen[kw.lower()] = True

        for q in original_queries + reformulated_queries:
            for token in TaskPlanningAgent._tokenize_keywords(q):
                if token not in seen:
                    seen[token] = True

        return list(seen.keys())

    # ── Prompt loading ──

    @staticmethod
    def _load_prompt(filename: str) -> str:
        """Load a prompt template from PROMPTS_DIR."""
        path = PROMPTS_DIR / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    # ── Main pipeline ──

    async def run(self, state: WorkflowState) -> dict:
        question = state.get("raw_question", "")

        # ═══════════════════════════════════════════════════════════
        # Phase 1: Query reformulation + keyword extraction
        # ═══════════════════════════════════════════════════════════
        self.log_step("1.1", "查询改写与关键词提取")
        ref_queries, ext_keywords = await self._reformulate(question)
        self.log_step("1.1", f"{len(ref_queries)} 个改写查询, {len(ext_keywords)} 个关键词", "done")

        # ═══════════════════════════════════════════════════════════
        # Phase 2: Strategy generation
        # ═══════════════════════════════════════════════════════════
        self.log_step("1.2", "检索策略生成与后处理")
        result = await self._generate_strategy(question, ref_queries, ext_keywords)
        subtasks = result.get("subtasks", [])
        search_strategy = result.get("search_strategy", {})

        # Inject reformulated_queries from Phase 1 into search_strategy
        search_strategy["reformulated_queries"] = ref_queries

        # ═══════════════════════════════════════════════════════════
        # Post-process: enrich keywords
        # ═══════════════════════════════════════════════════════════
        orig_queries = search_strategy.get("semantic_queries", [])
        llm_kw = search_strategy.get("primary_keywords", [])

        # Merge Phase 1 extracted keywords with Phase 2 primary_keywords
        merged_kw = list(dict.fromkeys(ext_keywords + llm_kw))
        enriched_kw = self._enrich_keywords(merged_kw, orig_queries, ref_queries)
        search_strategy["primary_keywords"] = enriched_kw

        # Enrich each subtask's keywords
        for st in subtasks:
            st_kw = st.get("keywords", [])
            st_sq = st.get("semantic_query", "")
            st["keywords"] = self._enrich_keywords(
                st_kw, [st_sq] if st_sq else [], ref_queries
            )

        n_orig = len(orig_queries)
        n_ref = len(ref_queries)
        self.log_step("1.2", f"{len(subtasks)} 个子任务, {len(enriched_kw)} 个关键词, {n_orig+n_ref} 条查询", "done")

        return {
            "structured_question": result.get("understood_question", question),
            "subtasks": subtasks,
            "search_strategy": search_strategy,
        }

    # ── Phase 1: Reformulation ──

    async def _reformulate(self, question: str) -> tuple[list[str], list[str]]:
        """Phase 1: Rewrite original query + extract keywords via LLM.

        Returns (reformulated_queries, extracted_keywords).
        Falls back to manual reformulation if LLM is unavailable or fails.
        """
        if not self.llm:
            return self._manual_reformulate(question)

        reform_prompt = self._load_prompt(self.reformulation_prompt_file)
        if not reform_prompt:
            self.log("Reformulation prompt not found, using manual", "dim")
            return self._manual_reformulate(question)

        system = self._SYSTEM_REFORMULATION
        user = reform_prompt.format(question=question)

        try:
            result = await self.llm.complete_json(system, user, max_tokens=self.config.get("reformulation_max_tokens", 4096))
            ref_queries: list[str] = result.get("reformulated_queries", [])
            ext_keywords: list[str] = result.get("extracted_keywords", [])

            if not ref_queries or not ext_keywords:
                self.log("LLM reformulation returned empty, using manual", "yellow")
                return self._manual_reformulate(question)

            # Log the reformulated queries
            for i, rq in enumerate(ref_queries):
                self.log(f"  Reformulated #{i+1}: {rq}", "dim")
            self.log(f"  Extracted keywords: {ext_keywords}", "dim")

            return ref_queries, ext_keywords
        except Exception as e:
            self.log(f"LLM reformulation failed ({e}), using manual", "yellow")
            return self._manual_reformulate(question)

    # ── Phase 2: Strategy generation ──

    async def _generate_strategy(
        self,
        question: str,
        ref_queries: list[str],
        ext_keywords: list[str],
    ) -> dict:
        """Phase 2: Generate search strategy using reformulated queries + keywords.

        Returns dict with: understood_question, subtasks, search_strategy.
        Falls back to manual strategy if LLM is unavailable or fails.
        """
        if not self.llm:
            return self._manual_strategy(question, ref_queries, ext_keywords)

        strategy_prompt = self.prompt  # task_planning.txt
        if not strategy_prompt:
            self.log("Strategy prompt not found, using manual", "dim")
            return self._manual_strategy(question, ref_queries, ext_keywords)

        system = self._SYSTEM_STRATEGY
        user = strategy_prompt.format(
            question=question,
            reformulated_queries="\n".join(f"- {q}" for q in ref_queries),
            extracted_keywords=", ".join(ext_keywords),
        )

        try:
            result = await self.llm.complete_json(system, user, max_tokens=self.config.get("max_tokens", 8192))
            if not result.get("subtasks") or not result.get("search_strategy", {}).get("primary_keywords"):
                self.log("LLM strategy generation returned empty, using manual", "yellow")
                return self._manual_strategy(question, ref_queries, ext_keywords)
            return result
        except Exception as e:
            self.log(f"LLM strategy generation failed ({e}), using manual", "yellow")
            return self._manual_strategy(question, ref_queries, ext_keywords)

    # ── Manual fallbacks ──

    def _manual_reformulate(self, question: str) -> tuple[list[str], list[str]]:
        """Manual query reformulation + keyword extraction (no LLM)."""
        base = question.strip().rstrip("？?")
        reformulated = [
            f"advances in {base}",
            f"recent progress on {base}",
            f"research on {base}",
            f"review of {base}",
        ]

        # Extract raw keywords from the original question (Chinese-aware split)
        raw_kw: list[str] = []
        for t in re.sub(r"[的与和及之其是了在]", " ", question).split():
            t = t.strip()
            if len(t) > 1:
                raw_kw.append(t)

        keywords = self._enrich_keywords(raw_kw, [question], reformulated)
        return reformulated, keywords

    def _manual_strategy(
        self,
        question: str,
        ref_queries: list[str],
        ext_keywords: list[str],
    ) -> dict:
        """Manual strategy generation using pre-extracted reformulated queries + keywords."""
        return {
            "understood_question": question,
            "subtasks": [
                {
                    "id": "T1",
                    "dimension": "材料体系",
                    "keywords": ext_keywords[:8],
                    "semantic_query": question,
                    "priority": 1,
                    "filters": {"year_start": 2015, "year_end": 2026},
                }
            ],
            "search_strategy": {
                "primary_keywords": ext_keywords,
                "semantic_queries": [question],
                "filters": {
                    "lang": "en",
                    "publication_published_year": {"gte": 2015, "lte": 2026},
                },
            },
        }
