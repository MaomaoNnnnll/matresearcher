"""LangGraph node wrappers for each agent in the 18-step pipeline.

Each node function:
1. Takes the current WorkflowState as input
2. Invokes the corresponding agent
3. Returns partial state updates
4. Logs progress
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

from rich.console import Console

from ..state import WorkflowState
from ..agents.task_planning import TaskPlanningAgent
from ..agents.literature_search import LiteratureSearchAgent
from ..agents.literature_filter import LiteratureFilterAgent
from ..agents.llm_prefilter import LLMPrefilterAgent
from ..agents.pdf_parsing import PDFParsingAgent
from ..agents.knowledge_extraction import KnowledgeExtractionAgent
from ..agents.knowledge_fusion import KnowledgeFusionAgent
from ..agents.gap_identification import GapIdentificationAgent
from ..agents.evidence_verification import EvidenceVerificationAgent
from ..agents.report_generation import ReportGenerationAgent

import os
from pathlib import Path

# Project root (independent of CWD): F:/projects/matresearcher
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def create_all_nodes(config: dict[str, Any], run_cache=None) -> dict[str, Callable]:
    """Create all LangGraph node functions with shared agent instances.

    Args:
        config: Pipeline configuration dict.
        run_cache: Optional RunCache instance for node/doc-level checkpointing.

    Returns a dict mapping node_name -> async callable(state) -> dict.
    """
    from ..tools.llm import LLMClient
    from ..tools.sciverse import SciverseClient
    from ..tools.mineru import MinerUParser
    from ..tools.embedding import EmbeddingModel
    from ..tools.reranker import RerankerModel
    from ..knowledge_base.vector_store import VectorStore
    from ..knowledge_base.relational import RelationalStore

    # Initialize shared tools
    agent_configs = config.get("agents", {})
    llm_cfg = agent_configs.get("default_llm", {})

    llm = LLMClient(
        api_base=llm_cfg.get("api_base") or os.getenv("LLM_API_BASE"),
        api_key=llm_cfg.get("api_key") or os.getenv("LLM_API_KEY") or os.getenv("MINIMAX_API_KEY") ,
        model=llm_cfg.get("model") or os.getenv("LLM_MODEL"),
        temperature=llm_cfg.get("temperature", 0.3),
        # engine.py creates ONE shared TokenCounter BEFORE create_all_nodes and
        # stores it in config["token_counter"]; without passing it here every
        # agent's usage silently goes uncounted (token_usage.json stayed 0/0).
        token_counter=config.get("token_counter"),
    )

    sciverse = SciverseClient(
        api_base=config.get("sciverse_api_base"),
        api_key=config.get("sciverse_api_key"),
    )

    mineru = MinerUParser(
        api_url=config.get("mineru_api_url"),
        extract_tables=config.get("pdf_extract_tables", True),
        extract_figures=config.get("pdf_extract_figures", True),
    )

    embedding = EmbeddingModel(
        model_name=config.get("embedding_model", "BAAI/bge-m3"),
    )

    reranker = RerankerModel(
        model_name=config.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
    )

    vector_store = VectorStore(
        # Absolute path under the project root (never CWD-relative — a
        # relative default used to split data between ./data and src/data).
        db_path=config.get("vector_persist_dir")
        or str(PROJECT_ROOT / "data" / "chroma"),
    )

    relational_store = RelationalStore(
        db_url=config.get("database_url")
        or f"sqlite:///{(PROJECT_ROOT / 'data' / 'matresearcher.db').as_posix()}",
    )

    # Log file output directory (None = file logging disabled)
    log_dir = config.get("log_dir")

    # Create agent instances
    task_planning = TaskPlanningAgent(
        llm=llm,
        config=agent_configs.get("task_planning", {}),
        log_dir=log_dir,
    )

    literature_search = LiteratureSearchAgent(
        llm=llm,
        config=agent_configs.get("literature_search", {}),
        sciverse=sciverse,
        log_dir=log_dir,
    )

    llm_prefilter = LLMPrefilterAgent(
        llm=llm,
        config=agent_configs.get("llm_prefilter", {}),
        log_dir=log_dir,
    )

    literature_filter = LiteratureFilterAgent(
        llm=llm,
        config=agent_configs.get("literature_filter", {}),
        reranker=reranker,
        log_dir=log_dir,
    )

    pdf_parsing = PDFParsingAgent(
        llm=llm,
        config=agent_configs.get("pdf_parsing", {}),
        mineru=mineru,
        sciverse=sciverse,
        log_dir=log_dir,
    )

    knowledge_extraction = KnowledgeExtractionAgent(
        llm=llm,
        config=agent_configs.get("knowledge_extraction", {}),
        log_dir=log_dir,
        run_cache=run_cache,
    )

    knowledge_fusion = KnowledgeFusionAgent(
        llm=llm,
        config=agent_configs.get("knowledge_fusion", {}),
        vector_store=vector_store,
        relational_store=relational_store,
        embedding_model=embedding,
        log_dir=log_dir,
    )

    gap_identification = GapIdentificationAgent(
        llm=llm,
        config=agent_configs.get("gap_identification", {}),
        log_dir=log_dir,
    )

    evidence_verification = EvidenceVerificationAgent(
        llm=llm,
        config=agent_configs.get("evidence_verification", {}),
        sciverse=sciverse,
        log_dir=log_dir,
    )

    report_generation = ReportGenerationAgent(
        llm=llm,
        config=agent_configs.get("report_generation", {}),
        log_dir=log_dir,
    )

    # ─── Node Functions ───

    async def node_task_planning(state: WorkflowState) -> dict:
        """Steps 1: Decompose question into subtasks and search strategy."""
        console.print(f"\n  [bold green]▸ Step 1:[/bold green] 任务规划与子任务分解")
        _log_step(state, 1, "任务规划和策略生成")
        result = await task_planning.run(state)
        _log_step(state, 1, "任务规划和策略生成完成")
        console.print(f"  [bold green]✔ Step 1 完成:[/bold green] 子任务已分解完毕")
        return result

    async def node_literature_search(state: WorkflowState) -> dict:
        """Step 2: Execute literature search via Sciverse."""
        console.print(f"\n  [bold green]▸ Step 2:[/bold green] 文献检索 (Sciverse)")
        _log_step(state, 2, "文献检索")
        result = await literature_search.run(state)
        n = len(result.get('candidate_literature', []))
        _log_step(state, 2, f"文献检索完成: {n} 篇候选")
        console.print(f"  [bold green]✔ Step 2 完成:[/bold green] {n} 篇候选文献")
        return result

    async def node_coverage_check(state: WorkflowState) -> dict:
        """Step 3: Verify search coverage is sufficient."""
        console.print(f"\n  [bold green]▸ Step 3:[/bold green] 检索覆盖度核验")
        _log_step(state, "3", "检索覆盖度核验")
        result = await evidence_verification.check_coverage(state)
        passed = result.get("coverage_check_result", {}).get("passed", False)
        _log_step(state, "3", f"覆盖度核验: {'通过' if passed else '未通过'}")
        console.print(f"  [bold green]✔ Step 3 完成:[/bold green] {'通过' if passed else '未通过 — 将补充检索'}")
        return result

    async def node_llm_prefilter(state: WorkflowState) -> dict:
        """Step 4: LLM three-way classification to reduce Reranker workload."""
        console.print(f"\n  [bold green]▸ Step 4:[/bold green] LLM三分类预筛")
        _log_step(state, 4, "LLM三分类预筛")
        result = await llm_prefilter.run(state)
        kept = len(result.get("candidate_literature", []))
        _log_step(state, 4, f"LLM预筛完成: {kept} 篇进入精筛")
        console.print(f"  [bold green]✔ Step 4 完成:[/bold green] {kept} 篇进入精筛")
        return result

    async def node_literature_filter(state: WorkflowState) -> dict:
        """Step 5: Dedup + rerank candidate literature."""
        console.print(f"\n  [bold green]▸ Step 5:[/bold green] 文献筛选 (去重 + 重排序)")
        _log_step(state, 5, "文献精筛")
        result = await literature_filter.run(state)
        filtered = result.get("filtered_literature", [])
        _log_step(state, 5, f"文献精筛完成: {len(filtered)} 篇保留")
        console.print(f"  [bold green]✔ Step 5 完成:[/bold green] {len(filtered)} 篇保留")

        # Persist literature metadata to relational DB
        if filtered:
            lit_dicts = [
                {
                    "id": lit.id,
                    "doi": lit.metadata.doi,
                    "title": lit.metadata.title,
                    "authors": lit.metadata.authors,
                    "year": lit.metadata.year,
                    "journal": lit.metadata.journal,
                    "abstract": lit.metadata.abstract,
                    "relevance_score": lit.relevance_score,
                    "verification_status": lit.verification_status,
                    "extra": {
                        "keywords": lit.metadata.keywords,
                        "pdf_url": lit.metadata.pdf_url,
                        "doc_id": lit.metadata.doc_id,
                        "citation_count": lit.metadata.citation_count,
                        "query_source": lit.metadata.query_source,
                        "is_content_accessible": lit.metadata.is_content_accessible,
                    },
                }
                for lit in filtered
            ]
            try:
                relational_store.add_literature_batch(lit_dicts)
            except Exception as e:
                print(f"[WARNING] Literature persistence failed: {e}")

        return result

    async def node_pdf_parsing(state: WorkflowState) -> dict:
        """Step 6: Parse PDFs with MinerU."""
        console.print(f"\n  [bold green]▸ Step 6:[/bold green] PDF解析 (MinerU + Sciverse fallback)")
        _log_step(state, 6, "PDF解析")
        result = await pdf_parsing.run(state)
        _log_step(state, 6, "PDF解析完成")
        console.print(f"  [bold green]✔ Step 6 完成:[/bold green] PDF解析完成")
        return result

    async def node_knowledge_extraction(state: WorkflowState) -> dict:
        """Step 7: Extract knowledge + three-way quality gate."""
        console.print(f"\n  [bold green]▸ Step 7:[/bold green] 知识提取 + 数据质量核验 (PASS/REVIEW/FAIL)")
        _log_step(state, 7, "知识提取 + 数据质量核验")
        result = await knowledge_extraction.run(state)
        anomalies = result.get("anomaly_records", [])
        if anomalies:
            n_review = sum(1 for r in anomalies if getattr(r, "quality_status", "") == "review")
            n_fail = len(anomalies) - n_review
            _log_step(state, 7, f"数据质量核验: {n_review} 条 REVIEW + {n_fail} 条 FAIL 已分离 (见 quality_report.json)")
            console.print(f"  [bold green]✔ Step 7 完成:[/bold green] 提取完成; REVIEW {n_review} 条 / FAIL {n_fail} 条已分离")
        else:
            _log_step(state, 7, "知识提取完成")
            console.print(f"  [bold green]✔ Step 7 完成:[/bold green] 提取完成; 数据质量正常")
        return result

    async def node_knowledge_fusion(state: WorkflowState) -> dict:
        """Steps 8: Normalize, store, fuse, detect conflicts."""
        console.print(f"\n  [bold green]▸ Step 8:[/bold green] 知识融合 (归一化 + 存储 + 融合 + 冲突检测)")
        _log_step(state, 8, "知识融合 (归一化+存储+融合+冲突检测)")
        result = await knowledge_fusion.run(state)
        n_materials = result.get('fused_table', {}).total_materials if hasattr(result.get('fused_table', {}), 'total_materials') else 0
        _log_step(state, 8, f"知识融合完成: {n_materials} 种材料")
        console.print(f"  [bold green]✔ Step 8 完成:[/bold green] {n_materials} 种材料已融合")
        return result

    async def node_gap_generation(state: WorkflowState) -> dict:
        """Steps 9: Generate and score Research Gaps."""
        console.print(f"\n  [bold green]▸ Step 9:[/bold green] 研究缺口识别与评分")
        _log_step(state, 9, "研究缺口识别 + 评分")
        result = await gap_identification.run(state)
        gaps = result.get("scored_gaps", result.get("gaps", []))
        top1_score = f"{gaps[0].score.total_score:.3f}" if gaps and gaps[0].score else "N/A"
        _log_step(state, 9, f"缺口识别完成: {len(gaps)} 个缺口 (Top-1 评分: {top1_score})")
        console.print(f"  [bold green]✔ Step 9 完成:[/bold green] {len(gaps)} 个缺口 (Top-1: {top1_score})")
        console.print(f"  [dim green]   → 进入 Step 10: 证据溯源核验[/dim green]")
        return result

    async def node_evidence_verification(state: WorkflowState) -> dict:
        """Step 10: Backtrack gap claims to original full-text passages."""
        console.print(f"\n  [bold green]▸ Step 10:[/bold green] 证据溯源核验")
        _log_step(state, 10, "证据溯源核验")
        result = await evidence_verification.verify_gaps(state)
        gaps = result.get("scored_gaps", [])
        passed = sum(1 for g in gaps if hasattr(g, 'verification_status') and g.verification_status == "passed")
        _log_step(state, 10, f"证据溯源完成: {passed}/{len(gaps)} 条缺口通过核验")
        console.print(f"  [bold green]✔ Step 10 完成:[/bold green] {passed}/{len(gaps)} 条缺口通过核验")
        console.print(f"  [dim green]   → 进入 Step 11: 报告生成[/dim green]")
        return result

    async def node_report_generation(state: WorkflowState) -> dict:
        """Step 11: Generate structured literature survey report."""
        console.print(f"\n  [bold green]▸ Step 11:[/bold green] 调研报告生成")
        _log_step(state, 11, "调研报告生成")
        result = await report_generation.run(state)
        _log_step(state, 11, f"报告生成完成: {len(result.get('draft_report', ''))} 字符")
        console.print(f"  [bold green]✔ Step 11 完成:[/bold green] {len(result.get('draft_report', ''))} 字符")
        console.print(f"  [dim green]   → 进入 Step 12: 事实核查[/dim green]")
        return result

    async def node_fact_check(state: WorkflowState) -> dict:
        """Step 12: Final fact-checking of the draft report."""
        console.print(f"\n  [bold green]▸ Step 12:[/bold green] 最终事实核查")
        _log_step(state, 12, "最终事实核查")
        result = await evidence_verification.fact_check_report(state)
        issues = result.get("fact_check_issues", [])
        _log_step(state, 12, f"事实核查完成: {len(issues)} 条待确认项")
        console.print(f"  [bold green]✔ Step 12 完成:[/bold green] {len(issues)} 条待确认项")
        return result

    # Return all node functions keyed by node name
    return {
        "task_planning": node_task_planning,
        "literature_search": node_literature_search,
        "coverage_check": node_coverage_check,
        "llm_prefilter": node_llm_prefilter,
        "literature_filter": node_literature_filter,
        "pdf_parsing": node_pdf_parsing,
        "knowledge_extraction": node_knowledge_extraction,
        "knowledge_fusion": node_knowledge_fusion,
        "gap_generation": node_gap_generation,
        "evidence_verification": node_evidence_verification,
        "report_generation": node_report_generation,
        "fact_check": node_fact_check,
    }


# ─── Global console for step banners ───
console = Console()


def _log_step(state: dict, step: str | int, message: str):
    """Append a step log entry to the workflow state."""
    if "step_log" not in state:
        state["step_log"] = []
    state["step_log"].append({
        "timestamp": datetime.now().isoformat(),
        "step": str(step),
        "message": message,
    })
