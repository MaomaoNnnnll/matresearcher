"""LangGraph workflow engine for the MatResearcher pipeline (12 core steps + optional Step 8.5/10.5 extensions, 2 conditional branches).

Pipeline:
  Step 1:   Task Planning and Search Strategy Generation
  Step 2:   Literature Search (Sci-api)
  Step 3:   Literature Coverage Check
  Step 4:   LLM Prefilter (three-way classification: relevant/partial/irrelevant)
  Step 5:   Literature Filter (dedup + rerank)
  Step 6:   PDF Parsing (MinerU)
  Step 7:   Knowledge Extraction + Data Quality Check
  Step 8:   Knowledge Fusion (normalize + store + fuse + conflicts)
  Step 9:   Gap Generation + Scoring
  Step 10:     Evidence Verification (backtrack to full text)
  Step 11:     Report Generation
  Step 12:     Final Fact Check

Conditional edges (2):
  - After coverage_check: if not passed → refine strategy → back to literature_search
  - After fact_check: while revisions remain → back to report_generation; else → END

Two optional nodes are inserted when enabled in config/workflow.yaml:
  - structure_property (Step 8.5): knowledge_fusion → structure_property → gap_generation
  - hypothesis_crosscheck (Step 10.5): evidence_verification → hypothesis_crosscheck
    → report_generation
"""
from __future__ import annotations


import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.progress import Progress, SpinnerColumn, TextColumn
import yaml

from ..state import WorkflowState
from .nodes import create_all_nodes
from .run_cache import RunCache

console = Console()

# Nodes that are part of a back-edge (cycle) in the graph. The run-level node
# cache is keyed only by node NAME (no input hash) and is consulted both across
# process runs (--resume) and within a single run. For nodes on a cycle the
# cache is semantically wrong: on the 2nd loop visit it replays a STALE patch
# (e.g. the previous fact_check result with revision=1 / needs_revision=True),
# so the fact_check counter never advances and the loop spins forever until the
# recursion limit is hit. Excluding these nodes from the cache guarantees the
# loop re-executes fresh each pass and terminates after max_revisions rounds.
LOOP_NODES = {"report_generation", "fact_check"}


class MatResearcherWorkflow:
    """Orchestrate the full MatResearcher pipeline using LangGraph.

    Usage:
        workflow = MatResearcherWorkflow("config/workflow.yaml")
        wf = workflow.compile()
        result = await wf.ainvoke({"raw_question": "..."})
    """

    def __init__(self, config_path: str | None = None, resume_run_id: str | None = None):
        """Initialize the workflow engine.

        Args:
            config_path: Path to workflow YAML config; defaults to config/workflow.yaml
            resume_run_id: If set, resume from a previous run (reuse its log dir
                and node/document caches, skipping already-completed steps).
        """
        self.config = self._load_config(config_path)
        self.nodes = None  # lazy init in compile()
        self._compiled = None
        self.resume_run_id = resume_run_id
        self._cache: RunCache | None = None
        self._current_node: str | None = None

        # Extract loop limits from config
        wf_config = self.config.get("workflow", {})
        self.max_coverage_retries = wf_config.get("coverage_check", {}).get("max_retries", 2)
        self.data_quality_enabled = wf_config.get("data_quality_check", {}).get("enabled", True)

        # Extract logging config
        log_cfg = wf_config.get("logging", {})
        self.log_enabled = log_cfg.get("enabled", True)
        self.log_dir = self._resolve_log_dir(log_cfg.get("output_dir", "outputs/logs"))

    def compile(self):
        """Compile the LangGraph StateGraph. Must be called before invoking."""
        if self._compiled is not None:
            return self._compiled

        # Initialize nodes (lazy, creates agent instances)
        if self.log_enabled:
            # 每次运行创建独立的时间戳子文件夹；--resume 时复用原 run 目录
            run_ts = self.resume_run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
            run_log_dir = os.path.join(self.log_dir, run_ts)
            self.config["log_dir"] = run_log_dir
            self._run_log_dir = run_log_dir
            # Run-level cache (node pkl + per-paper pkl + MANIFEST)
            self._cache = RunCache(run_log_dir)
            if self.resume_run_id:
                cached = self._cache.completed_nodes()
                if cached:
                    console.print(
                        f"  [cyan]Resume run {run_ts}: {len(cached)} 个节点缓存可用 "
                        f"({', '.join(cached)})[/cyan]"
                    )
        else:
            self.config["log_dir"] = None
            self._run_log_dir = None
            self._cache = None

        # ── Token usage accounting (per-stage LLM cost control) ──
        # Created BEFORE nodes so every agent's LLMClient shares one counter;
        # only calls made by THIS process are counted (cached nodes never call
        # the LLM, so a resumed run reports only freshly spent tokens).
        from ..tools.token_counter import TokenCounter
        self._token_counter = TokenCounter()
        self.config["token_counter"] = self._token_counter

        self.nodes = create_all_nodes(self.config, run_cache=self._cache)

        # Build the state graph
        graph = StateGraph(WorkflowState)

        # Add all nodes (wrapped with cache check/save)
        for name, func in self.nodes.items():
            graph.add_node(name, self._wrap_node(name, func))

        # Set entry point
        graph.set_entry_point("task_planning")

        # ─── Linear edges ───
        graph.add_edge("task_planning", "literature_search")
        
        graph.add_edge("literature_search", "coverage_check")

        graph.add_conditional_edges(
            "coverage_check",
            self._route_after_coverage,
            {
                "retry_search": "literature_search",
                "continue": "llm_prefilter",
            },
        )

        graph.add_edge("llm_prefilter", "literature_filter")
        
        # linear chain
        graph.add_edge("literature_filter", "pdf_parsing")
        graph.add_edge("pdf_parsing", "knowledge_extraction")
        graph.add_edge("knowledge_extraction", "knowledge_fusion")

        # Step 8.5 (optional): quantitative structure-property analysis.
        # Present in self.nodes only when enabled in config/workflow.yaml.
        if "structure_property" in self.nodes:
            graph.add_edge("knowledge_fusion", "structure_property")
            graph.add_edge("structure_property", "gap_generation")
        else:
            graph.add_edge("knowledge_fusion", "gap_generation")

        graph.add_edge("gap_generation", "evidence_verification")

        # Step 10.5 (optional): cross-check each gap's hypothesis against the
        # Materials Project API + Sci-Base offline corpus. Present in self.nodes
        # only when materials_project is enabled in config/workflow.yaml.
        if "hypothesis_crosscheck" in self.nodes:
            graph.add_edge("evidence_verification", "hypothesis_crosscheck")
            graph.add_edge("hypothesis_crosscheck", "report_generation")
        else:
            graph.add_edge("evidence_verification", "report_generation")

        graph.add_edge("report_generation", "fact_check")

        # ── Fact-check closed loop ──
        # Instead of ending with a report that still contains its own issue log,
        # send the draft back to report_generation while revisions remain.
        graph.add_conditional_edges(
            "fact_check",
            self._route_after_fact_check,
            {
                "revise": "report_generation",
                "done": END,
            },
        )

        self._compiled = graph.compile()
        return self._compiled

    def _route_after_fact_check(self, state: WorkflowState) -> str:
        """Conditional routing after Step 12 fact-check.

        Returns "revise" while issues remain and the revision budget
        (`agents.evidence_verification.max_revisions`) is not exhausted;
        otherwise ends the run."""
        return "revise" if state.get("needs_revision") else "done"

    def _wrap_node(self, name: str, func):
        """Wrap a LangGraph node with run-cache check/save.

        On cache hit: skip execution entirely (no LLM / no Sciverse) and
        return the previously persisted state-patch. On cache miss: run the
        node, then atomically persist its returned patch so a crash can be
        resumed with --resume <run_id>.
        """
        async def wrapped(state: WorkflowState) -> dict:
            self._current_node = name
            cache = self._cache
            # Nodes on a cycle must never replay a cached patch — see LOOP_NODES.
            if cache is not None and name not in LOOP_NODES and cache.has_node(name):
                patch = cache.get_node(name)
                if patch is not None:
                    console.print(
                        f"  [dim cyan]🔄 {name}: 命中缓存，跳过执行[/dim cyan]"
                    )
                    state.setdefault("step_log", []).append({
                        "timestamp": datetime.now().isoformat(),
                        "step": name,
                        "message": "缓存复用（--resume）",
                    })
                    return patch
                console.print(
                    f"  [yellow]⚠ {name}: 缓存文件损坏，重新执行[/yellow]"
                )
            result = await func(state)
            if cache is not None and name not in LOOP_NODES:
                try:
                    cache.save_node(name, result)
                except Exception as e:
                    console.print(
                        f"  [yellow]⚠ {name}: 缓存保存失败: {rich_escape(str(e))}[/yellow]"
                    )
            return result
        return wrapped

    @property
    def run_output_dir(self) -> str | None:
        """Return the run-specific output directory (logs + report share the same folder)."""
        return self._run_log_dir

    async def run(self, question: str, **kwargs) -> dict:
        """Run the full pipeline for a research question.

        Args:
            question: Natural language research question.
            **kwargs: Additional state overrides.

        Returns:
            Final WorkflowState dict with all steps completed.
        """
        wf = self.compile()

        initial_state: WorkflowState = {
            "raw_question": question,
            "coverage_retry_count": 0,
            "config": self.config,
            "step_log": [],
            **kwargs,
        }

        start_time = datetime.now()
        console.print(f"\n[bold green]MatResearcher Pipeline Started[/bold green]")
        console.print(f"Question: {question[:120]}")
        console.print(f"Max coverage retries: {self.max_coverage_retries}\n")

        try:
            final_state = await wf.ainvoke(
                initial_state,
                config={"recursion_limit": 150},  # 兜底防御：一旦引入新环可快速失败，而非空转 10007 次
            )
        except Exception as e:
            # rich_escape: an exception message containing '[' (e.g. MarkupError
            # text, citation markers) must not be parsed as rich markup — that
            # would mask the real error with a secondary MarkupError.
            console.print(f"\n[bold red]Pipeline Error: {rich_escape(str(e))}[/bold red]")
            if self._cache is not None:
                console.print(
                    f"[yellow]运行中断于节点 {self._current_node}。"
                    f"可执行 --resume {self._cache.run_id} 从断点续跑"
                    f"（已完成的节点将复用缓存，不消耗 token）[/yellow]"
                )
            import traceback
            traceback.print_exc()
            return {
                **initial_state,
                "error": str(e),
                "final_report": f"# Pipeline Error\n\nError: {e}\n\nPlease check the logs for details.",
            }

        elapsed = (datetime.now() - start_time).total_seconds()
        console.print(f"\n[bold green]✓ Pipeline Complete[/bold green] ({elapsed:.1f}s)")

        # Print summary
        log = final_state.get("step_log", [])
        if log:
            console.print(f"\n[bold green]Step Summary:[/bold green]")
            for entry in log:
                console.print(f"  [dim green][{entry.get('step')}][/dim green] {entry.get('message')}")

        # ── Token usage summary (per-stage LLM cost control) ──
        if getattr(self, "_token_counter", None) is not None:
            self._print_token_summary()

        # Print report preview (submission 模式优先展示参赛方案文档)
        report = (
            final_state.get("submission_report")
            or final_state.get("final_report")
            or final_state.get("draft_report", "")
        )
        if report:
            preview = report[:500] + ("..." if len(report) > 500 else "")
            console.print(f"\n[bold green]Report Preview:[/bold green]\n{preview}")

        return final_state

    def _print_token_summary(self):
        """Render the per-stage token usage table and persist token_usage.json."""
        counter = self._token_counter
        try:
            console.print()  # blank line
            console.print(counter.format_table())
        except Exception:
            pass  # token accounting must never break the pipeline

        # Persist machine-readable dump next to the run logs
        if self._run_log_dir:
            try:
                import json
                dump_path = os.path.join(self._run_log_dir, "token_usage.json")
                with open(dump_path, "w", encoding="utf-8") as f:
                    json.dump(counter.to_dict(), f, ensure_ascii=False, indent=2)
                console.print(f"  [dim]Token 明细已保存: {dump_path}[/dim]")
            except Exception:
                pass

    def _route_after_coverage(self, state: WorkflowState) -> str:
        """Conditional routing after coverage check (Step 4a).

        Returns:
            "retry_search" if coverage failed and retries remain.
            "continue" to proceed to filtering otherwise.
        """
        coverage = state.get("coverage_check_result", {})
        retry = state.get("coverage_retry_count", 0)

        if not coverage.get("passed", True) and retry < self.max_coverage_retries:
            issues = coverage.get("issues", [])
            console.print(
                f"  [yellow]Coverage insufficient, retry {retry}/{self.max_coverage_retries}[/yellow]"
            )
            for issue in issues:
                console.print(f"    - {issue.get('type')}: {issue.get('detail', '')}")

            self._refine_search_strategy(state, issues)
            # Search strategy changed → the retrieved literature set will change,
            # so every downstream cached node (incl. per-paper caches) is stale.
            if self._cache is not None:
                removed = self._cache.invalidate_from("literature_search")
                if removed:
                    console.print(
                        f"  [yellow]缓存失效: 策略调整后 {len(removed)} 个下游缓存已清除，"
                        f"重新执行文献检索及后续步骤[/yellow]"
                    )
            return "retry_search"

        if not coverage.get("passed", True):
            console.print(
                f"  [red]Coverage check failed after {retry} retries, proceeding anyway[/red]"
            )

        return "continue"

    def _refine_search_strategy(self, state: dict, issues: list[dict]):
        """Refine search strategy based on coverage check feedback.

        For low_subtask_coverage: adds broader reformulated semantic queries
        targeting under-covered dimensions (not just appending dimension names
        as keywords, which Sciverse can't interpret).

        For poor_year_distribution: broadens year range.
        """
        strategy = state.get("search_strategy", {})

        for issue in issues:
            if issue.get("type") == "low_subtask_coverage":
                dim = issue.get("dimension", "")
                if not dim:
                    continue

                # Instead of appending dimension names as keywords (ineffective),
                # generate broader reformulated semantic queries
                queries = strategy.get("semantic_queries", [])
                broader = f"recent advances and comprehensive review of {dim}"
                if broader not in queries:
                    queries.append(broader)

                # Also add dimension keywords to broaden the keyword-based search
                keywords = strategy.get("primary_keywords", [])
                # Extract meaningful tokens from dimension name
                for token in dim.split():
                    token = token.strip()
                    if len(token) > 3 and token not in keywords:
                        keywords.append(token)

                strategy["semantic_queries"] = queries
                strategy["primary_keywords"] = keywords

            elif issue.get("type") == "poor_year_distribution":
                filters = strategy.get("filters", {})
                if "year_start" in filters:
                    filters["year_start"] = max(2010, filters["year_start"] - 5)
                strategy["filters"] = filters

        state["search_strategy"] = strategy

    def _resolve_log_dir(self, rel_path: str) -> str:
        """Resolve a relative log directory to an absolute path under the project root."""
        root = Path(__file__).resolve().parents[3]
        if os.path.isabs(rel_path):
            return rel_path
        return str(root / rel_path)

    # ─── Config loading ───

    @staticmethod
    def _load_config(config_path: str | None = None) -> dict:
        """Load workflow configuration from YAML file.

        Resolves ${ENV_VAR} placeholders from environment variables.
        Loads .env into os.environ first (defense-in-depth).
        """
        # Ensure env files are loaded (main.py does this first, but engine may
        # be instantiated directly from notebooks/scripts). Loads project .env
        # (non-secret) + ~/.matresearcher/secrets.env (API keys, outside repo).
        try:
            from ..env_loader import load_env_files
            load_env_files()
        except ImportError:
            pass
        if config_path is None:
            config_path = Path(__file__).resolve().parents[3] / "config" / "workflow.yaml"

        path = Path(config_path)
        if not path.exists():
            console.print(f"[yellow]Config not found: {path}, using defaults[/yellow]")
            return {}

        with open(path, encoding="utf-8") as f:
            raw = f.read()

        # Resolve ${VAR} placeholders
        import re
        def resolve_env(match):
            var = match.group(1)
            return os.getenv(var, "")

        resolved = re.sub(r'\$\{(\w+)\}', resolve_env, raw)
        return yaml.safe_load(resolved) or {}


# ─── Convenience Functions ───

async def run_survey(question: str, config_path: str | None = None) -> str:
    """High-level convenience function: run survey and get report.

    Args:
        question: Research question to survey.
        config_path: Optional path to workflow config.

    Returns:
        Final report as markdown string.
    """
    workflow = MatResearcherWorkflow(config_path)
    result = await workflow.run(question)
    return (
        result.get("submission_report")
        or result.get("final_report")
        or result.get("draft_report", "")
    )
