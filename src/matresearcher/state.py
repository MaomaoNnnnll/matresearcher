"""Workflow state definition for LangGraph.

This TypedDict is the shared state object passed through the entire
12-step pipeline (steps 1-12), with two optional nodes (Step 8.5
structure-property analysis, Step 10.5 hypothesis cross-check) mounted
when enabled in config/workflow.yaml.
"""
from __future__ import annotations

from typing import TypedDict, Optional, Any

from .models.literature import Literature
from .models.knowledge import KnowledgeRecord, NormalizedRecord, FusedKnowledgeTable
from .models.gap import ResearchGap, ConflictItem, MissingItem


class WorkflowState(TypedDict, total=False):
    """Shared state flowing through the LangGraph workflow.

    Each agent reads from and writes to this state dict.
    """

    # --- Step 1-3: Task Planning ---
    raw_question: str                          # user's natural-language question
    structured_question: dict                  # parsed scientific question
    subtasks: list[dict]                       # search subtasks with priority
    search_strategy: dict                      # Sciverse query plan

    # --- Step 4: Literature Search ---
    candidate_literature: list[Literature]     # raw search results

    # --- Step 4a: Coverage Check (NEW) ---
    coverage_check_result: dict                # {"passed": bool, "missing_dims": [...]}
    coverage_retry_count: int                  # loop counter (max 2)

    # --- Step 5: Filtering ---
    filtered_literature: list[Literature]      # after dedup + rerank

    # --- Step 6-7: PDF Parsing & Knowledge Extraction ---
    # literature.parsed_document populated by step 6
    # literature.knowledge_records populated by step 7

    # --- Step 7a: Data Quality Check (NEW) ---
    anomaly_records: list[KnowledgeRecord]     # failed verification, stored separately

    # --- Step 8-9: Normalization & Storage ---
    normalized_records: list[NormalizedRecord]

    # --- Step 10-11: Fusion & Conflict Detection ---
    fused_table: FusedKnowledgeTable
    conflicts: list[dict]                       # ConflictItem dicts
    missing_items: list[dict]                   # MissingItem dicts

    # --- Step 12-14: Gap Generation & Scoring ---
    gaps: list[ResearchGap]
    scored_gaps: list[ResearchGap]              # sorted by score

    # --- Step 15-16: Report ---
    draft_report: str                           # SURVEY markdown (文献调研报告本体；事实核查修订闭环基于它)
    final_report: str                           # verified markdown
    # Submission-mode deliverable (参赛方案文档). Kept SEPARATE from draft_report so
    # that draft_report always stays the survey report (with 参考文献清单 + 构效关系)
    # and the fact-check revision loop rewrites the survey, not the submission.
    submission_report: str                      # 参赛方案文档（output_mode=submission 时由 survey 改写而来；survey 模式为空串）

    # --- Step 12 closed loop: fact-check → regenerate → re-check ---
    # Previously fact_check only APPENDED an issue log to the draft and the
    # pipeline ended, so a report could ship with "11 条待确认项" still inside
    # it. The graph now routes back to report_generation while `needs_revision`
    # is True, up to `fact_check.max_revisions` times.
    fact_check_issues: list[str]                # issues found in the latest pass
    fact_check_revision: int                    # regeneration rounds used so far
    fact_check_status: str                      # clean | revised | unresolved
    needs_revision: bool                        # router flag
    # Per-claim machine-checkable results and human-readable correction
    # suggestions returned by the fact_check node. Declared as state channels
    # so LangGraph propagates them instead of silently dropping them
    # (same trap class as the old _structure_property_md bug).
    fact_check_checks: list[dict]               # structured check results
    fact_check_corrections: list[str]           # human-readable correction suggestions

    # --- Step 8.5: structure-property analysis (optional) ---
    structure_property_result: dict
    # CRITICAL: this key MUST be declared as a state channel. node_structure_property
    # returns it and report_generation reads it; without declaration LangGraph's
    # state schema silently drops the undeclared key, so the 构效关系 chapter
    # never reaches the report (observed 2026-09-11: pkl had the content, report
    # did not). Declaring it lets the channel propagate across the graph.
    _structure_property_md: str

    # --- Step 10.5: hypothesis cross-check against external / offline corpus ---
    hypothesis_cross_checks: list[dict]

    # --- Metadata ---
    config: dict                                # workflow configuration
    error: Optional[str]                        # error message if any
    step_log: list[dict]                        # audit log of each step
