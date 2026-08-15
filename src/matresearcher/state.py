"""Workflow state definition for LangGraph.

This TypedDict is the shared state object passed through the entire
18-step pipeline (steps 1-16 + 4a + 7a).
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
    draft_report: str                           # raw markdown
    final_report: str                           # verified markdown

    # --- Metadata ---
    config: dict                                # workflow configuration
    error: Optional[str]                        # error message if any
    step_log: list[dict]                        # audit log of each step
