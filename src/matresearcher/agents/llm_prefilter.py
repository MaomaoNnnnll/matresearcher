"""LLM Prefilter Agent — Three-way classification before Reranker.

Inserted between coverage_check and literature_filter in the workflow.
Uses LLM to rapidly classify candidate literature by title+abstract into
three categories (relevant / partial / irrelevant), cutting the Reranker
workload by ~50% while retaining borderline papers for a second pass.

Strategy:
- Batch papers in groups (configurable batch_size, default 20) to stay within token limits
- Keep all "relevant" + "partial" papers; discard only "irrelevant"
- Log per-batch statistics for traceability
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..state import WorkflowState
from .base import BaseAgent

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "config" / "prompts"

# ── Prefilter prompt template (loaded from file, with fallback) ──
_PREFILTER_TXT = PROMPTS_DIR / "llm_prefilter.txt"
if _PREFILTER_TXT.exists():
    _PREFILTER_PROMPT = _PREFILTER_TXT.read_text(encoding="utf-8")
else:
    _PREFILTER_PROMPT = ""  # error handled at runtime

# ── Default parameters ──
MAX_TITLE_LEN = 200      # truncate title
MAX_ABSTRACT_LEN = 500   # truncate abstract per paper


class LLMPrefilterAgent(BaseAgent):
    """Three-way LLM classifier to reduce Reranker workload.

    Reads candidate_literature from state, runs batched LLM classification,
    returns filtered candidates (relevant + partial only).
    """

    name = "llm_prefilter"
    role = "LLM 预筛 Agent"

    def __init__(self, llm=None, config=None, log_dir=None):
        super().__init__(llm, config, log_dir=log_dir)

    async def run(self, state: WorkflowState) -> dict:
        candidates = state.get("candidate_literature", [])
        if not candidates:
            self.log("No candidates to prefilter", "yellow")
            return {"candidate_literature": candidates}

        if not self.llm:
            self.log("LLM unavailable, skipping prefilter", "yellow")
            return {"candidate_literature": candidates}

        if not _PREFILTER_PROMPT:
            self.log("Prefilter prompt file missing, skipping", "yellow")
            return {"candidate_literature": candidates}

        batch_size = self.config.get("batch_size", 20)
        n_total = len(candidates)
        self.log(f"Prefilter: {n_total} candidates → batching in groups of {batch_size}")

        # Batch by batch_size
        all_labels: dict[str, str] = {}
        for batch_start in range(0, n_total, batch_size):
            batch_end = min(batch_start + batch_size, n_total)
            batch = candidates[batch_start:batch_end]
            labels = await self._classify_batch(batch, batch_start)
            all_labels.update(labels)

        # Split by classification
        relevant = []
        partial = []
        irrelevant = []
        for i, lit in enumerate(candidates):
            key = str(i)
            label = all_labels.get(key, "partial")  # default to partial on parse error
            if label == "relevant":
                relevant.append(lit)
            elif label == "partial":
                partial.append(lit)
            else:
                irrelevant.append(lit)

        kept = relevant + partial  # preserve order: relevant first, then partial
        self.log(
            f"Prefilter complete: relevant={len(relevant)}, partial={len(partial)}, "
            f"irrelevant={len(irrelevant)}, kept={len(kept)}/{n_total}"
        )

        return {"candidate_literature": kept}

    async def _classify_batch(
        self, batch: list, batch_start: int
    ) -> dict[str, str]:
        """Classify one batch of papers via LLM.

        Returns dict mapping paper index (str) → label.
        """
        # Build compact paper text
        lines = []
        for offset, lit in enumerate(batch):
            idx = batch_start + offset
            meta = lit.metadata
            title = (meta.title or "Untitled")[:MAX_TITLE_LEN]
            abstract = (meta.abstract or "")[:MAX_ABSTRACT_LEN]
            lines.append(f'[{idx}] "{title}"\n    Abstract: {abstract or "N/A"}')
        papers_text = "\n\n".join(lines)

        try:
            result = await self.llm.complete_json(
                "You are a materials science researcher specializing in solid-state batteries.",
                _PREFILTER_PROMPT.replace("{papers_text}", papers_text),
                max_tokens=4096,
                temperature=0.3,
                stage=self.name,
            )
        except Exception as e:
            self.log(
                f"LLM classification failed for batch {batch_start}: {e} "
                f"(prompt_len={len(_PREFILTER_PROMPT)}, papers_len={len(papers_text)})",
                "yellow",
            )
            # On failure, default all papers to "partial" (keep them)
            return {str(batch_start + i): "partial" for i in range(len(batch))}

        # Parse result — handle dict, str, and unexpected types
        labels: dict[str, str] = {}
        if isinstance(result, dict):
            for k, v in result.items():
                key = str(k)
                val = str(v).lower().strip()
                if val in ("relevant", "partial", "irrelevant"):
                    labels[key] = val
                else:
                    labels[key] = "partial"  # unknown label → keep
        elif isinstance(result, str):
            self.log(
                f"LLM returned string instead of dict: {result[:120]}", "yellow"
            )
            return {str(batch_start + i): "partial" for i in range(len(batch))}
        else:
            self.log(f"Unexpected LLM output format: {type(result)}", "yellow")
            return {str(batch_start + i): "partial" for i in range(len(batch))}

        return labels
