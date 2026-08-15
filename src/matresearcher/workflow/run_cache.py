"""Run-level result cache for checkpoint & resume support (v1.5).

Design:
- Node-level cache: each node's returned state-patch is pickled to
  <run_dir>/cache/node_<name>.pkl right after the node succeeds.
  On resume, a node whose cache exists is skipped entirely — no LLM
  calls, no Sciverse calls, zero token cost.
- Document-level cache: KnowledgeExtractionAgent persists per-paper
  results (<run_dir>/cache/doc_<lit_id>_<depth>.pkl) so the most
  expensive stage can resume mid-way after a process crash.
- MANIFEST.json records completed nodes + timestamps for inspection.
- All reads are defensive: corrupt / legacy pickles return None and
  the caller falls back to re-execution (safe degradation).
- Atomic writes (tmp + os.replace): a crash never leaves a half-written
  pkl that would later be trusted as a valid cache.
"""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Topological execution order of the LangGraph pipeline (linear + coverage back-edge).
# Used by invalidate_from() to decide which cached nodes depend on a changed input.
PIPELINE_ORDER = [
    "task_planning",
    "literature_search",
    "coverage_check",
    "llm_prefilter",
    "literature_filter",
    "pdf_parsing",
    "knowledge_extraction",
    "knowledge_fusion",
    "gap_generation",
    "evidence_verification",
    "report_generation",
    "fact_check",
]


class RunCache:
    """File-backed pickle cache scoped to a single run directory."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.cache_dir = self.run_dir / "cache"
        self.manifest_path = self.run_dir / "MANIFEST.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ─── Introspection ───

    @property
    def run_id(self) -> str:
        """The run identifier (== run directory name)."""
        return self.run_dir.name

    def completed_nodes(self) -> list[str]:
        """Node names whose node_*.pkl cache exists, in pipeline order."""
        names = set()
        for f in self.cache_dir.glob("node_*.pkl"):
            name = f.stem[len("node_"):]
            if name in PIPELINE_ORDER:
                names.add(name)
        return [n for n in PIPELINE_ORDER if n in names]

    def has_document_cache(self) -> bool:
        return any(self.cache_dir.glob("doc_*.pkl"))

    # ─── Node-level cache ───

    def node_path(self, name: str) -> Path:
        return self.cache_dir / f"node_{name}.pkl"

    def has_node(self, name: str) -> bool:
        return self.node_path(name).exists()

    def get_node(self, name: str) -> Optional[dict]:
        """Load a node's cached state-patch. Returns None on any failure
        (missing file, corrupt pickle, legacy schema) → caller re-executes."""
        return self._load_pickle(self.node_path(name))

    def save_node(self, name: str, patch: dict) -> None:
        """Atomically persist a node's returned state-patch."""
        self._dump_pickle(patch, self.node_path(name))
        self._write_manifest()

    # ─── Document-level cache (knowledge extraction) ───

    def doc_path(self, lit_id: str, depth: str) -> Path:
        return self.cache_dir / f"doc_{lit_id}_{depth}.pkl"

    def has_doc(self, lit_id: str, depth: str) -> bool:
        return self.doc_path(lit_id, depth).exists()

    def get_doc(self, lit_id: str, depth: str) -> Optional[dict]:
        """Load cached per-paper extraction result:
        {"verified": [KnowledgeRecord...], "anomalies": [KnowledgeRecord...]}"""
        return self._load_pickle(self.doc_path(lit_id, depth))

    def save_doc(self, lit_id: str, depth: str, verified: list, anomalies: list) -> None:
        self._dump_pickle({"verified": verified, "anomalies": anomalies},
                          self.doc_path(lit_id, depth))

    # ─── Invalidation (coverage back-edge) ───

    def invalidate_from(self, node_name: str) -> list[str]:
        """Remove node_name's cache and every node after it in the pipeline,
        plus ALL document-level caches (they depend on the filtered literature).

        Called when the search strategy is refined (coverage retry): the
        literature set changes, so every downstream result is stale.

        Returns the list of removed cache file names.
        """
        try:
            idx = PIPELINE_ORDER.index(node_name)
        except ValueError:
            return []
        removed: list[str] = []
        for f in self.cache_dir.glob("node_*.pkl"):
            name = f.stem[len("node_"):]
            if name in PIPELINE_ORDER and PIPELINE_ORDER.index(name) >= idx:
                try:
                    f.unlink(missing_ok=True)
                    removed.append(f.name)
                except OSError:
                    pass
        for f in self.cache_dir.glob("doc_*.pkl"):
            try:
                f.unlink(missing_ok=True)
                removed.append(f.name)
            except OSError:
                pass
        self._write_manifest()
        return removed

    # ─── Low-level helpers ───

    def _load_pickle(self, path: Path) -> Any:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            # Corrupt / legacy / missing — caller treats None as "no cache".
            return None

    def _dump_pickle(self, obj: Any, path: Path) -> None:
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def _write_manifest(self) -> None:
        manifest = {
            "run_id": self.run_id,
            "updated_at": datetime.now().isoformat(),
            "completed_nodes": self.completed_nodes(),
            "document_cache_count": len(list(self.cache_dir.glob("doc_*.pkl"))),
        }
        try:
            tmp = self.manifest_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.manifest_path)
        except OSError:
            pass  # manifest is best-effort; never block the pipeline on it
