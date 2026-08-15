"""Reranker model wrapper.

Uses sentence-transformers (BAAI/bge-reranker-v2-m3) for cross-encoder reranking.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


class RerankerModel:
    """Cross-encoder reranker for document relevance scoring."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        self._model = None

    def _load(self):
        """Lazy-load the model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score document relevance to a query.

        Returns a list of relevance scores (higher = more relevant).
        """
        self._load()
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)
        # Normalize to 0-1 range using sigmoid
        return [float(1 / (1 + np.exp(-s))) for s in scores]

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 20,
        min_score: float = 0.5,
    ) -> list[tuple[int, float]]:
        """Rerank documents and return top-k with scores.

        Returns list of (original_index, score) tuples, sorted by score descending.
        """
        scores = self.score(query, documents)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [(idx, sc) for idx, sc in indexed[:top_k] if sc >= min_score]
