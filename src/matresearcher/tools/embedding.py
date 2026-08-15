"""Embedding model wrapper.

Uses sentence-transformers (BAAI/bge-m3) for local embedding,
or OpenAI-compatible API for remote embedding.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


class EmbeddingModel:
    """Text embedding model wrapper."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self._model = None

    def _load(self):
        """Lazy-load the model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a vector."""
        self._load()
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts."""
        self._load()
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        # bge-m3 dimension
        return 1024

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return float(np.dot(vec1, vec2))
