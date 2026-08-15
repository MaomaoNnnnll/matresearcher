"""Vector store using ChromaDB for semantic search.

ChromaDB is a local vector database (Apache-2.0) that can be swapped for
Milvus/Qdrant in production. This wrapper provides a clean interface.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

console = Console()

# Project root (independent of CWD): F:/projects/matresearcher
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class VectorStore:
    """ChromaDB-based vector store for literature and knowledge records."""

    def __init__(self, db_path: str | None = None):
        # Default: absolute path under the project root (never CWD-relative —
        # a relative default used to split data between ./data and src/data).
        self.db_path = db_path or os.getenv(
            "CHROMA_DB_PATH", str(PROJECT_ROOT / "data" / "chroma")
        )
        self._client = None
        self._collections: dict[str, Any] = {}

    def _get_client(self):
        """Lazy-load ChromaDB client."""
        if self._client is not None:
            return self._client
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.db_path)
            console.print(f"[green]ChromaDB initialized at {self.db_path}[/green]")
        except ImportError:
            raise RuntimeError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        return self._client

    def _get_collection(self, name: str):
        """Get or create a collection."""
        if name not in self._collections:
            client = self._get_client()
            self._collections[name] = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def add(
        self,
        collection: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ):
        """Add documents to a collection."""
        col = self._get_collection(collection)
        col.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(ids),
        )

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Query a collection by embedding."""
        col = self._get_collection(collection)
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        # Flatten results
        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return items

    def delete_collection(self, name: str):
        """Delete a collection."""
        client = self._get_client()
        client.delete_collection(name=name)
        self._collections.pop(name, None)

    def count(self, collection: str) -> int:
        """Count items in a collection."""
        col = self._get_collection(collection)
        return col.count()
