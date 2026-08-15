"""Data models for MatResearcher."""
from .literature import Literature, LiteratureMetadata, ParsedDocument
from .knowledge import KnowledgeRecord, NormalizedRecord, FusedKnowledgeTable
from .gap import ResearchGap, GapScore, ConflictItem, MissingItem

__all__ = [
    "Literature", "LiteratureMetadata", "ParsedDocument",
    "KnowledgeRecord", "NormalizedRecord", "FusedKnowledgeTable",
    "ResearchGap", "GapScore", "ConflictItem", "MissingItem",
]
