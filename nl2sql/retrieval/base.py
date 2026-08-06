"""Core retrieval types shared by every retriever implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class DocumentKind(str, Enum):
    """What part of the Knowledge Base a retrievable document came from."""

    TABLE = "table"
    RELATIONSHIP = "relationship"
    RULE = "rule"
    GLOSSARY = "glossary"
    EXAMPLE = "example"


@dataclass(frozen=True, slots=True)
class Document:
    """An indexed, retrievable unit of Knowledge Base content."""

    id: str
    kind: DocumentKind
    text: str
    tables: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject documents that could never be retrieved."""
        if not self.id:
            raise ValueError("Document id must not be empty")
        if not self.text.strip():
            raise ValueError(f"Document {self.id!r} has empty text")


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    """A document together with its relevance score for one query."""

    document: Document
    score: float
    retriever: str = "hybrid"

    @property
    def id(self) -> str:
        """Identifier of the underlying document."""
        return self.document.id

    @property
    def kind(self) -> DocumentKind:
        """Kind of the underlying document."""
        return self.document.kind


@runtime_checkable
class Retriever(Protocol):
    """Anything that can rank documents against a natural-language query."""

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        """Return up to ``top_k`` documents ranked by relevance to ``query``."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Anything that can index embedded documents and search them by similarity."""

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        ...

    def index(self, documents: list[Document]) -> None:
        """Embed and store ``documents``, replacing any previous index."""
        ...

    def search(self, query: str, top_k: int) -> list[ScoredDocument]:
        """Return the ``top_k`` documents most similar to ``query``."""
        ...
