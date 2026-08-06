"""Hybrid retriever fusing keyword, vector and lexicon signals with RRF."""

from __future__ import annotations

from collections import defaultdict

from nl2sql.config import Settings
from nl2sql.exceptions import RetrievalError
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.logging_config import get_logger
from nl2sql.retrieval.base import Document, DocumentKind, ScoredDocument, VectorStore
from nl2sql.retrieval.document_builder import build_documents
from nl2sql.retrieval.embeddings import (
    EmbeddingProvider,
    TfidfEmbeddingProvider,
    build_embedding_provider,
)
from nl2sql.retrieval.keyword_index import BM25Index
from nl2sql.retrieval.text import extract_ngrams
from nl2sql.retrieval.vector_store import InMemoryVectorStore, build_vector_store

logger = get_logger(__name__)

# Rank-fusion smoothing constant, from the original RRF paper.
_RRF_K = 60

# Candidates each signal contributes before fusion.
_CANDIDATE_MULTIPLIER = 4
_MIN_CANDIDATES = 20


class HybridRetriever:
    """Ranks Knowledge Base documents against a natural-language question."""

    def __init__(
        self,
        registry: KnowledgeBaseRegistry,
        *,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        lexical_weight: float = 0.5,
    ) -> None:
        """Build the indexes.

        ``settings`` resolves the configured backends; omitting it gives the local
        TF-IDF pair, which needs no credentials. Passing either backend explicitly
        overrides it. ``lexical_weight`` weights the keyword signal in ``[0, 1]``;
        the vector signal receives the remainder.
        """
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between 0 and 1")

        self._registry = registry
        self._lexical_weight = lexical_weight
        self._vector_weight = 1.0 - lexical_weight

        self._documents = build_documents(registry)
        self._documents_by_id = {document.id: document for document in self._documents}

        self._keyword_index = BM25Index()
        self._keyword_index.index(self._documents)

        embedder = embedding_provider or (
            build_embedding_provider(settings) if settings else TfidfEmbeddingProvider()
        )
        self._vector_store = vector_store or (
            build_vector_store(settings, embedder)
            if settings
            else InMemoryVectorStore(embedder)
        )
        self._index_documents()

        logger.info("Hybrid retriever ready over %d documents", len(self._documents))

    def _index_documents(self) -> None:
        """Index the corpus, retrying locally if the configured backend cannot serve it.

        An expired key, a rate limit or a missing install should degrade the ranking,
        not stop the system from starting.
        """
        try:
            self._vector_store.index(self._documents)
            return
        except RetrievalError as exc:
            logger.warning("Embedding the Knowledge Base failed: %s", exc)

        self._vector_store = InMemoryVectorStore(TfidfEmbeddingProvider())
        self._vector_store.index(self._documents)
        logger.warning("Dense retrieval degraded to local TF-IDF embeddings")

    def _dense_candidates(self, query: str, limit: int) -> list[ScoredDocument]:
        """Rank by embedding similarity, contributing nothing if that is unavailable."""
        try:
            return self._vector_store.search(query, limit)
        except RetrievalError as exc:
            logger.warning("Dense retrieval unavailable for this question: %s", exc)
            return []

    @property
    def documents(self) -> list[Document]:
        """The full indexed corpus."""
        return list(self._documents)

    @property
    def backends(self) -> tuple[str, str]:
        """The embedder and store actually in use, after any fallback.

        A substituted store need not hold an embedder, so that half may be unknown.
        """
        embedder = getattr(self._vector_store, "_embedder", None)
        return (
            type(embedder).__name__ if embedder is not None else "unknown",
            type(self._vector_store).__name__,
        )

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        """Return up to ``top_k`` documents ranked by fused relevance."""
        if not query.strip():
            return []

        candidate_count = max(top_k * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATES)

        signals: list[tuple[list[ScoredDocument], float]] = [
            (self._keyword_index.search(query, candidate_count), self._lexical_weight),
            (self._dense_candidates(query, candidate_count), self._vector_weight),
            (self._lexicon_matches(query), 1.0),
        ]

        fused = self._reciprocal_rank_fusion(signals)
        return fused[:top_k]

    def _lexicon_matches(self, query: str) -> list[ScoredDocument]:
        """Match curated synonyms in the question onto their table documents.

        Longer phrases score higher: "failure count" is more specific than "count".
        """
        matched_tables: dict[str, float] = {}

        # Sorted so ranking does not depend on set iteration order.
        for ngram in sorted(extract_ngrams(query, max_size=3)):
            specificity = len(ngram.split())
            for match in self._registry.resolve_tables_for_phrase(ngram):
                matched_tables[match.table] = max(
                    matched_tables.get(match.table, 0.0), match.score * specificity
                )

        ordered = sorted(matched_tables.items(), key=lambda item: (-item[1], item[0]))

        results: list[ScoredDocument] = []
        for table_name, score in ordered:
            document = self._documents_by_id.get(f"table::{table_name}")
            if document is not None:
                results.append(
                    ScoredDocument(
                        document=document, score=score, retriever="lexicon"
                    )
                )

        return results

    @staticmethod
    def _reciprocal_rank_fusion(
        signals: list[tuple[list[ScoredDocument], float]],
    ) -> list[ScoredDocument]:
        """Fuse several ranked lists into one ordering."""
        fused_scores: dict[str, float] = defaultdict(float)
        documents: dict[str, Document] = {}
        contributors: dict[str, set[str]] = defaultdict(set)

        for ranked, weight in signals:
            if weight <= 0.0:
                continue
            for rank, scored in enumerate(ranked, start=1):
                fused_scores[scored.id] += weight / (_RRF_K + rank)
                documents[scored.id] = scored.document
                contributors[scored.id].add(scored.retriever)

        ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

        return [
            ScoredDocument(
                document=documents[document_id],
                score=score,
                retriever="+".join(sorted(contributors[document_id])),
            )
            for document_id, score in ordered
        ]

    def documents_of_kind(self, kind: DocumentKind) -> list[Document]:
        """Return every indexed document of a given kind."""
        return [document for document in self._documents if document.kind is kind]

    def document_by_id(self, document_id: str) -> Document | None:
        """Look up an indexed document by its identifier."""
        return self._documents_by_id.get(document_id)
