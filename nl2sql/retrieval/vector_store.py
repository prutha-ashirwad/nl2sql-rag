"""Vector stores for dense retrieval.

Both use inner product over L2-normalised vectors, so both compute the same
similarity and return the same ranking. ``MEMORY`` is an exhaustive numpy scan and
the default — exact, and sub-millisecond at Knowledge Base scale. ``FAISS`` is what
the same code path scales onto; set ``VECTOR_STORE=faiss`` to switch.
"""

from __future__ import annotations

import numpy as np

from nl2sql.config import Settings, VectorStoreName
from nl2sql.exceptions import RetrievalError
from nl2sql.logging_config import get_logger
from nl2sql.retrieval.base import Document, ScoredDocument
from nl2sql.retrieval.embeddings import EmbeddingProvider

logger = get_logger(__name__)


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so an inner product is a cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0.0, norms, 1.0)


class InMemoryVectorStore:
    """Exact nearest-neighbour search over embedded Knowledge Base documents."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedder = embedding_provider
        self._documents: list[Document] = []
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)

    def index(self, documents: list[Document]) -> None:
        """Embed and store ``documents``, replacing any previous index."""
        if not documents:
            raise RetrievalError("Cannot build a vector index from zero documents")

        self._documents = list(documents)
        self._matrix = self._embedder.embed_documents(
            [document.text for document in documents]
        )
        logger.debug(
            "Vector index built: %d documents, %d dimensions",
            self._matrix.shape[0],
            self._matrix.shape[1] if self._matrix.ndim > 1 else 0,
        )

    def search(self, query: str, top_k: int) -> list[ScoredDocument]:
        """Return the ``top_k`` documents most similar to ``query``."""
        if not self._documents:
            raise RetrievalError("Vector store queried before it was indexed")

        query_vector = self._embedder.embed_query(query)
        if query_vector.size == 0 or self._matrix.size == 0:
            return []

        # Both sides unit length, so the dot product below is cosine similarity.
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm > 0.0:
            query_vector = query_vector / query_norm

        similarities = self._matrix @ query_vector

        limit = min(top_k, len(self._documents))
        top_indices = np.argpartition(-similarities, limit - 1)[:limit]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return [
            ScoredDocument(
                document=self._documents[int(index)],
                score=float(similarities[int(index)]),
                retriever="vector",
            )
            for index in top_indices
            if similarities[int(index)] > 0.0
        ]


class FaissVectorStore:
    """FAISS-backed nearest-neighbour search over embedded documents.

    Uses ``IndexFlatIP`` — exhaustive inner product — because it is exact. Swapping
    in ``IndexIVFFlat`` or ``IndexHNSWFlat`` is a one-line change here once the
    corpus is large enough for approximate search to buy anything.
    """

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._embedder = embedding_provider
        self._documents: list[Document] = []
        self._index = None

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)

    def index(self, documents: list[Document]) -> None:
        """Embed and index ``documents``, replacing any previous index."""
        if not documents:
            raise RetrievalError("Cannot build a vector index from zero documents")

        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RetrievalError(
                "faiss-cpu is required for the FAISS vector store. "
                "Install it with: pip install -e '.[faiss]', or set VECTOR_STORE=memory."
            ) from exc

        self._documents = list(documents)
        matrix = self._embedder.embed_documents(
            [document.text for document in documents]
        )
        if matrix.size == 0:
            raise RetrievalError("The embedding provider returned no vectors")

        matrix = _normalise_rows(np.ascontiguousarray(matrix, dtype=np.float32))
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

        logger.debug(
            "FAISS index built: %d documents, %d dimensions",
            self._index.ntotal,
            matrix.shape[1],
        )

    def search(self, query: str, top_k: int) -> list[ScoredDocument]:
        """Return the ``top_k`` documents most similar to ``query``."""
        if self._index is None:
            raise RetrievalError("Vector store queried before it was indexed")

        query_vector = self._embedder.embed_query(query)
        if query_vector.size == 0:
            return []

        query_matrix = _normalise_rows(
            np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        )
        scores, indices = self._index.search(
            query_matrix, min(top_k, len(self._documents))
        )

        return [
            ScoredDocument(
                document=self._documents[int(position)],
                score=float(score),
                retriever="vector",
            )
            # FAISS pads with -1 when fewer than top_k neighbours exist.
            for score, position in zip(scores[0], indices[0], strict=True)
            if position >= 0 and score > 0.0
        ]


def build_vector_store(
    settings: Settings, embedding_provider: EmbeddingProvider
) -> InMemoryVectorStore | FaissVectorStore:
    """Resolve the configured store, degrading to the in-memory one.

    A missing FAISS install is not fatal: both return the same rankings.
    """
    if settings.vector_store is VectorStoreName.MEMORY:
        return InMemoryVectorStore(embedding_provider)

    try:
        import faiss  # noqa: F401 - probing availability before committing to it
    except ImportError:
        logger.info("faiss-cpu is not installed; using the in-memory vector store")
        return InMemoryVectorStore(embedding_provider)

    return FaissVectorStore(embedding_provider)
