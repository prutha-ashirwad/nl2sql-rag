"""Embeddings for the dense half of the retrieval pipeline.

Two providers behind one protocol. TF-IDF is the default: local, deterministic, and
strong on a corpus this size, where relevance is mostly vocabulary overlap. Hosted
embeddings capture wording the schema never uses and are worth their network cost on
a larger Knowledge Base; set ``EMBEDDING_PROVIDER=openai`` to switch.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol, runtime_checkable

import numpy as np

from nl2sql.config import EmbeddingProviderName, Settings
from nl2sql.exceptions import RetrievalError
from nl2sql.logging_config import get_logger
from nl2sql.retrieval.text import tokenize

logger = get_logger(__name__)

# The API caps how much one call may carry; a few hundred documents fit in these.
_OPENAI_BATCH_SIZE = 128

# Questions repeat heavily in a session, and a few hundred float arrays cost little.
_QUERY_CACHE_SIZE = 256


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into dense vectors that can be compared with cosine similarity."""

    @property
    def dimension(self) -> int:
        """Length of the vectors this provider emits."""
        ...

    def fit(self, corpus: list[str]) -> None:
        """Prepare the provider for ``corpus``; a no-op for stateless providers."""
        ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents into a ``(len(texts), dimension)`` matrix."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query into a ``(dimension,)`` vector."""
        ...


class TfidfEmbeddingProvider:
    """TF-IDF vectoriser producing L2-normalised vectors.

    Vectors are unit length, so a dot product is exactly the cosine similarity.
    """

    def __init__(self, *, min_document_frequency: int = 1) -> None:
        self._min_document_frequency = min_document_frequency
        self._vocabulary: dict[str, int] = {}
        self._inverse_document_frequency: np.ndarray = np.zeros(0, dtype=np.float32)
        self._is_fitted = False

    @property
    def dimension(self) -> int:
        """Size of the learned vocabulary."""
        return len(self._vocabulary)

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has been called."""
        return self._is_fitted

    def fit(self, corpus: list[str]) -> None:
        """Learn the vocabulary and inverse document frequencies from ``corpus``."""
        document_frequency: Counter[str] = Counter()
        for text in corpus:
            document_frequency.update(set(tokenize(text)))

        self._vocabulary = {
            token: index
            for index, (token, frequency) in enumerate(
                sorted(document_frequency.items())
            )
            if frequency >= self._min_document_frequency
        }

        total_documents = max(len(corpus), 1)
        idf = np.zeros(len(self._vocabulary), dtype=np.float32)
        for token, index in self._vocabulary.items():
            # Smoothing keeps a term present in every document weighted above zero.
            idf[index] = math.log(
                (1 + total_documents) / (1 + document_frequency[token])
            ) + 1.0

        self._inverse_document_frequency = idf
        self._is_fitted = True

    def _vectorise(self, text: str) -> np.ndarray:
        """Project ``text`` onto the learned vocabulary as a unit vector."""
        vector = np.zeros(len(self._vocabulary), dtype=np.float32)
        if not self._vocabulary:
            return vector

        counts = Counter(tokenize(text))
        for token, count in counts.items():
            index = self._vocabulary.get(token)
            if index is None:
                continue
            # Sublinear term frequency dampens tokens repeated inside long documents.
            vector[index] = (1.0 + math.log(count)) * self._inverse_document_frequency[
                index
            ]

        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0.0 else vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents, fitting on them if not already fitted."""
        if not self._is_fitted:
            self.fit(texts)
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._vectorise(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query using the vocabulary learned at fit time."""
        if not self._is_fitted:
            raise RuntimeError("TfidfEmbeddingProvider.fit must be called before use")
        return self._vectorise(text)


class OpenAIEmbeddingProvider:
    """Hosted embeddings, returned L2-normalised so a dot product is the cosine.

    The corpus is embedded once when the retriever is built; per question only the
    question itself is sent.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        timeout_seconds: float = 30.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RetrievalError(
                "The openai package is required for hosted embeddings. "
                "Install it with: pip install -e '.[openai]'"
            ) from exc

        # max_retries=0: the caller degrades to TF-IDF, and retrying a slow request
        # turns a bounded timeout into an unbounded wait.
        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        self._model = model
        self._dimensions = dimensions
        self._query_cache: dict[str, np.ndarray] = {}

    @property
    def dimension(self) -> int:
        """Length of the vectors the configured model emits."""
        return self._dimensions

    def fit(self, corpus: list[str]) -> None:
        """No-op: a hosted model needs no corpus statistics."""

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` in batches and return them as one matrix."""
        vectors: list[list[float]] = []

        for start in range(0, len(texts), _OPENAI_BATCH_SIZE):
            batch = texts[start : start + _OPENAI_BATCH_SIZE]
            try:
                response = self._client.embeddings.create(
                    model=self._model, input=batch, dimensions=self._dimensions
                )
            except Exception as exc:  # noqa: BLE001 - vendor SDK raises broadly
                raise RetrievalError(f"Embedding request failed: {exc}") from exc

            # Sorted by index rather than trusting wire order; a misordered corpus
            # would silently misalign every document with someone else's vector.
            vectors.extend(
                item.embedding for item in sorted(response.data, key=lambda x: x.index)
            )

        return np.array(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed the corpus."""
        if not texts:
            return np.zeros((0, self._dimensions), dtype=np.float32)

        matrix = self._embed(texts)
        logger.info("Embedded %d document(s) with %s", len(texts), self._model)
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question, reusing the vector if it has been seen before."""
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached

        vector = self._embed([text])[0]

        if len(self._query_cache) >= _QUERY_CACHE_SIZE:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[text] = vector

        return vector


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Resolve the configured provider, falling back to TF-IDF when it cannot be built.

    Embeddings are a quality knob, not a precondition: no key, no SDK, or a client
    that refuses to construct all degrade rather than stopping start-up.
    """
    if settings.embedding_provider is EmbeddingProviderName.TFIDF:
        return TfidfEmbeddingProvider()

    if not settings.openai_api_key:
        logger.info("No OPENAI_API_KEY set; using local TF-IDF embeddings instead")
        return TfidfEmbeddingProvider()

    try:
        return OpenAIEmbeddingProvider(
            settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    except RetrievalError as exc:
        logger.warning("Falling back to TF-IDF embeddings: %s", exc)
        return TfidfEmbeddingProvider()
