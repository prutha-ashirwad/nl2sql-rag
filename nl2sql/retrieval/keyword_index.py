"""BM25 keyword index for the sparse half of the retrieval pipeline."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from nl2sql.exceptions import RetrievalError
from nl2sql.logging_config import get_logger
from nl2sql.retrieval.base import Document, ScoredDocument
from nl2sql.retrieval.text import tokenize

logger = get_logger(__name__)

# Okapi BM25 term-frequency saturation and length-normalisation defaults.
_K1 = 1.5
_B = 0.75


class BM25Index:
    """Okapi BM25 ranking over Knowledge Base documents."""

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_lengths: list[int] = []
        self._inverted_index: dict[str, set[int]] = defaultdict(set)
        self._inverse_document_frequency: dict[str, float] = {}
        self._average_length: float = 0.0

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)

    def index(self, documents: list[Document]) -> None:
        """Tokenise and index ``documents``, replacing any previous index."""
        if not documents:
            raise RetrievalError("Cannot build a keyword index from zero documents")

        self._documents = list(documents)
        self._term_frequencies = []
        self._document_lengths = []
        self._inverted_index = defaultdict(set)

        for position, document in enumerate(documents):
            tokens = tokenize(document.text)
            frequencies = Counter(tokens)

            self._term_frequencies.append(frequencies)
            self._document_lengths.append(len(tokens))
            for token in frequencies:
                self._inverted_index[token].add(position)

        total_documents = len(documents)
        self._average_length = sum(self._document_lengths) / total_documents
        self._inverse_document_frequency = {
            token: math.log(
                1.0
                + (total_documents - len(positions) + 0.5) / (len(positions) + 0.5)
            )
            for token, positions in self._inverted_index.items()
        }

        logger.debug(
            "Keyword index built: %d documents, %d unique terms",
            total_documents,
            len(self._inverted_index),
        )

    def search(self, query: str, top_k: int) -> list[ScoredDocument]:
        """Return the ``top_k`` documents best matching ``query``."""
        if not self._documents:
            raise RetrievalError("Keyword index queried before it was indexed")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)

        for token in query_tokens:
            positions = self._inverted_index.get(token)
            if not positions:
                continue

            idf = self._inverse_document_frequency[token]
            for position in positions:
                frequency = self._term_frequencies[position][token]
                length_ratio = self._document_lengths[position] / self._average_length
                numerator = frequency * (_K1 + 1.0)
                denominator = frequency + _K1 * (1.0 - _B + _B * length_ratio)
                scores[position] += idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        return [
            ScoredDocument(
                document=self._documents[position],
                score=score,
                retriever="keyword",
            )
            for position, score in ranked[:top_k]
            if score > 0.0
        ]
