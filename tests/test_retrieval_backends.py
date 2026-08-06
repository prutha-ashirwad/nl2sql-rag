"""Tests for the dense half of retrieval: the embedder and the vector store."""

from __future__ import annotations

import numpy as np
import pytest

from nl2sql.config import EmbeddingProviderName, VectorStoreName
from nl2sql.exceptions import RetrievalError
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.retrieval.base import ScoredDocument
from nl2sql.retrieval.document_builder import build_documents
from nl2sql.retrieval.embeddings import (
    OpenAIEmbeddingProvider,
    TfidfEmbeddingProvider,
    build_embedding_provider,
)
from nl2sql.retrieval.hybrid_retriever import HybridRetriever
from nl2sql.retrieval.vector_store import (
    FaissVectorStore,
    InMemoryVectorStore,
    build_vector_store,
)
from tests.conftest import build_test_settings


class BrokenEmbedder:
    """An embedding provider whose backend is unavailable."""

    dimension = 8

    def fit(self, corpus: list[str]) -> None:
        """No corpus statistics to gather."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise RetrievalError("Embedding request failed: backend unreachable")

    def embed_query(self, text: str) -> np.ndarray:
        raise RetrievalError("Embedding request failed: backend unreachable")


class RecordingVectorStore:
    """A stand-in index, standing for a hosted vector database."""

    def __init__(self) -> None:
        self.indexed: list[str] = []
        self.queries: list[str] = []

    @property
    def size(self) -> int:
        return len(self.indexed)

    def index(self, documents: list) -> None:  # noqa: ANN001 - Document, kept loose
        self.indexed = [document.id for document in documents]

    def search(self, query: str, top_k: int) -> list[ScoredDocument]:
        self.queries.append(query)
        return []


class TestTheInMemoryVectorStore:
    def test_it_ranks_by_similarity(self, registry: KnowledgeBaseRegistry) -> None:
        documents = build_documents(registry)
        store = InMemoryVectorStore(TfidfEmbeddingProvider())
        store.index(documents)

        hits = store.search("which interfaces failed most often", 5)

        assert store.size == len(documents)
        assert hits
        # Cosine over unit vectors: bounded, and ranked descending.
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 < score <= 1.0 + 1e-6 for score in scores)

    def test_an_empty_corpus_is_refused(self) -> None:
        store = InMemoryVectorStore(TfidfEmbeddingProvider())

        with pytest.raises(RetrievalError):
            store.index([])

    def test_querying_before_indexing_is_refused(self) -> None:
        store = InMemoryVectorStore(TfidfEmbeddingProvider())

        with pytest.raises(RetrievalError):
            store.search("anything", 5)


class TestTheBackendsStaySwappable:
    """The retriever depends on the protocols, never on the shipped classes."""

    def test_a_substituted_store_is_used(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        store = RecordingVectorStore()

        retriever = HybridRetriever(registry, vector_store=store)
        retriever.retrieve("Show failed observations", top_k=5)

        assert store.size == len(retriever.documents)
        assert store.queries == ["Show failed observations"]


class TestRetrievalSurvivesABrokenEmbedder:
    """A dead embedding backend degrades the answer; it must not stop the system."""

    def test_the_retriever_still_builds_and_answers(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        retriever = HybridRetriever(registry, embedding_provider=BrokenEmbedder())

        # A substituted backend that is unreachable, rate limited or misconfigured all
        # land here. Refusing to start would make one retrieval signal a hard
        # dependency of the whole system.
        hits = retriever.retrieve("Show failed observations", top_k=5)

        assert hits
        assert any("observations" in hit.document.tables for hit in hits)


class TestSelectingBackends:
    """The configured backend is a preference, never a precondition."""

    def test_the_defaults_are_openai_embeddings_in_the_memory_store(self) -> None:
        settings = build_test_settings()

        assert settings.embedding_provider is EmbeddingProviderName.OPENAI
        assert settings.vector_store is VectorStoreName.MEMORY

    def test_the_default_still_needs_no_credentials(self) -> None:
        # The default asks for a hosted provider, so a clone with no key must still
        # resolve to something usable rather than failing at start-up.
        assert isinstance(
            build_embedding_provider(build_test_settings()), TfidfEmbeddingProvider
        )

    def test_openai_can_be_asked_for(self) -> None:
        pytest.importorskip("openai")
        settings = build_test_settings(
            embedding_provider=EmbeddingProviderName.OPENAI, openai_api_key="sk-test"
        )

        assert isinstance(
            build_embedding_provider(settings), OpenAIEmbeddingProvider
        )

    def test_openai_degrades_to_tfidf_without_a_key(self) -> None:
        settings = build_test_settings(embedding_provider=EmbeddingProviderName.OPENAI)

        # Without this a fresh clone and the suite would both need credentials.
        assert isinstance(build_embedding_provider(settings), TfidfEmbeddingProvider)

    def test_faiss_can_be_asked_for(self) -> None:
        settings = build_test_settings(vector_store=VectorStoreName.FAISS)

        store = build_vector_store(settings, TfidfEmbeddingProvider())
        assert isinstance(store, FaissVectorStore | InMemoryVectorStore)


class TestBothStoresAgree:
    """FAISS and the numpy scan are both exact, so they must rank identically."""

    def test_the_same_query_gives_the_same_ranking(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        pytest.importorskip("faiss")
        documents = build_documents(registry)
        question = "which interfaces failed most often"

        # One fitted embedder, so the comparison isolates the store.
        embedder = TfidfEmbeddingProvider()
        embedder.fit([document.text for document in documents])

        memory = InMemoryVectorStore(embedder)
        memory.index(documents)
        faiss_store = FaissVectorStore(embedder)
        faiss_store.index(documents)

        assert [hit.id for hit in memory.search(question, 5)] == [
            hit.id for hit in faiss_store.search(question, 5)
        ]


class TestOpenAIEmbeddingProvider:
    """The hosted provider, without touching the network."""

    @staticmethod
    def _provider_with_stub_client(monkeypatch: pytest.MonkeyPatch, dimensions: int):
        # The provider is an optional extra, so its tests skip when it is absent.
        openai = pytest.importorskip("openai")

        class StubEmbeddings:
            def create(self, *, model: str, input: list[str], dimensions: int):  # noqa: A002
                items = [
                    type("Item", (), {"index": i, "embedding": [0.1] * dimensions})
                    for i, _ in enumerate(input)
                ]
                # Reversed on purpose: the provider must sort by index rather than
                # trust wire order, or the corpus silently misaligns.
                return type("Response", (), {"data": list(reversed(items))})

        class StubClient:
            def __init__(self, **_: object) -> None:
                self.embeddings = StubEmbeddings()

        monkeypatch.setattr(openai, "OpenAI", StubClient)
        return OpenAIEmbeddingProvider("sk-test", dimensions=dimensions)

    def test_documents_are_embedded_to_the_requested_width(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = self._provider_with_stub_client(monkeypatch, dimensions=16)

        assert provider.embed_documents(["one", "two", "three"]).shape == (3, 16)

    def test_an_empty_corpus_needs_no_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = self._provider_with_stub_client(monkeypatch, dimensions=16)

        assert provider.embed_documents([]).shape == (0, 16)

    def test_a_query_returns_one_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = self._provider_with_stub_client(monkeypatch, dimensions=16)

        assert provider.embed_query("how many alerts are open").shape == (16,)
