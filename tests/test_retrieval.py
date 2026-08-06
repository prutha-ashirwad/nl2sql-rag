"""Tests for the RAG retrieval pipeline."""

from __future__ import annotations

import pytest

from nl2sql.exceptions import RetrievalError
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.retrieval.base import Document, DocumentKind
from nl2sql.retrieval.context_builder import SchemaContextBuilder
from nl2sql.retrieval.document_builder import build_documents
from nl2sql.retrieval.embeddings import TfidfEmbeddingProvider
from nl2sql.retrieval.hybrid_retriever import HybridRetriever
from nl2sql.retrieval.keyword_index import BM25Index
from nl2sql.retrieval.text import extract_ngrams, tokenize
from nl2sql.retrieval.vector_store import InMemoryVectorStore


class TestTextProcessing:
    def test_splits_identifiers_into_parts(self) -> None:
        tokens = tokenize("observed_at")
        assert "observed_at" in tokens
        assert "observed" in tokens

    def test_drops_stop_words(self) -> None:
        assert "the" not in tokenize("the observations")

    def test_stems_plurals_together(self) -> None:
        assert tokenize("interfaces")[0] == tokenize("interface")[0]

    def test_extracts_multi_word_phrases(self) -> None:
        ngrams = extract_ngrams("show failed observations", max_size=3)
        assert "failed observations" in ngrams
        assert "show failed observations" in ngrams


class TestDocumentBuilder:
    def test_builds_one_document_per_entity(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        documents = build_documents(registry)
        expected = (
            len(registry.tables)
            + len(registry.relationships)
            + len(registry.rules)
            + len(registry.glossary)
            + len(registry.examples)
        )
        assert len(documents) == expected

    def test_table_documents_carry_column_detail(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        documents = {doc.id: doc for doc in build_documents(registry)}
        text = documents["table::observations"].text
        assert "observed_at" in text
        assert "FAILED" in text
        assert "Grain:" in text

    def test_rejects_empty_documents(self) -> None:
        with pytest.raises(ValueError, match="empty text"):
            Document(id="x", kind=DocumentKind.TABLE, text="   ")


class TestIndexes:
    def test_keyword_index_finds_exact_identifiers(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        index = BM25Index()
        index.index(build_documents(registry))

        results = index.search("observed_at", top_k=5)
        assert results
        # BM25's job is exact-token recall. Length normalisation means a short rule
        # can outrank the long table document, which is why the hybrid retriever
        # fuses this signal with the vector and lexicon ones rather than using it
        # alone.
        assert all("observed_at" in result.document.text for result in results)

    def test_keyword_index_scores_unmatched_queries_at_zero(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        index = BM25Index()
        index.index(build_documents(registry))
        assert index.search("zzzzqqqqxxxx", top_k=5) == []

    def test_vector_store_finds_paraphrases(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        store = InMemoryVectorStore(TfidfEmbeddingProvider())
        store.index(build_documents(registry))

        results = store.search("network port on a device", top_k=5)
        assert any(result.id == "table::interfaces" for result in results)

    def test_index_rejects_empty_corpus(self) -> None:
        with pytest.raises(RetrievalError):
            BM25Index().index([])

    def test_querying_an_unbuilt_index_fails_clearly(self) -> None:
        with pytest.raises(RetrievalError, match="before it was indexed"):
            BM25Index().search("anything", top_k=3)


class TestHybridRetriever:
    @pytest.mark.parametrize(
        ("question", "expected_table"),
        [
            ("Show all failed observations", "observations"),
            ("Which ports keep going down?", "interfaces"),
            ("List the data centres we operate", "sites"),
            ("Who acknowledged the alert?", "alerts"),
            ("What incidents are still open?", "incidents"),
        ],
    )
    def test_retrieves_the_relevant_table(
        self, retriever: HybridRetriever, question: str, expected_table: str
    ) -> None:
        results = retriever.retrieve(question, top_k=8)
        tables = {table for result in results for table in result.document.tables}
        assert expected_table in tables

    def test_returns_no_more_than_requested(
        self, retriever: HybridRetriever
    ) -> None:
        assert len(retriever.retrieve("failed observations", top_k=3)) <= 3

    def test_empty_question_returns_nothing(
        self, retriever: HybridRetriever
    ) -> None:
        assert retriever.retrieve("   ", top_k=5) == []

    def test_results_are_ordered_by_score(
        self, retriever: HybridRetriever
    ) -> None:
        scores = [item.score for item in retriever.retrieve("failed observations", 8)]
        assert scores == sorted(scores, reverse=True)

    def test_retrieval_is_deterministic(self, retriever: HybridRetriever) -> None:
        first = [item.id for item in retriever.retrieve("failure count by site", 8)]
        for _ in range(5):
            assert [
                item.id for item in retriever.retrieve("failure count by site", 8)
            ] == first


class TestContextBuilder:
    def test_pulls_in_tables_needed_to_complete_the_joins(
        self, context_builder: SchemaContextBuilder
    ) -> None:
        # The question names interfaces and environments; observations is the fact
        # table that connects them and must be retrieved even if unnamed.
        context = context_builder.build(
            "Show interfaces and environments with failures"
        )
        assert "observations" in context.table_names

    def test_anchors_on_the_fact_table(
        self, context_builder: SchemaContextBuilder
    ) -> None:
        context = context_builder.build("failed observations by environment")
        assert context.base_table == "observations"

    def test_always_includes_the_safety_rules(
        self, context_builder: SchemaContextBuilder
    ) -> None:
        context = context_builder.build("failed observations")
        assert any(rule.id == "RULE-001" for rule in context.rules)

    def test_attaches_similar_examples(
        self, context_builder: SchemaContextBuilder
    ) -> None:
        context = context_builder.build(
            "Show all failed observations in the last 24 hours."
        )
        assert context.examples

    def test_rendered_context_contains_every_section(
        self, context_builder: SchemaContextBuilder
    ) -> None:
        rendered = context_builder.build(
            "Show all failed observations in the last 24 hours."
        ).render()

        assert "## Available tables" in rendered
        assert "## Verified join paths" in rendered
        assert "## SQL generation rules" in rendered
        assert "## Similar answered questions" in rendered

    def test_unrelated_question_yields_no_schema(
        self, context_builder: SchemaContextBuilder
    ) -> None:
        context = context_builder.build("zzzz qqqq xxxx")
        assert context.is_empty
