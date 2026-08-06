"""Tests for the model-backed generator and the validation repair loop.

A stub client stands in for a real provider so the whole path — prompt assembly,
response parsing, fallback and repair — is exercised deterministically and offline.
"""

from __future__ import annotations

import pytest

from nl2sql.analysis.question_analyzer import QuestionAnalyzer
from nl2sql.config import LLMProvider, Settings
from nl2sql.database.executor import QueryExecutor
from nl2sql.exceptions import ConfigurationError, LLMError
from nl2sql.generation.deterministic.generator import DeterministicSQLGenerator
from nl2sql.generation.llm_generator import LLMSQLGenerator
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.llm.base import LLMResponse
from nl2sql.pipeline import NL2SQLPipeline
from nl2sql.prompts.templates import build_generation_prompts, build_repair_prompts
from nl2sql.retrieval.context_builder import SchemaContextBuilder

VALID_SQL = """
SELECT o.observation_id
FROM observations o
WHERE o.status = 'FAILED'
LIMIT 10
"""

INVALID_SQL = "SELECT o.not_a_column FROM observations o"


class StubLLMClient:
    """Returns queued responses and records the prompts it was given."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    @property
    def model_name(self) -> str:
        return "stub-model"

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls.append((system_prompt, user_prompt))

        if not self._responses:
            raise AssertionError("StubLLMClient was called more times than expected")

        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt

        return LLMResponse(
            text=nxt, model="stub-model", input_tokens=100, output_tokens=20
        )


@pytest.fixture
def question() -> str:
    return "Show all failed observations in the last 24 hours."


class TestPrompts:
    def test_generation_prompt_embeds_retrieved_context(
        self,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
        question: str,
    ) -> None:
        system_prompt, user_prompt = build_generation_prompts(
            context_builder.build(question), analyzer.analyze(question), "sqlite"
        )

        assert "## Available tables" in system_prompt
        assert "observations" in system_prompt
        assert "RULE-001" in system_prompt
        assert question in user_prompt

    def test_generation_prompt_carries_the_analysis_hints(
        self,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
        question: str,
    ) -> None:
        _, user_prompt = build_generation_prompts(
            context_builder.build(question), analyzer.analyze(question), "sqlite"
        )
        assert "Time window requested" in user_prompt
        assert "FAILED" in user_prompt

    def test_repair_prompt_lists_the_errors_to_fix(
        self, context_builder: SchemaContextBuilder, question: str
    ) -> None:
        _, user_prompt = build_repair_prompts(
            context_builder.build(question),
            INVALID_SQL,
            ["Column 'not_a_column' does not exist on table 'observations'."],
            "sqlite",
        )
        assert INVALID_SQL in user_prompt
        assert "not_a_column" in user_prompt


class TestLLMGenerator:
    def test_returns_the_model_response_as_sql(
        self,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
        question: str,
    ) -> None:
        generator = LLMSQLGenerator(StubLLMClient([VALID_SQL]))
        result = generator.generate(
            context_builder.build(question), analyzer.analyze(question)
        )

        assert result.succeeded
        assert "observations" in result.sql
        assert result.tokens_used == 120

    def test_strips_markdown_fences(
        self,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
        question: str,
    ) -> None:
        generator = LLMSQLGenerator(StubLLMClient([f"```sql\n{VALID_SQL}\n```"]))
        result = generator.generate(
            context_builder.build(question), analyzer.analyze(question)
        )
        assert result.sql.startswith("SELECT")
        assert "```" not in result.sql

    def test_honours_the_insufficient_context_marker(
        self,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
        question: str,
    ) -> None:
        generator = LLMSQLGenerator(
            StubLLMClient(["INSUFFICIENT_CONTEXT: no pricing table is available."])
        )
        result = generator.generate(
            context_builder.build(question), analyzer.analyze(question)
        )

        assert not result.succeeded
        assert result.insufficient_context
        assert "pricing" in result.explanation

    def test_falls_back_when_the_provider_fails(
        self,
        registry: KnowledgeBaseRegistry,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
        question: str,
    ) -> None:
        generator = LLMSQLGenerator(
            StubLLMClient([LLMError("provider unavailable")]),
            fallback=DeterministicSQLGenerator(registry),
        )
        result = generator.generate(
            context_builder.build(question), analyzer.analyze(question)
        )

        assert result.succeeded
        assert result.generator == "deterministic-planner"

    def test_provider_failure_without_a_fallback_is_reported(
        self,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
        question: str,
    ) -> None:
        generator = LLMSQLGenerator(StubLLMClient([LLMError("provider unavailable")]))
        result = generator.generate(
            context_builder.build(question), analyzer.analyze(question)
        )

        assert not result.succeeded
        assert "unavailable" in result.explanation


class TestRepairLoop:
    def _pipeline(
        self,
        registry: KnowledgeBaseRegistry,
        settings: Settings,
        executor: QueryExecutor,
        responses: list[str | Exception],
    ) -> tuple[NL2SQLPipeline, StubLLMClient]:
        client = StubLLMClient(responses)
        pipeline = NL2SQLPipeline(
            registry,
            settings,
            generator=LLMSQLGenerator(client),
            executor=executor,
        )
        return pipeline, client

    def test_repairs_a_query_that_failed_validation(
        self,
        registry: KnowledgeBaseRegistry,
        settings: Settings,
        executor: QueryExecutor,
        question: str,
    ) -> None:
        pipeline, client = self._pipeline(
            registry, settings, executor, [INVALID_SQL, VALID_SQL]
        )
        answer = pipeline.answer(question)

        assert answer.succeeded
        assert answer.repair_attempts == 1
        assert len(client.calls) == 2
        assert "not_a_column" in client.calls[1][1], "repair prompt must cite the error"

    def test_gives_up_after_the_repair_budget_is_spent(
        self,
        registry: KnowledgeBaseRegistry,
        settings: Settings,
        executor: QueryExecutor,
        question: str,
    ) -> None:
        attempts = settings.max_repair_attempts
        pipeline, client = self._pipeline(
            registry, settings, executor, [INVALID_SQL] * (attempts + 1)
        )
        answer = pipeline.answer(question)

        assert not answer.succeeded
        assert answer.repair_attempts == attempts
        assert len(client.calls) == attempts + 1
        assert answer.validation_errors

    def test_a_valid_first_attempt_needs_no_repair(
        self,
        registry: KnowledgeBaseRegistry,
        settings: Settings,
        executor: QueryExecutor,
        question: str,
    ) -> None:
        pipeline, client = self._pipeline(registry, settings, executor, [VALID_SQL])
        answer = pipeline.answer(question)

        assert answer.succeeded
        assert answer.repair_attempts == 0
        assert len(client.calls) == 1

    def test_a_write_statement_from_the_model_is_blocked(
        self,
        registry: KnowledgeBaseRegistry,
        settings: Settings,
        executor: QueryExecutor,
        question: str,
    ) -> None:
        attempts = settings.max_repair_attempts
        pipeline, _ = self._pipeline(
            registry,
            settings,
            executor,
            ["DELETE FROM observations"] * (attempts + 1),
        )
        answer = pipeline.answer(question)

        assert not answer.succeeded
        assert any("read-only" in error.lower() for error in answer.validation_errors)


class TestProviderDegradation:
    """A provider that cannot be built must not stop the system from starting."""

    def test_starts_when_the_provider_sdk_is_not_installed(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nl2sql import pipeline as pipeline_module

        def unavailable(_settings: Settings) -> None:
            raise ConfigurationError("The 'openai' package is required")

        monkeypatch.setattr(pipeline_module, "build_llm_client", unavailable)

        configured = settings.model_copy(
            update={"llm_provider": LLMProvider.OPENAI, "openai_api_key": "test-key"}
        )
        built = NL2SQLPipeline.create(configured)

        answer = built.answer("Display observations grouped by environment.")
        assert answer.succeeded
        assert answer.generator == "deterministic-planner"

    def test_starts_when_credentials_are_missing(self, settings: Settings) -> None:
        configured = settings.model_copy(
            update={"llm_provider": LLMProvider.OPENAI, "openai_api_key": None}
        )
        built = NL2SQLPipeline.create(configured)

        answer = built.answer("Display observations grouped by environment.")
        assert answer.succeeded


class TestUnboundedQueries:
    def test_sql_without_a_limit_is_accepted_as_written(
        self,
        registry: KnowledgeBaseRegistry,
        settings: Settings,
        executor: QueryExecutor,
        question: str,
    ) -> None:
        """A question naming no row count produces SQL with no LIMIT."""
        client = StubLLMClient(
            ["SELECT o.observation_id FROM observations o WHERE o.status = 'FAILED'"]
        )
        pipeline = NL2SQLPipeline(
            registry, settings, generator=LLMSQLGenerator(client), executor=executor
        )
        answer = pipeline.answer(question)

        assert answer.succeeded
        assert answer.repair_attempts == 0
        assert "LIMIT" not in answer.sql.upper()
        # Nothing caps it: every matching row comes back.
        assert not answer.truncated
