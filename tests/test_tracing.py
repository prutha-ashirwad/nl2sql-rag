"""Tests for LangSmith tracing.

Tracing is optional, so the property that matters most is that switching it off costs
nothing and leaks nothing: no environment variable set to a live value, no network
call, and a workflow that still runs. Nothing here contacts LangSmith.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from nl2sql import tracing
from nl2sql.analysis.question_analyzer import QuestionAnalyzer
from nl2sql.config import LLMProvider, Settings, get_settings
from nl2sql.pipeline import NL2SQLPipeline
from nl2sql.retrieval.context_builder import SchemaContextBuilder
from nl2sql.tracing import (
    configure_tracing,
    is_tracing_active,
    run_config,
    summarise_payload,
)
from tests.conftest import build_test_settings

TRACING_VARS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
)


@pytest.fixture(autouse=True)
def isolate_tracing(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo everything a test does to the module's global tracing state.

    Enabling tracing installs a process-wide tracer. Without this teardown a test that
    turns it on leaves it on, and a later test's pipeline call would try to upload to
    whatever endpoint that test configured — a real network request from a unit suite.
    """
    for name in TRACING_VARS:
        monkeypatch.delenv(name, raising=False)
    # Restored rather than left unset: conftest sets this off for the whole session so
    # that ambient settings cannot enable tracing, and deleting it here would reopen
    # exactly the hole conftest closes.
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    yield

    tracing._tracer = None
    tracing._active = False


class TestConfigureTracing:
    """Turning tracing on requires it to be both requested and usable."""

    def test_off_by_default(self) -> None:
        assert configure_tracing(build_test_settings()) is False
        assert is_tracing_active() is False

    def test_requested_without_a_key_stays_off(self) -> None:
        settings = build_test_settings(langsmith_tracing=True)

        # Better to run untraced than to fail start-up over an observability setting.
        assert configure_tracing(settings) is False
        assert is_tracing_active() is False

    def test_requested_with_a_key_turns_on(self) -> None:
        settings = build_test_settings(
            langsmith_tracing=True,
            langsmith_api_key="test-key",
            langsmith_project="my-experiments",
        )

        assert configure_tracing(settings) is True
        assert is_tracing_active() is True

    def test_an_ambient_variable_does_not_enable_tracing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A stray LANGSMITH_TRACING in the shell must not start shipping questions to
        # a remote service behind the operator's back.
        monkeypatch.setenv("LANGSMITH_TRACING", "true")

        assert configure_tracing(build_test_settings()) is False
        assert is_tracing_active() is False

    def test_settings_reach_the_environment(self) -> None:
        configure_tracing(
            build_test_settings(
                langsmith_tracing=True,
                langsmith_api_key="test-key",
                langsmith_project="my-experiments",
                langsmith_endpoint="https://eu.api.smith.langchain.com",
            )
        )

        assert os.environ["LANGSMITH_PROJECT"] == "my-experiments"
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"


class TestRunConfig:
    """Each run records the settings that could have changed its answer."""

    def test_records_the_settings_that_shape_an_answer(self) -> None:
        config = run_config(
            build_test_settings(lexical_weight=0.75, retrieval_top_k=12),
            generator_name="deterministic-planner",
        )
        metadata = config["metadata"]

        # Two runs differing only in these are an experiment; the trace has to carry
        # them or the difference cannot be attributed to anything.
        assert metadata["lexical_weight"] == 0.75
        assert metadata["retrieval_top_k"] == 12
        assert metadata["generator"] == "deterministic-planner"

    def test_tags_carry_the_generator_and_provider(self) -> None:
        config = run_config(
            build_test_settings(), generator_name="deterministic-planner"
        )

        assert "generator:deterministic-planner" in config["tags"]
        assert "provider:deterministic" in config["tags"]

    def test_caller_tags_and_metadata_are_merged(self) -> None:
        config = run_config(
            build_test_settings(),
            generator_name="deterministic-planner",
            tags=["comparison", "engine:gpt-4o"],
            metadata={"comparison_question": "How many alerts are open?"},
        )

        assert "comparison" in config["tags"]
        assert "generator:deterministic-planner" in config["tags"]
        assert config["metadata"]["comparison_question"] == "How many alerts are open?"

    def test_no_model_is_recorded_for_the_planner(self) -> None:
        config = run_config(
            build_test_settings(), generator_name="deterministic-planner"
        )

        # Recording a null model would make the field useless for filtering runs.
        assert "model" not in config["metadata"]

    def test_the_model_is_recorded_for_a_configured_provider(self) -> None:
        settings = build_test_settings(
            llm_provider=LLMProvider.OPENAI, llm_model="gpt-4o-mini"
        )

        config = run_config(settings, generator_name="llm-generator")

        assert config["metadata"]["model"] == "gpt-4o-mini"


class TestPayloadSummarising:
    """What gets uploaded has to stay small, or runs are dropped.

    LangGraph traces the whole state at every node, and the state carries the retrieved
    context — every document and the full metadata of every table. Uploaded verbatim
    that is roughly 400 KB per question across eleven spans, which saturates the
    background uploader and loses runs silently.
    """

    @staticmethod
    def _traced_state(
        analyzer: QuestionAnalyzer, context_builder: SchemaContextBuilder
    ) -> dict[str, object]:
        """Build a state dict shaped like the one the tracer sees mid-workflow."""
        question = "Display observations grouped by environment."
        analysis = analyzer.analyze(question)

        return {
            "question": question,
            "analysis": analysis,
            "context": context_builder.build(question, analysis),
            "errors": [],
        }

    def test_the_payload_shrinks_by_an_order_of_magnitude(
        self, analyzer: QuestionAnalyzer, context_builder: SchemaContextBuilder
    ) -> None:
        import json

        state = self._traced_state(analyzer, context_builder)

        raw = len(json.dumps(state, default=str))
        summarised = len(json.dumps(summarise_payload(state), default=str))

        assert summarised < raw / 10, (
            f"summarising cut {raw} bytes to {summarised}, which is not enough: "
            "a full workflow uploads this eleven times over"
        )

    def test_the_summary_still_explains_the_answer(
        self, analyzer: QuestionAnalyzer, context_builder: SchemaContextBuilder
    ) -> None:
        summary = summarise_payload(self._traced_state(analyzer, context_builder))

        # Small is only useful if a reader can still see what retrieval decided.
        assert summary["context"]["base_table"] == "observations"
        assert "environments" in summary["context"]["tables"]
        assert summary["context"]["document_count"] > 0
        assert summary["analysis"]["intent"] == "aggregate"
        assert summary["analysis"]["groupings"] == ["environments.environment_name"]

    def test_result_rows_are_never_uploaded(self) -> None:
        from nl2sql.database.executor import QueryResult

        result = QueryResult(
            columns=["device_name"],
            rows=[{"device_name": "iad1-core-101"}],
            row_count=1,
        )

        summary = summarise_payload({"execution": result})

        # Rows are business data; an observability backend is the wrong place for them.
        assert "rows" not in summary["execution"]
        assert summary["execution"]["row_count"] == 1
        assert summary["execution"]["columns"] == ["device_name"]

    def test_unknown_fields_pass_through_untouched(self) -> None:
        payload = {"question": "How many alerts are open?", "repair_attempts": 1}

        assert summarise_payload(payload) == payload

    def test_a_non_dict_payload_is_returned_as_is(self) -> None:
        assert summarise_payload("not a state") == "not a state"

    def test_a_broken_object_does_not_break_the_run(self) -> None:
        class Exploding:
            @property
            def documents(self) -> list[object]:
                raise RuntimeError("boom")

        # Summarising is best-effort: a trace is never worth failing a question over.
        summary = summarise_payload({"context": Exploding()})

        assert summary["context"] == "<Exploding>"


class TestTheSuiteEmitsNoTelemetry:
    """Running the tests must never upload anything to a real project.

    The FastAPI lifespan calls ``get_settings()`` directly, which reads the developer's
    ``.env``. Booting the app in a test therefore used to install a live tracer for the
    remainder of the session, and every later test question was uploaded to whatever
    LangSmith project that developer had configured.
    """

    @staticmethod
    def _ambient_settings() -> Settings:
        """Settings exactly as the API lifespan builds them, reading any local .env."""
        get_settings.cache_clear()
        try:
            return get_settings()
        finally:
            get_settings.cache_clear()

    def test_ambient_settings_do_not_enable_tracing(self) -> None:
        # Whatever the developer has in .env, the suite must not be able to trace.
        assert self._ambient_settings().langsmith_tracing is False

    def test_configuring_from_ambient_settings_stays_off(self) -> None:
        assert configure_tracing(self._ambient_settings()) is False
        assert is_tracing_active() is False


class TestTracingIsOptional:
    """The workflow must behave identically whether or not tracing is on."""

    def test_answers_are_unaffected(self, pipeline: NL2SQLPipeline) -> None:
        assert is_tracing_active() is False

        answer = pipeline.answer("Display observations grouped by environment.")

        assert answer.succeeded is True
        assert answer.row_count > 0

    def test_caller_labels_do_not_disturb_the_answer(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        plain = pipeline.answer("Count observations by status.")
        labelled = pipeline.answer(
            "Count observations by status.",
            tags=["experiment:baseline"],
            metadata={"run_by": "test"},
        )

        assert labelled.sql == plain.sql
        assert labelled.row_count == plain.row_count
