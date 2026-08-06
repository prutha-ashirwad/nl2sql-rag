"""End-to-end tests for the workflow, database layer and pipeline."""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, text

from nl2sql.database.executor import QueryExecutor
from nl2sql.dialects import SQLITE
from nl2sql.exceptions import ExecutionError, NL2SQLError
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.pipeline import NL2SQLPipeline

# The three questions named in the brief. They are the acceptance criteria for the
# system, so they are asserted on directly rather than folded into a wider sweep.
HEADLINE_QUESTIONS = (
    "Show all failed observations in the last 24 hours.",
    "List interfaces with the highest failure count.",
    "Display observations grouped by environment.",
)

WIDER_QUESTION_SET = (
    *HEADLINE_QUESTIONS,
    "What is the success rate per environment over the last 7 days?",
    "Which devices had the most failures in production?",
    "Break down failures by failure reason.",
    "Show the top 5 sites by failure count in the last week.",
    "How many observations were recorded per collector?",
    "Show the failure rate by site.",
    "List open alerts by severity.",
    "Show failures grouped by vendor.",
    "How many incidents does each team own?",
)


class TestSchemaBuilder:
    @pytest.mark.parametrize(
        ("declared", "expected"),
        [
            ("INTEGER", "INTEGER"),
            ("BIGINT", "INTEGER"),
            ("VARCHAR(64)", "TEXT"),
            ("DECIMAL(18,4)", "REAL"),
            ("BOOLEAN", "INTEGER"),
            ("TIMESTAMP", "TEXT"),
        ],
    )
    def test_maps_declared_types_to_storage_classes(
        self, declared: str, expected: str
    ) -> None:
        assert SQLITE.storage_type(declared) == expected


class TestDatabase:
    def test_every_declared_table_exists(
        self, seeded_engine: Engine, registry: KnowledgeBaseRegistry
    ) -> None:
        with seeded_engine.connect() as connection:
            rows = connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table'")
            ).fetchall()

        created = {row[0] for row in rows}
        assert set(registry.table_names) <= created

    def test_fact_table_is_populated(self, executor: QueryExecutor) -> None:
        result = executor.execute("SELECT COUNT(*) AS c FROM observations")
        assert result.rows[0]["c"] > 0

    def test_foreign_keys_all_resolve(self, executor: QueryExecutor) -> None:
        # An orphaned foreign key would make generated joins silently drop rows.
        result = executor.execute(
            """
            SELECT COUNT(*) AS orphans
            FROM observations o
            LEFT JOIN interfaces i ON o.interface_id = i.interface_id
            WHERE i.interface_id IS NULL
            """
        )
        assert result.rows[0]["orphans"] == 0

    def test_executor_refuses_writes(self, executor: QueryExecutor) -> None:
        with pytest.raises(ExecutionError, match="read-only|SELECT"):
            executor.execute("DELETE FROM observations")

    def test_executor_refuses_multiple_statements(
        self, executor: QueryExecutor
    ) -> None:
        with pytest.raises(ExecutionError, match="Multiple statements"):
            executor.execute("SELECT 1 FROM observations; SELECT 2")

    def test_executor_caps_returned_rows(self, seeded_engine: Engine) -> None:
        capped = QueryExecutor(seeded_engine, max_rows=5)
        result = capped.execute("SELECT observation_id FROM observations")
        assert result.row_count == 5
        assert result.truncated


class TestPipeline:
    @pytest.mark.parametrize("question", HEADLINE_QUESTIONS)
    def test_answers_the_headline_questions(
        self, pipeline: NL2SQLPipeline, question: str
    ) -> None:
        answer = pipeline.answer(question)
        assert answer.succeeded, answer.answer
        assert answer.sql
        assert not answer.validation_errors

    @pytest.mark.parametrize("question", WIDER_QUESTION_SET)
    def test_produces_executable_sql(
        self, pipeline: NL2SQLPipeline, question: str
    ) -> None:
        answer = pipeline.answer(question)
        assert answer.succeeded, f"{question}: {answer.answer}"
        # Execution ran without error; a query that failed would have recorded one.
        assert not answer.errors, answer.errors

    def test_failed_observations_query_returns_only_failures(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        answer = pipeline.answer("Show all failed observations in the last 24 hours.")
        assert answer.rows
        assert all(row["status"] == "FAILED" for row in answer.rows)

    def test_failed_observations_query_returns_distinct_rows(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        # A fan-out join would repeat the same observation many times.
        answer = pipeline.answer("Show all failed observations in the last 24 hours.")
        identifiers = [row["observation_id"] for row in answer.rows]
        assert len(identifiers) == len(set(identifiers))

    def test_environment_breakdown_covers_every_environment(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        answer = pipeline.answer("Display observations grouped by environment.")
        assert {row["environment_name"] for row in answer.rows} >= {
            "Production",
            "Staging",
        }

    def test_ranking_is_ordered_descending(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        answer = pipeline.answer("List interfaces with the highest failure count.")
        counts = [row["failed_count"] for row in answer.rows]
        assert counts == sorted(counts, reverse=True)

    def test_success_rate_is_a_percentage(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        answer = pipeline.answer("What is the success rate per environment?")
        assert answer.rows
        assert all(0 <= row["success_rate_pct"] <= 100 for row in answer.rows)

    def test_refuses_data_modification(self, pipeline: NL2SQLPipeline) -> None:
        answer = pipeline.answer("Delete all failed observations")
        assert not answer.succeeded
        assert answer.sql is None
        assert "read-only" in answer.answer.lower()

    def test_reports_an_unanswerable_question_clearly(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        answer = pipeline.answer("What is the airspeed velocity of a swallow?")
        assert not answer.succeeded
        assert answer.answer

    def test_rejects_an_empty_question(self, pipeline: NL2SQLPipeline) -> None:
        with pytest.raises(NL2SQLError, match="must not be empty"):
            pipeline.answer("   ")

    def test_records_a_trace_of_every_step(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        answer = pipeline.answer("Display observations grouped by environment.")
        nodes = [event.node for event in answer.trace]
        assert nodes == [
            "analyze_question",
            "retrieve_context",
            "generate_sql",
            "validate_sql",
            "execute_sql",
            "finalize",
        ]

    def test_short_circuits_before_generating_for_a_rejected_question(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        answer = pipeline.answer("Drop the observations table")
        nodes = [event.node for event in answer.trace]
        assert nodes == ["analyze_question", "finalize"]

    def test_answer_serialises_to_json_safe_types(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        import json

        answer = pipeline.answer("Display observations grouped by environment.")
        assert json.dumps(answer.to_dict(), default=str)

    def test_repeated_questions_produce_identical_sql(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        question = "Show the top 5 sites by failure count in the last week."
        baseline = pipeline.answer(question).sql
        for _ in range(5):
            assert pipeline.answer(question).sql == baseline
