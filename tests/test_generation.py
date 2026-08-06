"""Tests for SQL generation: the deterministic planner and the dialect layer."""

from __future__ import annotations

import pytest

from nl2sql.analysis.question_analyzer import (
    QuestionAnalyzer,
    TimeUnit,
    TimeWindow,
)
from nl2sql.dialects import MYSQL, POSTGRES, SQLITE, get_dialect
from nl2sql.generation.deterministic.generator import DeterministicSQLGenerator
from nl2sql.generation.deterministic.query_plan import QueryPlan, SelectExpression
from nl2sql.generation.llm_generator import extract_sql
from nl2sql.retrieval.context_builder import SchemaContextBuilder
from nl2sql.validation.validator import SQLValidator


def generate(
    generator: DeterministicSQLGenerator,
    context_builder: SchemaContextBuilder,
    analyzer: QuestionAnalyzer,
    question: str,
) -> str:
    """Run the generator end to end and return the SQL it produced."""
    result = generator.generate(
        context_builder.build(question), analyzer.analyze(question)
    )
    assert result.succeeded, result.explanation
    return result.sql


class TestDialects:
    def test_sqlite_converts_weeks_to_days(self) -> None:
        # SQLite has no 'weeks' modifier and evaluates it to NULL without erroring.
        assert SQLITE.timestamp_at_offset(2, TimeUnit.WEEK) == (
            "datetime('now', '-14 days')"
        )

    def test_sqlite_renders_supported_units_directly(self) -> None:
        assert SQLITE.timestamp_at_offset(24, TimeUnit.HOUR) == (
            "datetime('now', '-24 hours')"
        )

    def test_postgres_uses_interval_syntax(self) -> None:
        assert POSTGRES.timestamp_at_offset(7, TimeUnit.DAY) == (
            "NOW() - INTERVAL '7 days'"
        )

    def test_mysql_uses_singular_uppercase_units(self) -> None:
        assert MYSQL.timestamp_at_offset(7, TimeUnit.DAY) == (
            "DATE_SUB(NOW(), INTERVAL 7 DAY)"
        )

    def test_closed_windows_get_an_upper_bound(self) -> None:
        predicates = SQLITE.time_window_predicates(
            "o.observed_at",
            TimeWindow(1, TimeUnit.DAY, "yesterday", include_upper_bound=True),
        )
        assert len(predicates) == 2
        assert "<" in predicates[1]

    def test_unknown_dialect_falls_back_to_sqlite(self) -> None:
        assert get_dialect("not-a-database").name == "sqlite"

    def test_known_aliases_resolve(self) -> None:
        assert get_dialect("postgresql").name == "postgres"


class TestQueryPlan:
    def test_group_by_is_derived_from_the_select_list(self) -> None:
        plan = QueryPlan(base_table="observations", base_alias="o")
        plan.select = [
            SelectExpression("e.environment_name"),
            SelectExpression("COUNT(*)", alias="c", is_aggregate=True),
        ]
        assert "GROUP BY e.environment_name" in plan.to_sql()

    def test_no_group_by_without_an_aggregate(self) -> None:
        plan = QueryPlan(base_table="observations", base_alias="o")
        plan.select = [SelectExpression("o.observation_id")]
        assert "GROUP BY" not in plan.to_sql()


class TestDeterministicGeneration:
    @pytest.mark.parametrize(
        ("question", "expected_fragments"),
        [
            (
                "Show all failed observations in the last 24 hours.",
                ["FROM observations o", "o.status = 'FAILED'", "-24 hours"],
            ),
            (
                "List interfaces with the highest failure count.",
                ["COUNT(*)", "GROUP BY", "ORDER BY", "i.interface_name"],
            ),
            (
                "Display observations grouped by environment.",
                ["e.environment_name", "GROUP BY e.environment_name", "COUNT(*)"],
            ),
            (
                "What is the success rate per environment?",
                ["success_rate_pct", "CASE WHEN", "GROUP BY"],
            ),
        ],
    )
    def test_generates_the_expected_query_shape(
        self,
        generator: DeterministicSQLGenerator,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
        question: str,
        expected_fragments: list[str],
    ) -> None:
        sql = generate(generator, context_builder, analyzer, question)
        for fragment in expected_fragments:
            assert fragment in sql, f"{fragment!r} missing from:\n{sql}"

    def test_uses_a_left_join_for_nullable_keys(
        self,
        generator: DeterministicSQLGenerator,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
    ) -> None:
        sql = generate(
            generator,
            context_builder,
            analyzer,
            "Show all failed observations in the last 24 hours.",
        )
        assert "LEFT JOIN failure_reasons" in sql

    def test_listing_rows_never_joins_a_fan_out_table(
        self,
        generator: DeterministicSQLGenerator,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
    ) -> None:
        # One observation has many metric rows and many alerts; joining either would
        # duplicate the observations being listed.
        sql = generate(
            generator,
            context_builder,
            analyzer,
            "Show all failed observations in the last 24 hours.",
        )
        assert "observation_metrics" not in sql
        assert "JOIN alerts" not in sql

    def test_prefers_the_business_join_path(
        self,
        generator: DeterministicSQLGenerator,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
    ) -> None:
        sql = generate(
            generator,
            context_builder,
            analyzer,
            "Show the top 5 sites by failure count.",
        )
        assert "JOIN devices" in sql
        assert "JOIN collectors" not in sql

    def test_honours_an_explicit_row_limit(
        self,
        generator: DeterministicSQLGenerator,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
    ) -> None:
        sql = generate(
            generator,
            context_builder,
            analyzer,
            "Show the top 5 sites by failure count.",
        )
        assert sql.rstrip().endswith("LIMIT 5")

    def test_output_is_deterministic(
        self,
        generator: DeterministicSQLGenerator,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
    ) -> None:
        question = "List interfaces with the highest failure count."
        baseline = generate(generator, context_builder, analyzer, question)
        for _ in range(10):
            assert generate(generator, context_builder, analyzer, question) == baseline

    def test_reports_insufficient_context_rather_than_guessing(
        self,
        generator: DeterministicSQLGenerator,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
    ) -> None:
        question = "zzzz qqqq xxxx"
        result = generator.generate(
            context_builder.build(question), analyzer.analyze(question)
        )
        assert not result.succeeded
        assert result.insufficient_context
        assert result.explanation


class TestDefaultFilters:
    """A table's declared default filters must reach the generated query."""

    def test_decommissioned_devices_are_excluded(
        self,
        generator: DeterministicSQLGenerator,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
    ) -> None:
        # Retired hardware stops reporting, so counting it inflates every failure
        # total and depresses availability.
        sql = generate(
            generator,
            context_builder,
            analyzer,
            "Which devices had the most failures in production?",
        )
        assert "device_status <> 'DECOMMISSIONED'" in sql

    def test_inactive_sites_are_excluded(
        self,
        generator: DeterministicSQLGenerator,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
    ) -> None:
        sql = generate(
            generator,
            context_builder,
            analyzer,
            "Show the top 5 sites by failure count.",
        )
        # ``= TRUE`` rather than ``= 1``: the Knowledge Base's default filters are
        # copied verbatim into the query, so they have to read on every engine.
        assert "is_active = TRUE" in sql

    def test_filters_are_bound_to_the_query_alias(
        self,
        generator: DeterministicSQLGenerator,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
    ) -> None:
        sql = generate(
            generator,
            context_builder,
            analyzer,
            "Which devices had the most failures in production?",
        )
        # Rendered with the alias, never the bare table name.
        assert "d.device_status" in sql
        assert "devices.device_status" not in sql

    def test_an_explicit_filter_overrides_the_default(
        self,
        generator: DeterministicSQLGenerator,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
    ) -> None:
        # Asking about retired hardware must not be silently filtered to nothing.
        sql = generate(
            generator,
            context_builder,
            analyzer,
            "List decommissioned devices.",
        )
        assert "<> 'DECOMMISSIONED'" not in sql


class TestGeneratedSQLIsValid:
    """Everything the planner emits must survive the validator."""

    QUESTIONS = (
        "Show all failed observations in the last 24 hours.",
        "List interfaces with the highest failure count.",
        "Display observations grouped by environment.",
        "What is the success rate per environment over the last 7 days?",
        "Which devices had the most failures in production?",
        "Break down failures by failure reason.",
        "Show the top 5 sites by failure count in the last week.",
        "How many observations were recorded per collector?",
        "Show the failure rate by site.",
        "List open alerts by severity.",
    )

    @pytest.mark.parametrize("question", QUESTIONS)
    def test_generated_sql_passes_validation(
        self,
        generator: DeterministicSQLGenerator,
        context_builder: SchemaContextBuilder,
        analyzer: QuestionAnalyzer,
        validator: SQLValidator,
        question: str,
    ) -> None:
        sql = generate(generator, context_builder, analyzer, question)
        report = validator.validate(sql)
        assert report.is_valid, f"{question}\n{sql}\n{report.error_messages()}"


class TestResponseParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("SELECT 1", "SELECT 1"),
            ("```sql\nSELECT 1\n```", "SELECT 1"),
            ("```\nSELECT 1\n```", "SELECT 1"),
            ("SELECT 1;", "SELECT 1"),
            ("  SELECT 1  \n", "SELECT 1"),
        ],
    )
    def test_strips_markdown_and_trailing_punctuation(
        self, raw: str, expected: str
    ) -> None:
        assert extract_sql(raw) == expected
