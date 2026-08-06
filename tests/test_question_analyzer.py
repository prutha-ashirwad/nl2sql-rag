"""Tests for structured question analysis."""

from __future__ import annotations

import pytest

from nl2sql.analysis.question_analyzer import QueryIntent, QuestionAnalyzer, TimeUnit


class TestIntentClassification:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("Show all failed observations", QueryIntent.LIST),
            ("List the open incidents", QueryIntent.LIST),
            ("Display observations grouped by environment", QueryIntent.AGGREGATE),
            ("How many failures were there?", QueryIntent.AGGREGATE),
            ("List interfaces with the highest failure count", QueryIntent.RANKING),
            ("Top 5 devices by failures", QueryIntent.RANKING),
        ],
    )
    def test_classifies_intent(
        self, analyzer: QuestionAnalyzer, question: str, expected: QueryIntent
    ) -> None:
        assert analyzer.analyze(question).intent is expected


class TestSafety:
    @pytest.mark.parametrize(
        "question",
        [
            "Delete all failed observations",
            "Update the device status to ACTIVE",
            "Drop the observations table",
            "Insert a new device",
        ],
    )
    def test_rejects_data_modification(
        self, analyzer: QuestionAnalyzer, question: str
    ) -> None:
        analysis = analyzer.analyze(question)
        assert analysis.intent is QueryIntent.UNSUPPORTED
        assert not analysis.is_supported
        assert analysis.rejection_reason


class TestTimeWindows:
    @pytest.mark.parametrize(
        ("question", "quantity", "unit"),
        [
            ("failures in the last 24 hours", 24, TimeUnit.HOUR),
            ("failures over the past 7 days", 7, TimeUnit.DAY),
            ("failures in the last hour", 1, TimeUnit.HOUR),
            ("failures in the previous 2 weeks", 2, TimeUnit.WEEK),
            ("failures in the last three months", 3, TimeUnit.MONTH),
        ],
    )
    def test_parses_relative_windows(
        self,
        analyzer: QuestionAnalyzer,
        question: str,
        quantity: int,
        unit: TimeUnit,
    ) -> None:
        window = analyzer.analyze(question).time_window
        assert window is not None
        assert window.quantity == quantity
        assert window.unit is unit

    def test_yesterday_is_a_closed_window(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        window = analyzer.analyze("failures yesterday").time_window
        assert window is not None
        assert window.include_upper_bound

    def test_no_window_when_none_is_mentioned(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        assert analyzer.analyze("show all failures").time_window is None


class TestValueFilters:
    def test_maps_everyday_wording_to_the_stored_literal(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        filters = analyzer.analyze("show failed observations").value_filters
        assert any(
            item.table == "observations"
            and item.column == "status"
            and item.value == "FAILED"
            for item in filters
        )

    def test_maps_production_to_the_environment_code(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        filters = analyzer.analyze(
            "failed observations in production environments"
        ).value_filters
        assert any(item.value == "PROD" for item in filters)

    def test_ignores_enum_words_when_the_table_is_not_mentioned(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        # "break down" must not be read as interfaces.oper_status = 'DOWN'.
        filters = analyzer.analyze(
            "Break down failures by failure reason."
        ).value_filters
        assert all(item.table != "interfaces" for item in filters)

    def test_curated_wording_is_trusted_without_naming_the_table(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        # "failures" is a declared value synonym for FAILED, so it stands on its
        # own even though the question never says "observations".
        analysis = analyzer.analyze(
            "Which devices had the most failures in production?"
        )
        assert "observations" not in analysis.mentioned_tables
        assert any(
            item.table == "observations" and item.value == "FAILED"
            for item in analysis.value_filters
        )

    def test_both_filters_survive_together(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        values = {
            item.value
            for item in analyzer.analyze(
                "Which devices had the most failures in production?"
            ).value_filters
        }
        assert {"FAILED", "PROD"} <= values


class TestGroupings:
    @pytest.mark.parametrize(
        ("question", "table", "column"),
        [
            ("observations grouped by environment", "environments", "environment_name"),
            ("failures per device", "devices", "device_name"),
            ("failures broken down by site", "sites", "site_name"),
        ],
    )
    def test_resolves_grouping_dimensions(
        self, analyzer: QuestionAnalyzer, question: str, table: str, column: str
    ) -> None:
        groupings = analyzer.analyze(question).groupings
        assert any(
            item.table == table and item.column == column for item in groupings
        )

    def test_by_measure_is_not_a_grouping(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        # "by failure count" names the sort order, not the breakdown.
        analysis = analyzer.analyze("top 5 sites by failure count")
        assert all(item.table != "observations" for item in analysis.groupings)


class TestMetrics:
    def test_recognises_a_declared_metric(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        metric = analyzer.analyze("success rate per environment").metric
        assert metric is not None
        assert metric.alias == "success_rate_pct"

    def test_metric_words_do_not_become_filters(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        # Filtering to successful rows would make every success rate exactly 100%.
        analysis = analyzer.analyze("success rate per environment")
        assert all(item.value != "SUCCESS" for item in analysis.value_filters)


class TestRowLimits:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [("top 5 devices", 5), ("first 20 observations", 20), ("show failures", None)],
    )
    def test_parses_explicit_limits(
        self, analyzer: QuestionAnalyzer, question: str, expected: int | None
    ) -> None:
        assert analyzer.analyze(question).row_limit == expected

    @pytest.mark.parametrize(
        "question",
        [
            "Show all failed observations in the last 24 hours.",
            "Show all broken observations and i need all the rows",
            "Show broken observations, no limit",
            "Display observations grouped by environment.",
        ],
    )
    def test_a_question_without_a_number_is_unbounded(
        self, analyzer: QuestionAnalyzer, question: str
    ) -> None:
        # No LIMIT in the SQL: the executor's row cap bounds the result and reports
        # when it bit, so the answer is never silently truncated.
        assert analyzer.analyze(question).row_limit is None

    def test_lowest_reverses_the_sort_direction(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        lowest = analyzer.analyze("devices with the lowest failure count")
        highest = analyzer.analyze("devices with the highest failure count")
        assert lowest.descending is False
        assert highest.descending is True


class TestDeterminism:
    def test_repeated_analysis_is_identical(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        question = "top 5 sites by failure count in the last week"
        baseline = analyzer.analyze(question)

        for _ in range(20):
            repeat = analyzer.analyze(question)
            assert repeat.intent is baseline.intent
            assert repeat.mentioned_tables == baseline.mentioned_tables
            assert repeat.groupings == baseline.groupings
            assert repeat.value_filters == baseline.value_filters

