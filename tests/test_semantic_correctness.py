"""Tests for answers that are syntactically valid but semantically wrong.

A query that parses, validates and runs can still answer a different question from
the one that was asked — grouping by the wrong table's column, silently widening the
grain, or reporting a database error as an empty result. Those failures are invisible
to the validator, so they are pinned down here instead.

Each expectation is checked against the database rather than against the SQL text
where possible, so the tests describe the answer rather than one particular way of
writing the query.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, text

from nl2sql.analysis.question_analyzer import QuestionAnalyzer
from nl2sql.config import Settings
from nl2sql.exceptions import ExecutionError
from nl2sql.generation.deterministic.generator import DeterministicSQLGenerator
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.pipeline import NL2SQLAnswer, NL2SQLPipeline
from nl2sql.ui.components import results_dataframe


class TestGroupingResolvesToTheSubjectOfTheQuestion:
    """A dimension word shared by several tables must resolve to the one asked about.

    ``status`` is declared by ``observations``, ``alerts`` and ``devices``. Picking
    whichever was declared first makes the answer depend on Knowledge Base file
    ordering, which is how "observations by status" came to group alerts instead.
    """

    @pytest.mark.parametrize(
        ("question", "table", "column"),
        [
            ("Count observations by status.", "observations", "status"),
            ("How many alerts are there per severity?", "alerts", "severity"),
            ("List incidents by severity.", "incidents", "severity"),
        ],
    )
    def test_the_named_table_wins(
        self,
        analyzer: QuestionAnalyzer,
        question: str,
        table: str,
        column: str,
    ) -> None:
        groupings = analyzer.analyze(question).groupings

        assert [(item.table, item.column) for item in groupings] == [(table, column)]

    def test_a_shared_word_really_does_match_several_tables(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        # Guards the test above: if "status" ever stopped being ambiguous the
        # assertions would still pass while testing nothing.
        matches = registry.resolve_dimension_for_phrase("status")

        assert len({match.table for match in matches}) > 1

    def test_each_introduces_a_grouping(self, analyzer: QuestionAnalyzer) -> None:
        # "How many incidents does each team own?" previously parsed as an ungrouped
        # count, answering "how many incidents are there" instead.
        groupings = analyzer.analyze("How many incidents does each team own?").groupings

        assert [item.table for item in groupings] == ["teams"]


class TestTheQueryIsAnchoredOnItsSubject:
    """Retrieval returns whatever is topically near; the anchor must be the subject.

    Anchoring on a fact table is usually right, but only for a fact the question is
    about. An unrelated fact is nearly always somewhere in the candidates, and
    anchoring on it answers a different question entirely — "total purchase order
    value by vendor" was answered from ``observations``.
    """

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("Show total purchase order value by vendor.", "purchase_orders"),
            ("List purchase orders by order status.", "purchase_orders"),
            ("How many purchase orders were delivered?", "purchase_orders"),
        ],
    )
    def test_a_question_about_a_table_reads_that_table(
        self, pipeline: NL2SQLPipeline, question: str, expected: str
    ) -> None:
        answer = pipeline.answer(question)

        assert expected in answer.tables_used
        assert "observations" not in answer.tables_used

    @pytest.mark.parametrize(
        "question",
        [
            "Which devices had the most failures in production?",
            "List interfaces with the highest failure count.",
            "Show the top 5 sites by failure count in the last week.",
            "Display observations grouped by environment.",
        ],
    )
    def test_measurements_are_still_read_from_the_fact_table(
        self, pipeline: NL2SQLPipeline, question: str
    ) -> None:
        # The subject rule must not cost the fact anchor where it was already right:
        # a failure count lives in observations however the question is phrased.
        answer = pipeline.answer(question)

        assert "observations" in answer.tables_used
        assert answer.row_count > 0

    def test_a_metric_pulls_in_the_table_it_is_computed_over(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        # "success rate" names no table at all; the glossary expression does.
        answer = pipeline.answer(
            "What is the success rate per environment over the last 7 days?"
        )

        assert "observations" in answer.tables_used
        assert answer.row_count > 0

    def test_a_column_word_alone_does_not_make_a_table_the_subject(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        analysis = analyzer.analyze("List purchase orders by order status.")

        # "status" touches alerts, but only "purchase orders" named a table.
        assert "purchase_orders" in analysis.named_tables
        assert "alerts" not in analysis.named_tables


class TestOnePhraseYieldsOneFilter:
    """An enum word shared by several columns must not filter on all of them.

    "DOWN" is an allowed value of both ``admin_status`` and ``oper_status``. Emitting
    a predicate for each and ANDing them asks for a row that is simultaneously in two
    mutually exclusive states, which no row can satisfy.
    """

    @pytest.mark.parametrize(
        ("question", "column"),
        [
            ("List interfaces whose oper_status is DOWN.", "oper_status"),
            ("Show interfaces that are down.", "oper_status"),
            ("Show me degraded observations.", "status"),
        ],
    )
    def test_a_single_column_is_filtered(
        self, analyzer: QuestionAnalyzer, question: str, column: str
    ) -> None:
        filters = analyzer.analyze(question).value_filters

        # The phrase resolves to exactly one column, and it is the right one.
        assert [item.column for item in filters] == [column]

    def test_a_word_spent_naming_a_table_is_not_also_a_value(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        # "maintenance" belongs to the phrase "maintenance windows", which names a
        # table. Reading it again as devices.device_status = 'MAINTENANCE' filtered
        # the query down to a table the question never asked about.
        filters = analyzer.analyze("Show maintenance windows by status.").value_filters

        assert all(item.table != "devices" for item in filters)

    def test_a_prefixed_column_is_reachable_from_the_bare_word(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        # maintenance_windows.window_status is not in the dimension lexicon under
        # "status", so the grouping used to land on another table's column.
        groupings = analyzer.analyze("Show maintenance windows by status.").groupings

        assert [(g.table, g.column) for g in groupings] == [
            ("maintenance_windows", "window_status")
        ]

    def test_the_word_really_is_shared_by_several_columns(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        # Without this the test above could pass because "down" stopped matching.
        matches = registry.resolve_enum_value("down")

        assert len({(match.table, match.column) for match in matches}) > 1

    def test_the_query_returns_rows(self, pipeline: NL2SQLPipeline) -> None:
        # The end-to-end symptom was a confident answer over zero rows.
        answer = pipeline.answer("Show interfaces that are down.")

        assert answer.succeeded is True
        assert answer.row_count > 0

    def test_distinct_phrases_still_contribute_separate_filters(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        # Narrowing to one filter per phrase must not collapse genuinely different
        # conditions drawn from different words.
        analysis = analyzer.analyze("Show failed observations in production.")
        filters = analysis.value_filters

        assert len({(item.table, item.column) for item in filters}) >= 2


class TestGroupedQueriesKeepTheirGrain:
    """A grouped query must break results down by what was asked for, and no more."""

    def test_no_extra_dimension_is_added_to_a_plain_attribute(
        self, pipeline: NL2SQLPipeline, seeded_engine: Engine
    ) -> None:
        answer = pipeline.answer("Count observations by status.")
        assert answer.sql is not None

        with seeded_engine.connect() as connection:
            generated = connection.execute(text(answer.sql)).fetchall()
            expected = connection.execute(
                text("SELECT status, COUNT(*) FROM observations GROUP BY 1")
            ).fetchall()

        # One row per status, not one row per status and some incidental column.
        assert len(generated) == len(expected)
        assert {row[0] for row in generated} == {row[0] for row in expected}

    def test_a_label_column_still_gets_its_parent_for_context(
        self, pipeline: NL2SQLPipeline
    ) -> None:
        # Two devices can each have an "Ethernet1/1", so grouping by interface name
        # is genuinely ambiguous without the device — this disambiguation must stay.
        answer = pipeline.answer("List interfaces with the highest failure count.")

        assert answer.sql is not None
        assert "device_name" in answer.sql


class TestExecutionFailuresAreReportedAsFailures:
    """A query the database rejects has not answered the question."""

    @staticmethod
    def _pipeline_with_broken_database(
        registry: KnowledgeBaseRegistry,
        settings: Settings,
        generator: DeterministicSQLGenerator,
    ) -> NL2SQLPipeline:
        class RejectingExecutor:
            """Stands in for a database that refuses every query."""

            def execute(self, sql: str) -> None:
                raise ExecutionError("no such table: observations")

        return NL2SQLPipeline(
            registry, settings, generator=generator, executor=RejectingExecutor()
        )

    def test_a_rejected_query_does_not_report_success(
        self,
        registry: KnowledgeBaseRegistry,
        settings: Settings,
        generator: DeterministicSQLGenerator,
    ) -> None:
        pipeline = self._pipeline_with_broken_database(registry, settings, generator)

        answer = pipeline.answer("How many observations failed in the last 24 hours?")

        # Previously this surfaced as succeeded=True with zero rows, which the UI
        # rendered as the affirmatively false "the query ran successfully".
        assert answer.succeeded is False
        assert any("no such table" in message for message in answer.errors)

    def test_the_failure_survives_serialisation(
        self,
        registry: KnowledgeBaseRegistry,
        settings: Settings,
        generator: DeterministicSQLGenerator,
    ) -> None:
        pipeline = self._pipeline_with_broken_database(registry, settings, generator)

        payload = pipeline.answer("Count observations by status.").to_dict()

        assert payload["succeeded"] is False
        assert payload["errors"]


class TestUngroundedQuestionsAreDeclined:
    """Retrieval has no floor, so an off-topic question still reaches a table.

    Cosine similarity always returns a nearest neighbour, however distant. Without a
    grounding check, "what is the meaning of life?" is answered with a confident,
    fully formed query about whichever table happened to rank first.
    """

    @pytest.mark.parametrize(
        "question",
        [
            "What is the meaning of life?",
            "who is the president",
            "zzzz qqqq wwww",
        ],
    )
    def test_a_question_about_nothing_is_declined(
        self, pipeline: NL2SQLPipeline, question: str
    ) -> None:
        answer = pipeline.answer(question)

        assert answer.succeeded is False
        assert answer.sql is None

    @pytest.mark.parametrize(
        "question",
        [
            "Show all failed observations in the last 24 hours.",
            "List interfaces with the highest failure count.",
            "Display observations grouped by environment.",
            "How many incidents does each team own?",
            "What is the success rate per environment over the last 7 days?",
            "Show total purchase order value by vendor.",
        ],
    )
    def test_real_questions_are_not_caught_by_the_gate(
        self, pipeline: NL2SQLPipeline, question: str
    ) -> None:
        # The gate is only worth having if it costs nothing on genuine questions.
        answer = pipeline.answer(question)

        assert answer.succeeded is True
        assert answer.row_count > 0

    def test_a_write_request_is_refused_before_retrieval(
        self, analyzer: QuestionAnalyzer
    ) -> None:
        analysis = analyzer.analyze("Delete all observations.")

        assert analysis.is_supported is False
        assert "read-only" in (analysis.rejection_reason or "")


class TestResultRenderingToleratesNulls:
    """LEFT JOINs produce NULLs, so the result renderer has to survive them."""

    @staticmethod
    def _answer_with(rows: list[dict[str, object]]) -> NL2SQLAnswer:
        return NL2SQLAnswer(
            question="q",
            succeeded=True,
            sql="SELECT 1",
            answer="",
            rows=rows,
            row_count=len(rows),
        )

    def test_a_fully_null_column_still_renders(self) -> None:
        answer = self._answer_with(
            [{"reason": None, "count": 1}, {"reason": None, "count": 2}]
        )

        frame = results_dataframe(answer)

        assert list(frame.columns) == ["Reason", "Count"]
        assert len(frame) == 2

    def test_a_single_null_still_renders(self) -> None:
        answer = self._answer_with(
            [{"device": "iad1-core-101"}, {"device": None}]
        )

        assert len(results_dataframe(answer)) == 2

    def test_no_rows_yields_an_empty_frame(self) -> None:
        assert results_dataframe(self._answer_with([])).empty
