"""Tests for the Streamlit interface's logic.

The renderers are exercised through their pure helpers rather than by driving a
browser: what is worth pinning down is the shaping of data — column names, NULL
handling, step labelling, engine resolution — not that Streamlit can draw a table.
"""

from __future__ import annotations

import pytest

from nl2sql.config import LLMProvider, Settings
from nl2sql.pipeline import NL2SQLAnswer, NL2SQLPipeline
from nl2sql.ui import components
from nl2sql.ui.state import PLANNER_LABEL, Workspace, unavailable_label
from nl2sql.ui.views.ask import EXAMPLE_GROUPS
from tests.conftest import build_test_settings


@pytest.fixture(scope="module")
def answer(pipeline: NL2SQLPipeline) -> NL2SQLAnswer:
    return pipeline.answer("Display observations grouped by environment.")


class TestResultRendering:
    def test_column_names_are_humanised(self, answer: NL2SQLAnswer) -> None:
        frame = components.results_dataframe(answer)

        assert "Environment Name" in frame.columns
        assert all("_" not in str(name) for name in frame.columns)

    def test_display_is_capped(self, answer: NL2SQLAnswer) -> None:
        assert len(components.results_dataframe(answer)) <= components.MAX_DISPLAY_ROWS

    def test_no_rows_yields_an_empty_frame(self) -> None:
        empty = NL2SQLAnswer(question="q", succeeded=True, sql=None, answer="")

        assert components.results_dataframe(empty).empty


class TestStepLabelling:
    """Workflow node names are internal; the interface shows plain language."""

    def test_every_node_has_a_title_and_an_icon(
        self, answer: NL2SQLAnswer
    ) -> None:
        for event in answer.trace:
            assert components.step_title(event.node) != event.node
            assert components.step_icon(event.node) != "•"

    def test_an_unknown_node_still_renders(self) -> None:
        # A node added later must not render as a blank row while the mapping
        # catches up.
        assert components.step_title("archive_result") == "Archive result"
        assert components.step_icon("archive_result") == "•"


class TestRetrievedContextIsCarriedOnTheAnswer:
    """The Agent Execution page shows what RAG retrieved, so the answer must hold it."""

    def test_documents_are_recorded_with_their_scores(
        self, answer: NL2SQLAnswer
    ) -> None:
        assert answer.retrieved_documents

        first = answer.retrieved_documents[0]
        assert {"id", "kind", "score", "retriever", "tables"} <= set(first)
        # Ranked best first, which is what the page renders as rank 1.
        scores = [document["score"] for document in answer.retrieved_documents]
        assert scores == sorted(scores, reverse=True)

    def test_it_survives_serialisation(self, answer: NL2SQLAnswer) -> None:
        assert answer.to_dict()["retrieved_documents"]


class TestSchemaTable:
    def test_one_row_per_declared_table(self, registry) -> None:  # noqa: ANN001
        frame = components.schema_dataframe(registry)

        assert len(frame) == len(registry.tables)
        assert "What it holds" in frame.columns


class TestEngineResolution:
    """Which engines the picker offers, and which one a visitor gets."""

    @staticmethod
    def _workspace(settings: Settings) -> Workspace:
        workspace = Workspace(settings=settings)
        workspace._build_engines()  # noqa: SLF001 - construction
        return workspace

    def test_the_planner_is_always_available(self, settings: Settings) -> None:
        workspace = self._workspace(settings)

        # With no credentials the app must still answer; that is a supported mode.
        assert PLANNER_LABEL in workspace.available
        assert workspace.default_engine() == PLANNER_LABEL
        assert workspace.ready_model_count == 0

    def test_models_are_listed_but_disabled_without_a_key(
        self, settings: Settings
    ) -> None:
        workspace = self._workspace(settings)

        # Listed rather than absent, so what is missing is visible.
        label = unavailable_label("gpt-4o", LLMProvider.OPENAI)
        assert label in workspace.engines
        assert workspace.engines[label] is None
        assert "OPENAI_API_KEY" in label

    def test_asking_records_the_run_for_other_pages(self, settings: Settings) -> None:
        workspace = self._workspace(settings)

        record = workspace.ask("Count observations by status.", PLANNER_LABEL)

        # The Agent Execution page renders from this rather than re-running.
        assert workspace.last_run is record
        assert record.engine == PLANNER_LABEL
        assert record.elapsed_seconds >= 0

    def test_an_unknown_dialect_is_refused_when_switching(
        self, settings: Settings
    ) -> None:
        from nl2sql.exceptions import ConfigurationError

        workspace = self._workspace(settings)

        with pytest.raises(ConfigurationError, match="Unknown SQL dialect"):
            workspace.switch_database(settings.database_url, "teradata")


class TestExampleQuestions:
    def test_the_brief_questions_are_all_offered(self) -> None:
        offered = {question for group in EXAMPLE_GROUPS.values() for question in group}

        # The three named in the brief are the acceptance criteria, so a reviewer
        # must be able to reach them without typing.
        assert "Show all failed observations in the last 24 hours." in offered
        assert "List interfaces with the highest failure count." in offered
        assert "Display observations grouped by environment." in offered

    def test_every_example_is_answerable(self, pipeline: NL2SQLPipeline) -> None:
        for group in EXAMPLE_GROUPS.values():
            for question in group:
                assert pipeline.answer(question).succeeded, question


class TestWorkspaceSettings:
    def test_the_database_is_described_without_its_password(self) -> None:
        settings = build_test_settings(
            database_url="postgresql://admin:hunter2@db.internal:5432/observability"
        )
        workspace = Workspace(settings=settings)

        described = workspace.describe_database()

        assert "hunter2" not in described
        assert "db.internal" in described


class TestTruncationIsAnnounced:
    """A capped result must not read as the complete answer.

    Queries carry a LIMIT only when the question asked for one, so the executor's row
    cap is what bounds everything else — and the caller has to be told when it bit.
    """

    @staticmethod
    def _answer(*, truncated: bool, rows: int) -> NL2SQLAnswer:
        return NL2SQLAnswer(
            question="Show all broken observations and i need all the rows",
            succeeded=True,
            sql="SELECT o.observation_id FROM observations o",
            answer=f"The query returned {rows} row(s).",
            row_count=rows,
            truncated=truncated,
        )

    def test_a_truncated_result_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, str]] = []
        monkeypatch.setattr(components.st, "warning", lambda m: seen.append(("warn", m)))
        monkeypatch.setattr(components.st, "success", lambda m: seen.append(("ok", m)))

        components.render_outcome(self._answer(truncated=True, rows=200))

        assert seen and seen[0][0] == "warn"
        assert "first 200 rows" in seen[0][1]

    def test_a_complete_result_does_not(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, str]] = []
        monkeypatch.setattr(components.st, "warning", lambda m: seen.append(("warn", m)))
        monkeypatch.setattr(components.st, "success", lambda m: seen.append(("ok", m)))

        components.render_outcome(self._answer(truncated=False, rows=12))

        assert seen and seen[0][0] == "ok"
        assert "12 results" in seen[0][1]
