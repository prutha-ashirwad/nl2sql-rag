"""State carried through the LangGraph workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from nl2sql.analysis.question_analyzer import QuestionAnalysis
from nl2sql.database.executor import QueryResult
from nl2sql.retrieval.context_builder import RetrievedContext
from nl2sql.validation.models import ValidationReport


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """A single step recorded while answering a question."""

    node: str
    summary: str
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def append_events(
    existing: list[TraceEvent] | None, incoming: list[TraceEvent] | None
) -> list[TraceEvent]:
    """Reducer that appends trace events instead of replacing them."""
    return [*(existing or []), *(incoming or [])]


class NL2SQLState(TypedDict, total=False):
    """Everything the workflow knows about the question in flight."""

    question: str
    analysis: QuestionAnalysis | None
    context: RetrievedContext | None

    sql: str | None
    """The current candidate query, replaced by each repair round."""

    generator: str
    explanation: str
    validation: ValidationReport | None
    repair_attempts: int
    execution: QueryResult | None

    execution_failed: bool
    """True when the database rejected the query.

    Distinct from ``execution is None``, which is also the state when execution is
    switched off entirely.
    """

    answer: str

    succeeded: bool
    """True when a valid query was produced and, if it ran, the database accepted it."""

    errors: list[str]

    tokens_used: int
    """Cumulative model token usage for this question."""

    trace: Annotated[list[TraceEvent], append_events]


def initial_state(question: str) -> NL2SQLState:
    """Build the starting state for a question."""
    return NL2SQLState(
        question=question,
        analysis=None,
        context=None,
        sql=None,
        generator="",
        explanation="",
        validation=None,
        repair_attempts=0,
        execution=None,
        execution_failed=False,
        answer="",
        succeeded=False,
        errors=[],
        tokens_used=0,
        trace=[],
    )


class StepTimer:
    """Measures how long a node took, for the trace."""

    def __init__(self) -> None:
        self._started_at = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Milliseconds since the timer was created."""
        return round((time.perf_counter() - self._started_at) * 1000.0, 2)
