"""The SQL generation interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from nl2sql.analysis.question_analyzer import QuestionAnalysis
from nl2sql.retrieval.context_builder import RetrievedContext


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The outcome of one attempt to turn a question into SQL."""

    sql: str | None
    generator: str
    explanation: str = ""
    tokens_used: int = 0
    insufficient_context: bool = False
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """True when a candidate query was produced."""
        return bool(self.sql and self.sql.strip())


@runtime_checkable
class SQLGenerator(Protocol):
    """Produces candidate SQL from a question and its retrieved context."""

    @property
    def name(self) -> str:
        """Short identifier used in traces and result metadata."""
        ...

    def generate(
        self, context: RetrievedContext, analysis: QuestionAnalysis
    ) -> GenerationResult:
        """Produce a first-pass query for the question in ``context``."""
        ...

    def repair(
        self,
        context: RetrievedContext,
        analysis: QuestionAnalysis,
        sql: str,
        errors: list[str],
    ) -> GenerationResult:
        """Produce a corrected query given the validation errors on ``sql``."""
        ...
