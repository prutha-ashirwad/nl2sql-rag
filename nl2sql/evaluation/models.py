"""What an evaluation case declares, and what running one produces."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Expectation(str, Enum):
    """What a case asserts about the system's response."""

    ANSWERED = "answered"
    """The question is answerable; ``gold_sql`` defines the correct answer set."""

    REFUSED = "refused"
    """The question must be declined rather than answered with a guess."""


class Verdict(str, Enum):
    """The outcome of one case."""

    EXACT = "exact"
    """Result set is identical to the reference, column for column."""

    ANSWER_SET = "answer_set"
    """Every reference column and row is present; extra descriptive columns allowed."""

    WRONG_ROWS = "wrong_rows"
    """Query ran, but the rows it returned are not the reference rows."""

    NOT_EXECUTED = "not_executed"
    """No SQL ran: the system declined, failed validation, or produced nothing."""

    REFUSED_CORRECTLY = "refused_correctly"
    """The system declined a question it was supposed to decline."""

    ANSWERED_WRONGLY = "answered_wrongly"
    """The system answered a question it was supposed to decline."""

    GOLD_BROKEN = "gold_broken"
    """The reference query itself failed to run — the case is at fault, not the system."""


PASSING = frozenset({Verdict.EXACT, Verdict.ANSWER_SET, Verdict.REFUSED_CORRECTLY})


@dataclass(frozen=True)
class EvaluationCase:
    """One question, and what a correct response to it looks like."""

    id: str
    question: str
    category: str
    expect: Expectation = Expectation.ANSWERED
    gold_sql: str | None = None
    required_tables: list[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        """Reject a case that cannot be scored."""
        if self.expect is Expectation.ANSWERED and not self.gold_sql:
            raise ValueError(f"{self.id}: an answered case needs gold_sql")
        if self.expect is Expectation.REFUSED and self.gold_sql:
            raise ValueError(f"{self.id}: a refused case must not carry gold_sql")


@dataclass
class CaseOutcome:
    """What happened when one case was run."""

    case: EvaluationCase
    verdict: Verdict
    detail: str = ""
    generated_sql: str | None = None
    row_count: int = 0
    gold_row_count: int = 0
    repair_attempts: int = 0
    tables_recalled: bool = True
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        """True when this case counts towards the accuracy figure."""
        return self.verdict in PASSING


@dataclass
class EvaluationReport:
    """Every outcome from one run, plus the totals worth quoting."""

    engine: str
    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Cases run, excluding any whose own reference query is broken."""
        return sum(1 for o in self.outcomes if o.verdict is not Verdict.GOLD_BROKEN)

    @property
    def passed(self) -> int:
        """Cases that produced the correct answer, or the correct refusal."""
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def accuracy(self) -> float:
        """Passing cases as a fraction of cases run."""
        return self.passed / self.total if self.total else 0.0

    @property
    def broken_cases(self) -> list[CaseOutcome]:
        """Cases excluded because the reference query would not run."""
        return [o for o in self.outcomes if o.verdict is Verdict.GOLD_BROKEN]

    def by_category(self) -> dict[str, tuple[int, int]]:
        """Passed and total per category, so weak shapes are visible."""
        tally: dict[str, list[int]] = {}
        for outcome in self.outcomes:
            if outcome.verdict is Verdict.GOLD_BROKEN:
                continue
            entry = tally.setdefault(outcome.case.category, [0, 0])
            entry[0] += int(outcome.passed)
            entry[1] += 1
        return {name: (hit, run) for name, (hit, run) in sorted(tally.items())}

    def count(self, verdict: Verdict) -> int:
        """How many cases ended with ``verdict``."""
        return sum(1 for o in self.outcomes if o.verdict is verdict)

    @property
    def table_recall(self) -> float | None:
        """Share of cases where every required table was queried, or None if untested."""
        scored = [
            o
            for o in self.outcomes
            if o.case.required_tables and o.verdict is not Verdict.GOLD_BROKEN
        ]
        if not scored:
            return None
        return sum(1 for o in scored if o.tables_recalled) / len(scored)
