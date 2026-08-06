"""Result types produced by SQL validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """How serious a validation finding is."""

    ERROR = "error"
    """The query is wrong or unsafe and must not be executed."""

    WARNING = "warning"
    """The query will run, but is likely to surprise the reader."""


class IssueCode(str, Enum):
    """Stable identifiers for each class of validation finding."""

    SYNTAX_ERROR = "syntax_error"
    EMPTY_QUERY = "empty_query"
    NOT_READ_ONLY = "not_read_only"
    MULTIPLE_STATEMENTS = "multiple_statements"
    UNKNOWN_TABLE = "unknown_table"
    UNKNOWN_COLUMN = "unknown_column"
    AMBIGUOUS_COLUMN = "ambiguous_column"
    UNDECLARED_JOIN = "undeclared_join"
    INVALID_ENUM_VALUE = "invalid_enum_value"
    MISSING_JOIN_CONDITION = "missing_join_condition"
    MISSING_LIMIT = "missing_limit"
    SELECT_STAR = "select_star"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single finding raised against a generated query."""

    code: IssueCode
    severity: Severity
    message: str
    hint: str = ""

    def format(self) -> str:
        """Render the issue as one line for a prompt or a log."""
        return f"{self.message} {self.hint}".strip()


@dataclass(slots=True)
class ValidationReport:
    """The full outcome of validating one query."""

    sql: str
    issues: list[ValidationIssue] = field(default_factory=list)
    referenced_tables: list[str] = field(default_factory=list)
    referenced_columns: list[str] = field(default_factory=list)
    normalised_sql: str | None = None

    @property
    def errors(self) -> list[ValidationIssue]:
        """Findings that block execution."""
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Findings that do not block execution."""
        return [issue for issue in self.issues if issue.severity is Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        """True when nothing blocks execution."""
        return not self.errors

    def error_messages(self) -> list[str]:
        """Blocking findings rendered for a repair prompt."""
        return [issue.format() for issue in self.errors]

    def add(
        self,
        code: IssueCode,
        severity: Severity,
        message: str,
        hint: str = "",
    ) -> None:
        """Record a finding."""
        self.issues.append(
            ValidationIssue(code=code, severity=severity, message=message, hint=hint)
        )
