"""Read-only execution of validated queries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

from nl2sql.exceptions import ExecutionError
from nl2sql.logging_config import get_logger

logger = get_logger(__name__)

_ALLOWED_PREFIXES = ("select", "with")


@dataclass(slots=True)
class QueryResult:
    """Rows returned by a successfully executed query."""

    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    duration_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        """True when the query returned no rows."""
        return self.row_count == 0


class QueryExecutor:
    """Executes validated SELECT statements against the analytics database."""

    def __init__(self, engine: Engine, *, max_rows: int | None = None) -> None:
        """``max_rows`` is an optional safety valve; ``None`` returns every row."""
        self._engine = engine
        self._max_rows = max_rows

    def execute(self, sql: str) -> QueryResult:
        """Run ``sql`` and return its rows.

        Every matching row is returned unless ``max_rows`` was configured, in which
        case the result is capped and flagged as truncated.

        Raises:
            ExecutionError: if the statement is not read-only or the database
                rejects it.
        """
        self._assert_read_only(sql)

        started_at = time.perf_counter()
        try:
            with self._engine.connect() as connection:
                cursor = connection.execute(text(sql))
                columns = list(cursor.keys())
                if self._max_rows is None:
                    fetched = cursor.fetchall()
                else:
                    # One extra row is fetched purely to detect truncation.
                    fetched = cursor.fetchmany(self._max_rows + 1)
        except SQLAlchemyError as exc:
            raise ExecutionError(f"Query execution failed: {exc}") from exc

        duration_ms = (time.perf_counter() - started_at) * 1000.0
        truncated = self._max_rows is not None and len(fetched) > self._max_rows
        kept = fetched if self._max_rows is None else fetched[: self._max_rows]
        rows = [dict(zip(columns, row, strict=False)) for row in kept]

        logger.debug(
            "Query returned %d row(s) in %.1f ms (truncated=%s)",
            len(rows),
            duration_ms,
            truncated,
        )

        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            duration_ms=round(duration_ms, 2),
        )

    @staticmethod
    def _assert_read_only(sql: str) -> None:
        """Reject anything that is not a single read-only statement."""
        stripped = sql.strip().rstrip(";").strip()

        if not stripped.lower().startswith(_ALLOWED_PREFIXES):
            raise ExecutionError(
                "Only SELECT and WITH statements may be executed."
            )

        if ";" in stripped:
            raise ExecutionError("Multiple statements cannot be executed.")
