"""Run the evaluation cases and score what came back.

Scoring is by *execution*, not by string similarity: the reference query and the
generated query are both run, and their result sets are compared. A query that
reads differently from the reference but returns the same rows is correct, which
is the only definition that survives contact with a second SQL dialect.
"""

from __future__ import annotations

import time
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.engine import Engine

from nl2sql.config import Settings, get_settings
from nl2sql.database.engine import build_engine
from nl2sql.database.executor import QueryExecutor
from nl2sql.evaluation.models import (
    CaseOutcome,
    EvaluationCase,
    EvaluationReport,
    Expectation,
    Verdict,
)
from nl2sql.exceptions import NL2SQLError
from nl2sql.logging_config import get_logger
from nl2sql.pipeline import NL2SQLPipeline

logger = get_logger(__name__)

DEFAULT_CASES_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "cases.yaml"


def load_cases(path: Path | None = None) -> list[EvaluationCase]:
    """Read the case file. Raises if a case cannot be scored as written."""
    source = path or DEFAULT_CASES_PATH
    payload = yaml.safe_load(source.read_text()) or {}
    cases = [
        EvaluationCase(
            id=entry["id"],
            question=entry["question"],
            category=entry["category"],
            expect=Expectation(entry.get("expect", "answered")),
            gold_sql=entry.get("gold_sql"),
            required_tables=list(entry.get("required_tables", [])),
            note=entry.get("note", ""),
        )
        for entry in payload.get("cases", [])
    ]
    seen = Counter(case.id for case in cases)
    duplicates = [case_id for case_id, n in seen.items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate case ids: {', '.join(sorted(duplicates))}")
    return cases


def _scalar(value: Any) -> str:
    """Render one cell so that 1, 1.0 and Decimal('1.0') compare equal."""
    if value is None:
        return "∅"
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:.6f}"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).strip()


def _normalise(name: str) -> str:
    """Strip the table qualifier and case from a column name."""
    return name.split(".")[-1].strip().lower()


def _multiset(rows: list[dict[str, Any]], columns: list[str]) -> Counter[tuple[str, ...]]:
    """Rows reduced to a multiset of value tuples over ``columns``."""
    return Counter(tuple(_scalar(row.get(col)) for col in columns) for row in rows)


def _compare(
    gold_rows: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> tuple[Verdict, str]:
    """Score a generated result against the reference result.

    Column names are matched after normalisation; a reference column with no
    counterpart is a failure even if the values exist under another alias. That
    is deliberately conservative — it can only under-report accuracy, never
    inflate it.
    """
    if not gold_rows:
        return (
            (Verdict.EXACT, "both empty")
            if not rows
            else (Verdict.WRONG_ROWS, f"reference is empty, got {len(rows)} row(s)")
        )

    gold_columns = list(gold_rows[0].keys())
    available = {_normalise(name): name for name in rows[0]} if rows else {}

    missing = [c for c in gold_columns if _normalise(c) not in available]
    if missing:
        return Verdict.WRONG_ROWS, f"missing column(s): {', '.join(missing)}"

    mapped = [available[_normalise(c)] for c in gold_columns]
    expected = _multiset(gold_rows, gold_columns)
    actual = _multiset(rows, mapped)

    if expected != actual:
        short = sum((expected - actual).values())
        extra = sum((actual - expected).values())
        return (
            Verdict.WRONG_ROWS,
            f"{len(gold_rows)} reference row(s) vs {len(rows)}: "
            f"{short} missing, {extra} unexpected",
        )

    same_shape = len(rows) and len(rows[0]) == len(gold_columns)
    if same_shape:
        return Verdict.EXACT, "identical result set"
    return Verdict.ANSWER_SET, "reference columns all present and equal"


def _run_gold(executor: QueryExecutor, sql: str) -> tuple[list[dict[str, Any]], str]:
    """Execute a reference query, returning its rows or the reason it failed."""
    try:
        return executor.execute(sql).rows, ""
    except (NL2SQLError, Exception) as exc:  # noqa: BLE001 - reported, not raised
        return [], str(exc)


def evaluate(
    cases: list[EvaluationCase],
    *,
    settings: Settings | None = None,
    engine: Engine | None = None,
    pipeline: NL2SQLPipeline | None = None,
    on_case: Any = None,
) -> EvaluationReport:
    """Answer every case and score the result. One pipeline is reused throughout."""
    settings = settings or get_settings()
    pipeline = pipeline or NL2SQLPipeline.create(settings)
    engine = engine or build_engine(settings.database_url)
    executor = QueryExecutor(engine)

    report = EvaluationReport(engine=pipeline.generator_name)

    for case in cases:
        started = time.perf_counter()
        answer = pipeline.answer(case.question, tags=["evaluation", case.category])
        elapsed = (time.perf_counter() - started) * 1000.0

        outcome = _score(case, answer, executor)
        outcome.duration_ms = round(elapsed, 2)
        report.outcomes.append(outcome)

        if on_case is not None:
            on_case(outcome)

    return report


def _score(case: EvaluationCase, answer: Any, executor: QueryExecutor) -> CaseOutcome:
    """Turn one answer into a verdict."""
    if case.expect is Expectation.REFUSED:
        refused = not answer.succeeded
        return CaseOutcome(
            case=case,
            verdict=Verdict.REFUSED_CORRECTLY if refused else Verdict.ANSWERED_WRONGLY,
            detail="declined" if refused else "answered a question it should decline",
            generated_sql=answer.sql,
            row_count=answer.row_count,
            repair_attempts=answer.repair_attempts,
        )

    gold_rows, gold_error = _run_gold(executor, case.gold_sql or "")
    if gold_error:
        return CaseOutcome(
            case=case,
            verdict=Verdict.GOLD_BROKEN,
            detail=f"reference query failed: {gold_error}",
        )

    recalled = _tables_recalled(case, answer)

    if not answer.succeeded or not answer.sql:
        reason = "; ".join(answer.validation_errors or answer.errors) or "no SQL produced"
        return CaseOutcome(
            case=case,
            verdict=Verdict.NOT_EXECUTED,
            detail=reason,
            generated_sql=answer.sql,
            gold_row_count=len(gold_rows),
            repair_attempts=answer.repair_attempts,
            tables_recalled=recalled,
        )

    verdict, detail = _compare(gold_rows, answer.rows)
    return CaseOutcome(
        case=case,
        verdict=verdict,
        detail=detail,
        generated_sql=answer.sql,
        row_count=answer.row_count,
        gold_row_count=len(gold_rows),
        repair_attempts=answer.repair_attempts,
        tables_recalled=recalled,
    )


def _tables_recalled(case: EvaluationCase, answer: Any) -> bool:
    """True when every table the question needs was actually read."""
    if not case.required_tables:
        return True
    used = {_normalise(name) for name in answer.tables_used}
    return all(_normalise(name) in used for name in case.required_tables)
