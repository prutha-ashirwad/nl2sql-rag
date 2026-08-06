"""Workflow node implementations.

Nodes never raise for expected failures — problems are recorded in the state so the
routing functions can decide what happens next.
"""

from __future__ import annotations

from collections.abc import Callable

from nl2sql.analysis.question_analyzer import QuestionAnalyzer
from nl2sql.config import Settings
from nl2sql.database.executor import QueryExecutor
from nl2sql.exceptions import ExecutionError
from nl2sql.generation.base import SQLGenerator
from nl2sql.graph.state import NL2SQLState, StepTimer, TraceEvent
from nl2sql.logging_config import get_logger
from nl2sql.retrieval.context_builder import SchemaContextBuilder
from nl2sql.validation.validator import SQLValidator

logger = get_logger(__name__)

NODE_ANALYZE = "analyze_question"
NODE_RETRIEVE = "retrieve_context"
NODE_GENERATE = "generate_sql"
NODE_VALIDATE = "validate_sql"
NODE_REPAIR = "repair_sql"
NODE_EXECUTE = "execute_sql"
NODE_FINALIZE = "finalize"


class NL2SQLNodes:
    """The callable nodes that make up the NL2SQL workflow."""

    def __init__(
        self,
        analyzer: QuestionAnalyzer,
        context_builder: SchemaContextBuilder,
        generator: SQLGenerator,
        validator: SQLValidator,
        settings: Settings,
        executor: QueryExecutor | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._context_builder = context_builder
        self._generator = generator
        self._validator = validator
        self._settings = settings
        self._executor = executor

    # -- Nodes ----------------------------------------------------------------

    def analyze_question(self, state: NL2SQLState) -> NL2SQLState:
        """Extract structured intent from the question."""
        timer = StepTimer()
        question = state["question"]
        analysis = self._analyzer.analyze(question)

        errors = list(state.get("errors", []))
        if not analysis.is_supported and analysis.rejection_reason:
            errors.append(analysis.rejection_reason)

        return NL2SQLState(
            analysis=analysis,
            errors=errors,
            trace=[
                TraceEvent(
                    node=NODE_ANALYZE,
                    summary=f"Classified the question as {analysis.intent.value}",
                    duration_ms=timer.elapsed_ms(),
                    details={
                        "intent": analysis.intent.value,
                        "tables": analysis.mentioned_tables,
                        "filters": len(analysis.value_filters),
                        "time_window": (
                            analysis.time_window.describe()
                            if analysis.time_window
                            else None
                        ),
                    },
                )
            ],
        )

    def retrieve_context(self, state: NL2SQLState) -> NL2SQLState:
        """Retrieve and assemble Knowledge Base context for the question."""
        timer = StepTimer()
        context = self._context_builder.build(state["question"], state.get("analysis"))

        errors = list(state.get("errors", []))
        if context.is_empty:
            errors.append(
                "No relevant tables were found in the Knowledge Base for this question."
            )

        return NL2SQLState(
            context=context,
            errors=errors,
            trace=[
                TraceEvent(
                    node=NODE_RETRIEVE,
                    summary=(
                        f"Retrieved {len(context.documents)} document(s) covering "
                        f"{len(context.tables)} table(s)"
                    ),
                    duration_ms=timer.elapsed_ms(),
                    details={
                        "base_table": context.base_table,
                        "tables": context.table_names,
                        "joins": len(context.join_steps),
                        "examples": len(context.examples),
                    },
                )
            ],
        )

    def generate_sql(self, state: NL2SQLState) -> NL2SQLState:
        """Produce a first-pass candidate query."""
        timer = StepTimer()
        context = state["context"]
        analysis = state["analysis"]

        result = self._generator.generate(context, analysis)

        errors = list(state.get("errors", []))
        if not result.succeeded:
            errors.append(
                result.explanation or "No SQL query could be generated."
            )

        return NL2SQLState(
            sql=result.sql,
            generator=result.generator,
            explanation=result.explanation,
            tokens_used=state.get("tokens_used", 0) + result.tokens_used,
            errors=errors,
            trace=[
                TraceEvent(
                    node=NODE_GENERATE,
                    summary=(
                        f"Generated a candidate query with {result.generator}"
                        if result.succeeded
                        else "Could not generate a query"
                    ),
                    duration_ms=timer.elapsed_ms(),
                    details={
                        "generator": result.generator,
                        "insufficient_context": result.insufficient_context,
                        "tokens": result.tokens_used,
                    },
                )
            ],
        )

    def validate_sql(self, state: NL2SQLState) -> NL2SQLState:
        """Validate the candidate query against the Knowledge Base."""
        timer = StepTimer()
        sql = state.get("sql") or ""
        report = self._validator.validate(sql)

        update = NL2SQLState(validation=report)

        return NL2SQLState(
            **update,
            trace=[
                TraceEvent(
                    node=NODE_VALIDATE,
                    summary=(
                        "Query passed validation"
                        if report.is_valid
                        else f"Query failed validation with {len(report.errors)} error(s)"
                    ),
                    duration_ms=timer.elapsed_ms(),
                    details={
                        "valid": report.is_valid,
                        "errors": report.error_messages(),
                        "warnings": [issue.format() for issue in report.warnings],
                        "tables": report.referenced_tables,
                    },
                )
            ],
        )

    def repair_sql(self, state: NL2SQLState) -> NL2SQLState:
        """Ask the generator to correct a query that failed validation."""
        timer = StepTimer()
        report = state["validation"]
        attempt = state.get("repair_attempts", 0) + 1

        result = self._generator.repair(
            state["context"],
            state["analysis"],
            state.get("sql") or "",
            report.error_messages(),
        )

        errors = list(state.get("errors", []))
        if not result.succeeded:
            errors.append(result.explanation or "The query could not be repaired.")

        return NL2SQLState(
            sql=result.sql if result.succeeded else state.get("sql"),
            repair_attempts=attempt,
            explanation=result.explanation or state.get("explanation", ""),
            tokens_used=state.get("tokens_used", 0) + result.tokens_used,
            errors=errors,
            trace=[
                TraceEvent(
                    node=NODE_REPAIR,
                    summary=f"Repair attempt {attempt} of "
                    f"{self._settings.max_repair_attempts}",
                    duration_ms=timer.elapsed_ms(),
                    details={
                        "attempt": attempt,
                        "repaired": result.succeeded,
                        "errors_addressed": report.error_messages(),
                    },
                )
            ],
        )

    def execute_sql(self, state: NL2SQLState) -> NL2SQLState:
        """Run the validated query against the analytics database."""
        timer = StepTimer()

        if self._executor is None:
            return NL2SQLState(
                trace=[
                    TraceEvent(
                        node=NODE_EXECUTE,
                        summary="Execution is disabled; returning the query only",
                        duration_ms=timer.elapsed_ms(),
                    )
                ]
            )

        errors = list(state.get("errors", []))
        try:
            result = self._executor.execute(state["sql"])
        except ExecutionError as exc:
            logger.warning("Execution failed: %s", exc)
            errors.append(str(exc))
            return NL2SQLState(
                errors=errors,
                execution_failed=True,
                trace=[
                    TraceEvent(
                        node=NODE_EXECUTE,
                        summary="Query execution failed",
                        duration_ms=timer.elapsed_ms(),
                        details={"error": str(exc)},
                    )
                ],
            )

        return NL2SQLState(
            execution=result,
            trace=[
                TraceEvent(
                    node=NODE_EXECUTE,
                    summary=f"Query returned {result.row_count} row(s)",
                    duration_ms=timer.elapsed_ms(),
                    details={
                        "row_count": result.row_count,
                        "truncated": result.truncated,
                        "duration_ms": result.duration_ms,
                    },
                )
            ],
        )

    def finalize(self, state: NL2SQLState) -> NL2SQLState:
        """Compose the final answer from whatever the workflow achieved."""
        timer = StepTimer()
        report = state.get("validation")
        succeeded = (
            bool(state.get("sql"))
            and bool(report and report.is_valid)
            and not state.get("execution_failed", False)
        )

        answer = self._compose_answer(state, succeeded=succeeded)

        return NL2SQLState(
            succeeded=succeeded,
            answer=answer,
            trace=[
                TraceEvent(
                    node=NODE_FINALIZE,
                    summary="Answered successfully" if succeeded else "Could not answer",
                    duration_ms=timer.elapsed_ms(),
                )
            ],
        )

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _compose_answer(state: NL2SQLState, *, succeeded: bool) -> str:
        """Build the user-facing message for the current state."""
        if not succeeded:
            errors = state.get("errors", [])
            reason = errors[-1] if errors else "The question could not be answered."
            return f"I could not produce a reliable SQL query. {reason}"

        parts: list[str] = []
        execution = state.get("execution")

        if execution is not None:
            if execution.is_empty:
                parts.append(
                    "The query is valid but returned no rows for the current data."
                )
            else:
                suffix = " (truncated)" if execution.truncated else ""
                parts.append(
                    f"The query returned {execution.row_count} row(s){suffix}."
                )

        explanation = state.get("explanation", "").strip()
        if explanation:
            parts.append(explanation)

        report = state.get("validation")
        if report and report.warnings:
            warnings = "; ".join(issue.format() for issue in report.warnings)
            parts.append(f"Validation notes: {warnings}")

        return " ".join(parts) if parts else "The query is ready to run."


def route_after_analysis(state: NL2SQLState) -> str:
    """Skip straight to the end when the question cannot be served."""
    analysis = state.get("analysis")
    if analysis is None or not analysis.is_supported:
        return NODE_FINALIZE
    return NODE_RETRIEVE


def route_after_retrieval(state: NL2SQLState) -> str:
    """Only generate SQL when retrieval found usable schema."""
    context = state.get("context")
    if context is None or context.is_empty:
        return NODE_FINALIZE
    return NODE_GENERATE


def route_after_generation(state: NL2SQLState) -> str:
    """Only validate when a candidate query exists."""
    return NODE_VALIDATE if state.get("sql") else NODE_FINALIZE


def build_validation_router(
    settings: Settings, execute: bool
) -> Callable[[NL2SQLState], str]:
    """Build the validation router, which routes to execution when ``execute``."""

    def route_after_validation(state: NL2SQLState) -> str:
        report = state.get("validation")

        if report is not None and report.is_valid:
            return NODE_EXECUTE if execute else NODE_FINALIZE

        if state.get("repair_attempts", 0) < settings.max_repair_attempts:
            return NODE_REPAIR

        logger.info("Repair budget exhausted; returning the validation errors")
        return NODE_FINALIZE

    return route_after_validation
