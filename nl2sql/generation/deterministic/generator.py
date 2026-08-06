"""Deterministic SQL generator built on the rule-based planner."""

from __future__ import annotations

from nl2sql.analysis.question_analyzer import QuestionAnalysis
from nl2sql.dialects import get_dialect
from nl2sql.exceptions import GenerationError
from nl2sql.generation.base import GenerationResult
from nl2sql.generation.deterministic.planner import DeterministicPlanner
from nl2sql.generation.deterministic.query_plan import QueryPlan
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.logging_config import get_logger
from nl2sql.retrieval.context_builder import RetrievedContext

logger = get_logger(__name__)

GENERATOR_NAME = "deterministic-planner"


class DeterministicSQLGenerator:
    """Generates SQL by planning directly against the Knowledge Base."""

    def __init__(
        self,
        registry: KnowledgeBaseRegistry,
        *,
        dialect: str = "sqlite",
    ) -> None:
        self._planner = DeterministicPlanner(registry, get_dialect(dialect))

    @property
    def name(self) -> str:
        """Short identifier used in traces and result metadata."""
        return GENERATOR_NAME

    def generate(
        self, context: RetrievedContext, analysis: QuestionAnalysis
    ) -> GenerationResult:
        """Plan and render the query that answers the question."""
        if context.is_empty:
            return GenerationResult(
                sql=None,
                generator=self.name,
                insufficient_context=True,
                explanation=(
                    "No tables in the Knowledge Base matched this question."
                ),
            )

        try:
            plan = self._planner.plan(context, analysis)
        except GenerationError as exc:
            return GenerationResult(
                sql=None,
                generator=self.name,
                insufficient_context=True,
                explanation=str(exc),
            )

        return GenerationResult(
            sql=plan.to_sql(),
            generator=self.name,
            explanation=self._explain(plan, analysis),
            metadata={
                "base_table": plan.base_table,
                "joined_tables": ",".join(plan.referenced_tables()[1:]),
            },
        )

    def repair(
        self,
        context: RetrievedContext,
        analysis: QuestionAnalysis,
        sql: str,
        errors: list[str],
    ) -> GenerationResult:
        """Report the validation errors; the planner has no second candidate."""
        logger.info(
            "Deterministic planner cannot repair its own output; reporting %d error(s)",
            len(errors),
        )
        return GenerationResult(
            sql=None,
            generator=self.name,
            insufficient_context=True,
            explanation=(
                "The planned query did not pass validation and the deterministic "
                "planner produces a single candidate per question. "
                f"Errors: {'; '.join(errors)}"
            ),
        )

    @staticmethod
    def _explain(plan: QueryPlan, analysis: QuestionAnalysis) -> str:
        """Describe in prose how the query was constructed."""
        intent = analysis.intent.value
        article = "an" if intent[0] in "aeiou" else "a"
        parts = [
            f"Anchored the query on {plan.base_table} and interpreted the request as "
            f"{article} {intent} query."
        ]

        joined = plan.referenced_tables()[1:]
        if joined:
            parts.append(
                f"Joined {', '.join(joined)} using the join paths declared in the "
                f"Knowledge Base."
            )
        if analysis.time_window:
            parts.append(f"Restricted results to the {analysis.time_window.describe()}.")
        if analysis.value_filters:
            filters = ", ".join(
                f"{item.column} = '{item.value}'" for item in analysis.value_filters
            )
            parts.append(f"Applied filters: {filters}.")
        if plan.has_aggregate:
            parts.append("Aggregated the results and ordered them by the count.")

        return " ".join(parts)
