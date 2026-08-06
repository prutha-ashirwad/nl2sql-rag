"""The public entry point to the NL2SQL system.

Construction parses the Knowledge Base and builds both retrieval indexes, so build
one pipeline at start-up and reuse it; it is read-only afterwards and thread-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nl2sql.analysis.question_analyzer import QuestionAnalyzer
from nl2sql.config import Settings, get_settings
from nl2sql.database.engine import build_engine
from nl2sql.database.executor import QueryExecutor
from nl2sql.exceptions import ConfigurationError, NL2SQLError
from nl2sql.generation.base import SQLGenerator
from nl2sql.generation.deterministic.generator import DeterministicSQLGenerator
from nl2sql.generation.llm_generator import LLMSQLGenerator
from nl2sql.graph.nodes import NL2SQLNodes
from nl2sql.graph.state import TraceEvent, initial_state
from nl2sql.graph.workflow import build_workflow
from nl2sql.knowledge_base.loader import load_knowledge_base
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.llm.factory import build_llm_client
from nl2sql.logging_config import get_logger
from nl2sql.retrieval.context_builder import SchemaContextBuilder
from nl2sql.retrieval.hybrid_retriever import HybridRetriever
from nl2sql.tracing import run_config
from nl2sql.validation.validator import SQLValidator

logger = get_logger(__name__)


@dataclass(slots=True)
class NL2SQLAnswer:
    """The complete outcome of answering one question."""

    question: str
    succeeded: bool
    sql: str | None
    answer: str
    explanation: str = ""
    generator: str = ""
    tables_used: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    repair_attempts: int = 0
    tokens_used: int = 0
    trace: list[TraceEvent] = field(default_factory=list)

    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)
    """What the RAG step retrieved, ranked, with the score each document earned."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise the answer for an HTTP response or a log record."""
        return {
            "question": self.question,
            "succeeded": self.succeeded,
            "sql": self.sql,
            "answer": self.answer,
            "explanation": self.explanation,
            "generator": self.generator,
            "tables_used": self.tables_used,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "errors": self.errors,
            "repair_attempts": self.repair_attempts,
            "tokens_used": self.tokens_used,
            "retrieved_documents": self.retrieved_documents,
            "trace": [
                {
                    "node": event.node,
                    "summary": event.summary,
                    "duration_ms": event.duration_ms,
                    "details": event.details,
                }
                for event in self.trace
            ],
        }


class NL2SQLPipeline:
    """Converts natural-language questions into validated SQL."""

    def __init__(
        self,
        registry: KnowledgeBaseRegistry,
        settings: Settings,
        *,
        generator: SQLGenerator,
        executor: QueryExecutor | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        """Wire the pipeline together.

        Args:
            registry: Indexed Knowledge Base.
            settings: Runtime configuration.
            generator: Produces candidate SQL.
            executor: Runs validated SQL; ``None`` returns the query only.
            retriever: An already-built retriever to reuse across pipelines.
        """
        self._registry = registry
        self._settings = settings
        self._executor = executor
        self._generator = generator

        retriever = retriever or HybridRetriever(
            registry, settings=settings, lexical_weight=settings.lexical_weight
        )
        nodes = NL2SQLNodes(
            analyzer=QuestionAnalyzer(registry),
            context_builder=SchemaContextBuilder(registry, retriever, settings),
            generator=generator,
            validator=SQLValidator(registry, dialect=settings.sql_dialect),
            settings=settings,
            executor=executor,
        )
        self._workflow = build_workflow(
            nodes, settings, execute=executor is not None
        )

        logger.info(
            "Pipeline ready: %d tables, generator=%s, execution=%s",
            len(registry.table_names),
            generator.name,
            "enabled" if executor else "disabled",
        )

    @classmethod
    def create(cls, settings: Settings | None = None) -> NL2SQLPipeline:
        """Build a pipeline from configuration."""
        settings = settings or get_settings()

        knowledge_base = load_knowledge_base(settings.knowledge_base_path)
        registry = KnowledgeBaseRegistry(knowledge_base)

        deterministic = DeterministicSQLGenerator(
            registry,
            dialect=settings.sql_dialect,
        )
        generator = cls._build_generator(settings, deterministic)
        executor = cls._build_executor(settings)

        return cls(registry, settings, generator=generator, executor=executor)

    @staticmethod
    def _build_generator(
        settings: Settings, deterministic: DeterministicSQLGenerator
    ) -> SQLGenerator:
        """Build the model-backed generator, or the planner if it is unavailable."""
        try:
            llm_client = build_llm_client(settings)
        except ConfigurationError as exc:
            logger.warning(
                "Falling back to the deterministic planner; the %s provider could "
                "not be configured: %s",
                settings.resolve_provider().value,
                exc,
            )
            return deterministic

        if llm_client is None:
            return deterministic

        return LLMSQLGenerator(
            llm_client, dialect=settings.sql_dialect, fallback=deterministic
        )

    @staticmethod
    def _build_executor(settings: Settings) -> QueryExecutor | None:
        """Build the query executor, or ``None`` when execution is unavailable."""
        if not settings.execute_queries:
            return None

        try:
            engine = build_engine(settings.database_url)
            return QueryExecutor(engine, max_rows=settings.max_result_rows)
        except Exception as exc:  # noqa: BLE001 - degrade instead of failing start-up
            logger.warning(
                "Query execution disabled; the database is unavailable: %s", exc
            )
            return None

    @property
    def registry(self) -> KnowledgeBaseRegistry:
        """The Knowledge Base indexes backing this pipeline."""
        return self._registry

    @property
    def settings(self) -> Settings:
        """The settings this pipeline was built with."""
        return self._settings

    @property
    def generator_name(self) -> str:
        """Name of the generator actually in use, which can differ from the config."""
        return self._generator.name

    def answer(
        self,
        question: str,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NL2SQLAnswer:
        """Answer ``question`` end to end.

        Args:
            question: The natural-language question.
            tags: Labels attached to the LangSmith trace, for grouping runs.
            metadata: Extra fields recorded on the trace alongside the configuration.

        Raises:
            NL2SQLError: only for unexpected internal failures. Expected outcomes are
                reported on the returned answer instead.
        """
        if not question or not question.strip():
            raise NL2SQLError("The question must not be empty")

        logger.info("Answering: %s", question)
        config = run_config(
            self._settings,
            generator_name=self._generator.name,
            tags=tags,
            metadata=metadata,
        )
        final_state = self._workflow.invoke(initial_state(question.strip()), config)
        return self._to_answer(question, final_state)

    @staticmethod
    def _to_answer(question: str, state: dict[str, Any]) -> NL2SQLAnswer:
        """Project the final workflow state onto the public answer type."""
        report = state.get("validation")
        execution = state.get("execution")
        context = state.get("context")

        return NL2SQLAnswer(
            question=question,
            succeeded=bool(state.get("succeeded")),
            sql=state.get("sql"),
            answer=state.get("answer", ""),
            explanation=state.get("explanation", ""),
            generator=state.get("generator", ""),
            tables_used=(
                report.referenced_tables
                if report
                else (context.table_names if context else [])
            ),
            columns=execution.columns if execution else [],
            rows=execution.rows if execution else [],
            row_count=execution.row_count if execution else 0,
            truncated=execution.truncated if execution else False,
            validation_errors=report.error_messages() if report else [],
            validation_warnings=(
                [issue.format() for issue in report.warnings] if report else []
            ),
            errors=list(state.get("errors", [])),
            repair_attempts=state.get("repair_attempts", 0),
            tokens_used=state.get("tokens_used", 0),
            trace=list(state.get("trace", [])),
            retrieved_documents=(
                [
                    {
                        "id": scored.document.id,
                        "kind": scored.kind.value,
                        "score": scored.score,
                        "retriever": scored.retriever,
                        "tables": list(scored.document.tables),
                    }
                    for scored in context.documents
                ]
                if context
                else []
            ),
        )
