"""Assembling retrieved documents into grounded generation context."""

from __future__ import annotations

from dataclasses import dataclass, field

from nl2sql.analysis.question_analyzer import QuestionAnalysis
from nl2sql.config import Settings
from nl2sql.knowledge_base.models import ExampleQuery, SQLRule, TableMetadata
from nl2sql.knowledge_base.registry import JoinStep, KnowledgeBaseRegistry
from nl2sql.logging_config import get_logger
from nl2sql.retrieval.base import DocumentKind, ScoredDocument
from nl2sql.retrieval.document_builder import build_table_document
from nl2sql.retrieval.hybrid_retriever import HybridRetriever

logger = get_logger(__name__)

_FACT_TAG = "fact"

MAX_EXAMPLES_IN_CONTEXT = 3
MAX_TABLES_IN_CONTEXT = 8


@dataclass(slots=True)
class RetrievedContext:
    """Everything the generator is allowed to rely on for one question."""

    question: str
    documents: list[ScoredDocument] = field(default_factory=list)
    tables: list[TableMetadata] = field(default_factory=list)
    join_steps: list[JoinStep] = field(default_factory=list)
    rules: list[SQLRule] = field(default_factory=list)
    examples: list[ExampleQuery] = field(default_factory=list)
    base_table: str | None = None
    unreachable_tables: list[str] = field(default_factory=list)

    @property
    def table_names(self) -> list[str]:
        """Names of every table included in the context."""
        return [table.name for table in self.tables]

    @property
    def is_empty(self) -> bool:
        """True when retrieval found no usable schema for the question."""
        return not self.tables

    def render(self) -> str:
        """Render the context as the schema block injected into the prompt."""
        sections: list[str] = []

        if self.tables:
            table_blocks = [build_table_document(table).text for table in self.tables]
            sections.append("## Available tables\n\n" + "\n\n".join(table_blocks))

        if self.join_steps:
            join_lines = [
                f"- {step.source_table}.{step.source_column} -> "
                f"{step.target_table}.{step.target_column} "
                f"(use {step.join_type} JOIN)"
                for step in self.join_steps
            ]
            sections.append(
                "## Verified join paths\n\nJoin the tables above only along these "
                "paths:\n" + "\n".join(join_lines)
            )

        if self.rules:
            rule_lines = [f"- [{rule.id}] {rule.rule}" for rule in self.rules]
            sections.append("## SQL generation rules\n\n" + "\n".join(rule_lines))

        # Restated from the table definitions: omitting one silently inflates counts.
        required_filters = [
            f"- When {table.name} is joined, apply: {predicate}"
            for table in self.tables
            for predicate in table.default_filters
        ]
        if required_filters:
            sections.append(
                "## Filters that must always be applied\n\n"
                + "\n".join(required_filters)
                + "\n\nOmit one only if the question explicitly asks for the "
                "rows it excludes."
            )

        glossary_documents = [
            scored
            for scored in self.documents
            if scored.kind is DocumentKind.GLOSSARY
        ]
        if glossary_documents:
            sections.append(
                "## Business definitions\n\n"
                + "\n\n".join(scored.document.text for scored in glossary_documents)
            )

        if self.examples:
            example_blocks = [
                f"Question: {example.question}\nSQL:\n{example.sql.strip()}"
                for example in self.examples
            ]
            sections.append(
                "## Similar answered questions\n\n" + "\n\n".join(example_blocks)
            )

        return "\n\n".join(sections)


class SchemaContextBuilder:
    """Turns a question into grounded, join-complete Knowledge Base context."""

    def __init__(
        self,
        registry: KnowledgeBaseRegistry,
        retriever: HybridRetriever,
        settings: Settings,
    ) -> None:
        self._registry = registry
        self._retriever = retriever
        self._settings = settings

    def build(
        self, question: str, analysis: QuestionAnalysis | None = None
    ) -> RetrievedContext:
        """Retrieve and assemble everything needed to answer ``question``.

        ``analysis`` is optional; supplying it improves base-table selection.
        """
        scored_documents = self._retriever.retrieve(
            question, top_k=self._settings.retrieval_top_k
        )
        scored_documents = [
            scored
            for scored in scored_documents
            if scored.score >= self._settings.retrieval_min_score
        ]

        candidate_tables = self._collect_candidate_tables(scored_documents)
        if not candidate_tables:
            logger.warning("Retrieval found no candidate tables for: %s", question)
            return RetrievedContext(question=question, documents=scored_documents)

        base_table = self._select_base_table(candidate_tables, analysis)
        selected, join_steps, unreachable = self._expand_with_join_paths(
            base_table, candidate_tables
        )

        tables = [
            table
            for table in (self._registry.get_table(name) for name in selected)
            if table is not None
        ]
        rules = self._registry.rules_for_tables(set(selected))
        examples = self._select_examples(scored_documents)

        logger.debug(
            "Context for %r: base=%s tables=%s joins=%d examples=%d",
            question,
            base_table,
            selected,
            len(join_steps),
            len(examples),
        )

        return RetrievedContext(
            question=question,
            documents=scored_documents,
            tables=tables,
            join_steps=join_steps,
            rules=rules,
            examples=examples,
            base_table=base_table,
            unreachable_tables=unreachable,
        )

    def _collect_candidate_tables(
        self, scored_documents: list[ScoredDocument]
    ) -> list[str]:
        """Rank candidate tables by the evidence gathered across all documents."""
        scores: dict[str, float] = {}

        for scored in scored_documents:
            # A table's own document is direct evidence; a mention elsewhere is weaker.
            weight = 1.0 if scored.kind is DocumentKind.TABLE else 0.4
            for table_name in scored.document.tables:
                if self._registry.has_table(table_name):
                    scores[table_name] = scores.get(table_name, 0.0) + (
                        scored.score * weight
                    )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [name for name, _ in ranked[:MAX_TABLES_IN_CONTEXT]]

    def _select_base_table(
        self, candidate_tables: list[str], analysis: QuestionAnalysis | None = None
    ) -> str:
        """Pick the table to anchor the join plan on.

        Prefers a fact table, but only one the question actually refers to: retrieval
        returns topically near tables, and anchoring on an unrelated fact answers a
        different question.
        """
        mentioned = list(analysis.mentioned_tables) if analysis else []
        subjects = analysis.subject_tables() if analysis else set()

        # Tables named in a metric's expression are subjects too, even when unmentioned.
        if analysis is not None and analysis.metric is not None:
            subjects.update(
                self._registry.tables_in_expression(analysis.metric.expression)
            )

        for name in candidate_tables:
            table = self._registry.get_table(name)
            if table is None or _FACT_TAG not in table.tags:
                continue
            if analysis is None or name in subjects:
                return name

        for name in mentioned:
            if name in candidate_tables:
                return name

        return candidate_tables[0]

    def _expand_with_join_paths(
        self, base_table: str, candidate_tables: list[str]
    ) -> tuple[list[str], list[JoinStep], list[str]]:
        """Add whatever intermediate tables the joins require.

        Returns:
            A tuple of (ordered table names, join steps, unreachable table names).
        """
        targets = [name for name in candidate_tables if name != base_table]

        if not self._settings.retrieval_expand_joins:
            return [base_table, *targets], [], []

        join_steps, unreachable = self._registry.build_join_plan(base_table, targets)

        ordered: list[str] = [base_table]
        for step in join_steps:
            if step.target_table not in ordered:
                ordered.append(step.target_table)

        # Show unjoined tables anyway so the model explains rather than invents a join.
        for name in targets:
            if name not in ordered and name not in unreachable:
                ordered.append(name)

        return ordered, join_steps, unreachable

    def _select_examples(
        self, scored_documents: list[ScoredDocument]
    ) -> list[ExampleQuery]:
        """Resolve retrieved example documents back to their source records."""
        examples_by_id = {example.id: example for example in self._registry.examples}

        selected: list[ExampleQuery] = []
        for scored in scored_documents:
            if scored.kind is not DocumentKind.EXAMPLE:
                continue
            example = examples_by_id.get(scored.document.metadata.get("example_id", ""))
            if example is not None:
                selected.append(example)
            if len(selected) >= MAX_EXAMPLES_IN_CONTEXT:
                break

        return selected
