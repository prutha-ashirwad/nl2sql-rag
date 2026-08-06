"""Rule-based query planner driven entirely by the Knowledge Base."""

from __future__ import annotations

import re

from nl2sql.analysis.question_analyzer import (
    GroupingDimension,
    QueryIntent,
    QuestionAnalysis,
)
from nl2sql.dialects import SQLDialect
from nl2sql.exceptions import GenerationError
from nl2sql.generation.deterministic.query_plan import (
    PlannedJoin,
    QueryPlan,
    SelectExpression,
)
from nl2sql.knowledge_base.models import ColumnMetadata, ColumnRole, TableMetadata
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.logging_config import get_logger
from nl2sql.retrieval.context_builder import RetrievedContext

logger = get_logger(__name__)

# How many descriptive columns to project when listing raw rows.
MAX_DETAIL_COLUMNS = 5

# Column roles worth showing when listing individual rows, in priority order.
_DETAIL_ROLE_PRIORITY = (
    ColumnRole.TIMESTAMP,
    ColumnRole.DIMENSION,
    ColumnRole.MEASURE,
)

# Warehouse load metadata, never what the question was about.
_BOOKKEEPING_COLUMNS = frozenset({"created_at", "updated_at"})


class DeterministicPlanner:
    """Builds a query plan from analysed intent and retrieved schema context."""

    def __init__(
        self, registry: KnowledgeBaseRegistry, dialect: SQLDialect
    ) -> None:
        self._registry = registry
        self._dialect = dialect

    def plan(
        self, context: RetrievedContext, analysis: QuestionAnalysis
    ) -> QueryPlan:
        """Build the query plan that answers ``analysis``."""
        base_table_name = context.base_table
        if base_table_name is None:
            raise GenerationError(
                "No table could be identified for this question. The Knowledge Base "
                "may not cover the entities it refers to."
            )

        base_table = self._registry.get_table(base_table_name)
        if base_table is None:  # pragma: no cover - guarded by KB validation
            raise GenerationError(f"Unknown base table: {base_table_name}")

        groupings = self._resolve_groupings(context, analysis, base_table.name)
        required_tables = self._collect_required_tables(
            context, analysis, groupings, base_table.name
        )

        # Listing rows must not duplicate them; aggregates may fan out and compensate
        # by counting distinct base rows.
        preserve_grain = analysis.intent is QueryIntent.LIST
        joins, aliases = self._build_joins(
            base_table.name, required_tables, preserve_grain=preserve_grain
        )
        plan = QueryPlan(
            base_table=base_table.name,
            base_alias=aliases[base_table.name],
            joins=joins,
        )

        plan.where = self._build_predicates(analysis, base_table, aliases)

        if analysis.intent in (QueryIntent.AGGREGATE, QueryIntent.RANKING):
            self._apply_aggregation(plan, analysis, groupings, aliases, base_table)
        else:
            self._apply_detail_projection(plan, base_table, aliases)

        plan.limit = self._resolve_limit(plan, analysis)

        logger.debug(
            "Planned query: base=%s joins=%d predicates=%d grouped=%s",
            plan.base_table,
            len(plan.joins),
            len(plan.where),
            plan.has_aggregate,
        )
        return plan

    # -- Table selection ------------------------------------------------------

    def _resolve_groupings(
        self,
        context: RetrievedContext,
        analysis: QuestionAnalysis,
        base_table: str,
    ) -> list[GroupingDimension]:
        """Determine what the results should be broken down by.

        Failing an explicit "by <dimension>", a ranking question implies a grouping on
        the mentioned table's label column.
        """
        if analysis.groupings:
            return analysis.groupings

        if analysis.intent is not QueryIntent.RANKING:
            return []

        for table_name in analysis.mentioned_tables:
            if table_name == base_table:
                continue
            label_column = self._label_column(table_name)
            if label_column is not None:
                return [
                    GroupingDimension(
                        table=table_name,
                        column=label_column,
                        phrase=table_name,
                    )
                ]

        return []

    def _collect_required_tables(
        self,
        context: RetrievedContext,
        analysis: QuestionAnalysis,
        groupings: list[GroupingDimension],
        base_table: str,
    ) -> list[str]:
        """List the non-base tables the query must reach, in priority order.

        Retrieved context tables are added only when listing rows; joining them into an
        aggregate would change the grain of the result.
        """
        required: list[str] = []

        def add(table_name: str) -> None:
            if table_name != base_table and table_name not in required:
                required.append(table_name)

        for grouping in groupings:
            add(grouping.table)
            # Labels repeat across parents, so the parent keeps groups distinguishable.
            parent = self._identifying_parent(grouping.table)
            if parent is not None:
                add(parent)

        for value_filter in analysis.value_filters:
            add(value_filter.table)

        if analysis.metric is not None:
            for table_name in self._registry.tables_in_expression(
                analysis.metric.expression
            ):
                add(table_name)

        if analysis.intent is QueryIntent.LIST:
            for table in context.tables:
                add(table.name)

        return required

    def _build_joins(
        self,
        base_table: str,
        required_tables: list[str],
        *,
        preserve_grain: bool,
    ) -> tuple[list[PlannedJoin], dict[str, str]]:
        """Resolve join paths and bind every reached table to an alias."""
        steps, unreachable = self._registry.build_join_plan(
            base_table, required_tables, preserve_grain=preserve_grain
        )

        if unreachable:
            logger.info(
                "Skipping table(s) with no declared join path from %s: %s",
                base_table,
                ", ".join(unreachable),
            )

        aliases: dict[str, str] = {base_table: self._registry.alias_for(base_table)}
        joins: list[PlannedJoin] = []

        for step in steps:
            target_alias = self._registry.alias_for(step.target_table)
            aliases[step.target_table] = target_alias
            joins.append(
                PlannedJoin(
                    step=step,
                    source_alias=aliases[step.source_table],
                    target_alias=target_alias,
                )
            )

        return joins, aliases

    # -- Clause construction --------------------------------------------------

    def _build_predicates(
        self,
        analysis: QuestionAnalysis,
        base_table: TableMetadata,
        aliases: dict[str, str],
    ) -> list[str]:
        """Build the WHERE clause from resolved filters and the time window."""
        predicates: list[str] = []

        # Grouped by column so two values become an IN list; separate equalities would
        # AND together mutually exclusive conditions.
        values_by_column: dict[tuple[str, str], list[str]] = {}
        for value_filter in analysis.value_filters:
            key = (value_filter.table, value_filter.column)
            values = values_by_column.setdefault(key, [])
            if value_filter.value not in values:
                values.append(value_filter.value)

        for (table_name, column_name), values in values_by_column.items():
            alias = aliases.get(table_name)
            if alias is None:
                logger.info(
                    "Dropping filter on %s.%s: table is not reachable from %s",
                    table_name,
                    column_name,
                    base_table.name,
                )
                continue

            if len(values) == 1:
                predicates.append(f"{alias}.{column_name} = '{values[0]}'")
            else:
                rendered = ", ".join(f"'{value}'" for value in sorted(values))
                predicates.append(f"{alias}.{column_name} IN ({rendered})")

        predicates.extend(
            self._default_filters(aliases, set(values_by_column))
        )

        if analysis.time_window is not None:
            timestamp_column = self._registry.primary_timestamp_column(base_table.name)
            if timestamp_column is not None:
                qualified = f"{aliases[base_table.name]}.{timestamp_column.name}"
                predicates.extend(
                    self._dialect.time_window_predicates(
                        qualified, analysis.time_window
                    )
                )
            else:
                logger.info(
                    "Ignoring time window: %s declares no timestamp column",
                    base_table.name,
                )

        return predicates

    def _default_filters(
        self,
        aliases: dict[str, str],
        explicit_columns: set[tuple[str, str]],
    ) -> list[str]:
        """Apply each joined table's declared default filters.

        A default filter is skipped when the question already filters that same column,
        so "show decommissioned devices" is still answerable.
        """
        predicates: list[str] = []

        for table_name, alias in aliases.items():
            table = self._registry.get_table(table_name)
            if table is None:
                continue

            for expression in table.default_filters:
                column = self._filtered_column(table, expression)
                if column is not None and (table_name, column) in explicit_columns:
                    logger.debug(
                        "Skipping the default filter on %s.%s: the question "
                        "filters that column itself",
                        table_name,
                        column,
                    )
                    continue

                predicates.append(
                    re.sub(rf"\b{re.escape(table_name)}\.", f"{alias}.", expression)
                )

        return predicates

    @staticmethod
    def _filtered_column(table: TableMetadata, expression: str) -> str | None:
        """Return the column a default-filter expression constrains, if identifiable."""
        for column in table.columns:
            pattern = rf"\b{re.escape(table.name)}\.{re.escape(column.name)}\b"
            if re.search(pattern, expression):
                return column.name
        return None

    def _apply_aggregation(
        self,
        plan: QueryPlan,
        analysis: QuestionAnalysis,
        groupings: list[GroupingDimension],
        aliases: dict[str, str],
        base_table: TableMetadata,
    ) -> None:
        """Populate the SELECT and ORDER BY clauses for a grouped query."""
        for grouping in groupings:
            alias = aliases.get(grouping.table)
            if alias is None:
                continue

            plan.select.append(
                SelectExpression(expression=f"{alias}.{grouping.column}")
            )

            # Every extra column also lands in GROUP BY and so changes the grain; that
            # is only wanted to disambiguate a label, never a plain attribute.
            if grouping.column != self._label_column(grouping.table):
                continue

            context_label = self._parent_label_expression(
                grouping.table, aliases, exclude=grouping.column
            )
            if context_label is not None:
                plan.select.append(SelectExpression(expression=context_label))

        if analysis.metric is not None:
            metric_alias = analysis.metric.alias
            expression = self._bind_metric_expression(
                analysis.metric.expression, aliases
            )
        else:
            metric_alias = self._metric_alias(analysis, base_table)
            expression = self._count_expression(plan, base_table)

        plan.select.append(
            SelectExpression(
                expression=expression, alias=metric_alias, is_aggregate=True
            )
        )

        direction = "DESC" if analysis.descending else "ASC"
        plan.order_by = [f"{metric_alias} {direction}"]

    @staticmethod
    def _bind_metric_expression(expression: str, aliases: dict[str, str]) -> str:
        """Rewrite a declared metric expression to use this query's table aliases."""
        bound = expression
        # Longest table name first, so `observation_metrics.` is not partly matched
        # by the shorter `observations.` prefix.
        for table_name in sorted(aliases, key=len, reverse=True):
            bound = re.sub(
                rf"\b{re.escape(table_name)}\.", f"{aliases[table_name]}.", bound
            )
        return bound

    def _count_expression(self, plan: QueryPlan, base_table: TableMetadata) -> str:
        """Choose a counting expression that survives a fan-out join.

        ``COUNT(*)`` over-counts once a join multiplies base rows, so distinct base
        keys are counted instead.
        """
        fans_out = any(not join.step.preserves_grain for join in plan.joins)
        if not fans_out or not base_table.primary_key:
            return "COUNT(*)"

        return f"COUNT(DISTINCT {plan.base_alias}.{base_table.primary_key[0]})"

    def _apply_detail_projection(
        self,
        plan: QueryPlan,
        base_table: TableMetadata,
        aliases: dict[str, str],
    ) -> None:
        """Populate SELECT and ORDER BY for a query that lists individual rows."""
        base_alias = aliases[base_table.name]

        if base_table.primary_key:
            plan.select.append(
                SelectExpression(
                    expression=f"{base_alias}.{base_table.primary_key[0]}"
                )
            )

        for column in self._detail_columns(base_table):
            expression = f"{base_alias}.{column.name}"
            if expression not in {item.expression for item in plan.select}:
                plan.select.append(SelectExpression(expression=expression))

        # A joined dimension is only worth projecting through its label column.
        for table_name, alias in aliases.items():
            if table_name == base_table.name:
                continue
            label_column = self._label_column(table_name)
            if label_column is not None:
                plan.select.append(
                    SelectExpression(expression=f"{alias}.{label_column}")
                )

        timestamp_column = self._registry.primary_timestamp_column(base_table.name)
        if timestamp_column is not None:
            plan.order_by = [f"{base_alias}.{timestamp_column.name} DESC"]

    @staticmethod
    def _resolve_limit(plan: QueryPlan, analysis: QuestionAnalysis) -> int | None:
        """Return the row limit the question asked for, or none.

        A query is bounded only when the question named a number. Capping the result
        set is the executor's job, which stops fetching at ``MAX_RESULT_ROWS`` and
        reports the truncation — so an unbounded query is safe, and the answer stays
        the complete one the question asked for.
        """
        del plan  # The plan shape no longer changes the limit.
        return analysis.row_limit

    # -- Knowledge Base lookups -----------------------------------------------

    def _label_column(self, table_name: str) -> str | None:
        """Return the column that best identifies a row of ``table_name``."""
        return self._registry.label_column(table_name)

    def _identifying_parent(self, table_name: str) -> str | None:
        """Return the parent table whose label disambiguates rows of ``table_name``.

        The first declared foreign key pointing at a table with its own label column.
        """
        table = self._registry.get_table(table_name)
        if table is None:
            return None

        for foreign_key in table.foreign_keys:
            if self._label_column(foreign_key.references_table) is not None:
                return foreign_key.references_table

        return None

    def _parent_label_expression(
        self, table_name: str, aliases: dict[str, str], *, exclude: str
    ) -> str | None:
        """Return the qualified parent label column, when the parent is joined."""
        parent = self._identifying_parent(table_name)
        if parent is None:
            return None

        parent_alias = aliases.get(parent)
        parent_label = self._label_column(parent)
        if parent_alias is None or parent_label is None or parent_label == exclude:
            return None

        return f"{parent_alias}.{parent_label}"

    @staticmethod
    def _detail_columns(table: TableMetadata) -> list[ColumnMetadata]:
        """Choose which of a table's own columns to project when listing rows."""
        selected: list[ColumnMetadata] = []

        for role in _DETAIL_ROLE_PRIORITY:
            for column in table.columns_with_role(role):
                # Surrogate keys are projected separately by the caller.
                if column.is_primary_key or column.name in _BOOKKEEPING_COLUMNS:
                    continue
                selected.append(column)
                if len(selected) >= MAX_DETAIL_COLUMNS:
                    return selected

        return selected

    def _metric_alias(
        self, analysis: QuestionAnalysis, base_table: TableMetadata
    ) -> str:
        """Name the aggregate after what is actually being counted."""
        for value_filter in analysis.value_filters:
            if value_filter.table == base_table.name:
                return f"{value_filter.value.lower()}_count"

        return f"{base_table.name.rstrip('s')}_count"
