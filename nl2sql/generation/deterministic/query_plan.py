"""A structured, renderable representation of a SQL query."""

from __future__ import annotations

from dataclasses import dataclass, field

from nl2sql.knowledge_base.registry import JoinStep


@dataclass(frozen=True, slots=True)
class SelectExpression:
    """One entry in the SELECT list."""

    expression: str
    alias: str | None = None
    is_aggregate: bool = False

    def render(self) -> str:
        """Render the expression with its alias, if it has one."""
        return f"{self.expression} AS {self.alias}" if self.alias else self.expression


@dataclass(frozen=True, slots=True)
class PlannedJoin:
    """A join step with both sides already bound to their table aliases."""

    step: JoinStep
    source_alias: str
    target_alias: str

    def render(self) -> str:
        """Render the join as a SQL clause."""
        return self.step.to_sql(self.source_alias, self.target_alias)


@dataclass(slots=True)
class QueryPlan:
    """A complete, renderable SELECT statement."""

    base_table: str
    base_alias: str
    select: list[SelectExpression] = field(default_factory=list)
    joins: list[PlannedJoin] = field(default_factory=list)
    where: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    order_by: list[str] = field(default_factory=list)
    limit: int | None = None

    @property
    def has_aggregate(self) -> bool:
        """True when the SELECT list contains an aggregate expression."""
        return any(expression.is_aggregate for expression in self.select)

    def referenced_tables(self) -> list[str]:
        """Every table the plan reads from, base table first."""
        return [self.base_table, *(join.step.target_table for join in self.joins)]

    def to_sql(self) -> str:
        """Render the plan as formatted SQL."""
        lines: list[str] = ["SELECT"]

        select_lines = [f"    {expression.render()}" for expression in self.select]
        lines.append(",\n".join(select_lines))

        lines.append(f"FROM {self.base_table} {self.base_alias}")
        lines.extend(join.render() for join in self.joins)

        if self.where:
            lines.append("WHERE " + "\n  AND ".join(self.where))

        group_by = self._resolve_group_by()
        if group_by:
            lines.append("GROUP BY " + ", ".join(group_by))

        if self.order_by:
            lines.append("ORDER BY " + ", ".join(self.order_by))

        if self.limit is not None:
            lines.append(f"LIMIT {self.limit}")

        return "\n".join(lines)

    def _resolve_group_by(self) -> list[str]:
        """Return the GROUP BY list, completed from the SELECT list if needed."""
        if not self.has_aggregate:
            return []

        grouped = list(self.group_by)
        for expression in self.select:
            if expression.is_aggregate:
                continue
            if expression.expression not in grouped:
                grouped.append(expression.expression)

        return grouped
