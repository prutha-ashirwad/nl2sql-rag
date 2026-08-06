"""Semantic validation of generated SQL against the Knowledge Base.

Findings are returned rather than raised so the workflow can feed them back into a
repair attempt.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.logging_config import get_logger
from nl2sql.validation.models import (
    IssueCode,
    Severity,
    ValidationReport,
)

logger = get_logger(__name__)

_READ_ONLY_STATEMENTS = (exp.Select, exp.Union, exp.Except, exp.Intersect)

_WRITE_EXPRESSIONS = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Merge,
    exp.Grant,
)


class SQLValidator:
    """Validates generated SQL for syntax, safety and Knowledge Base conformance."""

    def __init__(
        self, registry: KnowledgeBaseRegistry, *, dialect: str = "sqlite"
    ) -> None:
        self._registry = registry
        self._dialect = dialect

    def validate(self, sql: str) -> ValidationReport:
        """Validate ``sql`` and return every finding."""
        report = ValidationReport(sql=sql)

        if not sql or not sql.strip():
            report.add(
                IssueCode.EMPTY_QUERY,
                Severity.ERROR,
                "No SQL query was produced.",
            )
            return report

        statements = self._parse(sql, report)
        if statements is None:
            return report

        statement = statements[0]

        self._check_read_only(statement, report)
        if not report.is_valid:
            # Every check below assumes a SELECT tree.
            return report

        alias_map = self._build_alias_map(statement, report)
        self._check_columns(statement, alias_map, report)
        self._check_joins(statement, alias_map, report)
        self._check_enum_literals(statement, alias_map, report)
        self._check_style(statement, report)

        report.referenced_tables = sorted(set(alias_map.values()))
        report.normalised_sql = statement.sql(dialect=self._dialect, pretty=True)

        logger.debug(
            "Validation finished: %d error(s), %d warning(s)",
            len(report.errors),
            len(report.warnings),
        )
        return report

    # -- Individual checks ----------------------------------------------------

    def _parse(self, sql: str, report: ValidationReport) -> list[exp.Expression] | None:
        """Parse ``sql``, recording a finding and returning ``None`` on failure."""
        try:
            statements = [
                statement
                for statement in sqlglot.parse(sql, dialect=self._dialect)
                if statement is not None
            ]
        # SqlglotError, not ParseError: TokenError is a sibling, not a subclass.
        except SqlglotError as exc:
            report.add(
                IssueCode.SYNTAX_ERROR,
                Severity.ERROR,
                f"The query could not be parsed: {exc}",
            )
            return None

        if not statements:
            report.add(
                IssueCode.EMPTY_QUERY,
                Severity.ERROR,
                "The query contained no statement.",
            )
            return None

        if len(statements) > 1:
            report.add(
                IssueCode.MULTIPLE_STATEMENTS,
                Severity.ERROR,
                f"Expected exactly one statement but found {len(statements)}.",
                hint="Return a single SELECT query.",
            )
            return None

        return statements

    def _check_read_only(
        self, statement: exp.Expression, report: ValidationReport
    ) -> None:
        """Reject anything that could modify data or schema."""
        for write_type in _WRITE_EXPRESSIONS:
            if isinstance(statement, write_type) or statement.find(write_type):
                report.add(
                    IssueCode.NOT_READ_ONLY,
                    Severity.ERROR,
                    f"Only read-only queries are permitted; found a "
                    f"{write_type.__name__.upper()} statement.",
                    hint="Rewrite the request as a SELECT query.",
                )
                return

        root = statement.this if isinstance(statement, exp.Subquery) else statement
        if not isinstance(root, (*_READ_ONLY_STATEMENTS, exp.With)):
            report.add(
                IssueCode.NOT_READ_ONLY,
                Severity.ERROR,
                f"Expected a SELECT statement but found "
                f"{type(root).__name__.upper()}.",
            )

    def _build_alias_map(
        self, statement: exp.Expression, report: ValidationReport
    ) -> dict[str, str]:
        """Map every alias in the query onto the real table it refers to.

        CTE names are registered too, so a CTE reference is not read as a
        hallucinated table.
        """
        alias_map: dict[str, str] = {}
        cte_names = {
            cte.alias_or_name.lower()
            for cte in statement.find_all(exp.CTE)
            if cte.alias_or_name
        }

        for table in statement.find_all(exp.Table):
            table_name = (table.name or "").lower()
            if not table_name:
                continue

            if table_name in cte_names:
                alias_map[(table.alias or table_name).lower()] = table_name
                continue

            if not self._registry.has_table(table_name):
                report.add(
                    IssueCode.UNKNOWN_TABLE,
                    Severity.ERROR,
                    f"Table {table_name!r} is not declared in the Knowledge Base.",
                    hint=f"Known tables: {', '.join(self._registry.table_names[:12])}…",
                )
                continue

            alias_map[(table.alias or table_name).lower()] = table_name

        return alias_map

    def _check_columns(
        self,
        statement: exp.Expression,
        alias_map: dict[str, str],
        report: ValidationReport,
    ) -> None:
        """Verify that every referenced column exists on the table it belongs to."""
        real_tables = {
            alias: table
            for alias, table in alias_map.items()
            if self._registry.has_table(table)
        }
        known_aliases = set(self._collect_output_aliases(statement))
        referenced: list[str] = []

        for column in statement.find_all(exp.Column):
            column_name = (column.name or "").lower()
            if not column_name or column_name == "*":
                continue

            qualifier = (column.table or "").lower()

            if qualifier:
                table_name = alias_map.get(qualifier)
                if table_name is None:
                    report.add(
                        IssueCode.UNKNOWN_TABLE,
                        Severity.ERROR,
                        f"Alias {qualifier!r} in {qualifier}.{column_name} is not "
                        f"bound to any table in the FROM clause.",
                    )
                    continue
                if not self._registry.has_table(table_name):
                    # A CTE column; its definition was validated separately.
                    continue
                if self._registry.get_column(table_name, column_name) is None:
                    report.add(
                        IssueCode.UNKNOWN_COLUMN,
                        Severity.ERROR,
                        f"Column {column_name!r} does not exist on table "
                        f"{table_name!r}.",
                        hint=self._suggest_columns(table_name),
                    )
                    continue
                referenced.append(f"{table_name}.{column_name}")
                continue

            # Unqualified: it may be a SELECT-list alias reused in ORDER BY/HAVING.
            if column_name in known_aliases:
                continue

            owners = [
                table
                for table in set(real_tables.values())
                if self._registry.get_column(table, column_name) is not None
            ]

            if not owners:
                report.add(
                    IssueCode.UNKNOWN_COLUMN,
                    Severity.ERROR,
                    f"Column {column_name!r} was not found on any table in the query.",
                )
            elif len(owners) > 1:
                report.add(
                    IssueCode.AMBIGUOUS_COLUMN,
                    Severity.WARNING,
                    f"Column {column_name!r} is ambiguous; it exists on "
                    f"{', '.join(sorted(owners))}.",
                    hint="Qualify every column with its table alias.",
                )
            else:
                referenced.append(f"{owners[0]}.{column_name}")

        report.referenced_columns = sorted(set(referenced))

    def _check_joins(
        self,
        statement: exp.Expression,
        alias_map: dict[str, str],
        report: ValidationReport,
    ) -> None:
        """Verify that every join follows a path declared in the Knowledge Base."""
        for join in statement.find_all(exp.Join):
            condition = join.args.get("on")
            if condition is None:
                if join.args.get("using") is not None:
                    continue
                report.add(
                    IssueCode.MISSING_JOIN_CONDITION,
                    Severity.ERROR,
                    "A join has no ON condition, which produces a cross product.",
                    hint="Join along a declared relationship.",
                )
                continue

            for equality in condition.find_all(exp.EQ):
                left, right = equality.this, equality.expression
                if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                    continue

                left_table = alias_map.get((left.table or "").lower())
                right_table = alias_map.get((right.table or "").lower())
                if left_table is None or right_table is None:
                    continue
                if not (
                    self._registry.has_table(left_table)
                    and self._registry.has_table(right_table)
                ):
                    # At least one side is a CTE, which has no declared relationship.
                    continue

                if not self._is_declared_join(
                    left_table, left.name, right_table, right.name
                ):
                    report.add(
                        IssueCode.UNDECLARED_JOIN,
                        Severity.ERROR,
                        f"The join {left_table}.{left.name} = "
                        f"{right_table}.{right.name} is not a relationship declared "
                        f"in the Knowledge Base.",
                        hint="Use one of the verified join paths from the context.",
                    )

    def _check_enum_literals(
        self,
        statement: exp.Expression,
        alias_map: dict[str, str],
        report: ValidationReport,
    ) -> None:
        """Verify that literals compared to enumerated columns are valid values."""
        for comparison in statement.find_all(exp.EQ, exp.NEQ, exp.In):
            column = comparison.this
            if not isinstance(column, exp.Column):
                continue

            table_name = alias_map.get((column.table or "").lower())
            if table_name is None or not self._registry.has_table(table_name):
                continue

            metadata = self._registry.get_column(table_name, column.name)
            if metadata is None or not metadata.is_enumerated:
                continue

            allowed = set(metadata.allowed_values)
            for literal in self._literals_of(comparison):
                if literal not in allowed:
                    report.add(
                        IssueCode.INVALID_ENUM_VALUE,
                        Severity.ERROR,
                        f"{literal!r} is not a valid value for "
                        f"{table_name}.{column.name}.",
                        hint=f"Allowed values: {', '.join(sorted(allowed))}.",
                    )

    @staticmethod
    def _check_style(statement: exp.Expression, report: ValidationReport) -> None:
        """Reject projections that return meaningless columns."""
        select = statement.find(exp.Select)
        if select is None:
            return

        if any(isinstance(item, exp.Star) for item in select.expressions):
            report.add(
                IssueCode.SELECT_STAR,
                Severity.ERROR,
                "The query selects all columns with '*'.",
                hint=(
                    "Select named columns, and prefer descriptive ones such as "
                    "device_name or interface_name over raw id columns."
                ),
            )

    # -- Helpers --------------------------------------------------------------

    def _is_declared_join(
        self, left_table: str, left_column: str, right_table: str, right_column: str
    ) -> bool:
        """True when the column pair matches a declared relationship or foreign key."""
        pair = {
            (left_table, left_column.lower()),
            (right_table, right_column.lower()),
        }

        for relationship in self._registry.relationships:
            declared = {
                (relationship.from_table, relationship.from_column),
                (relationship.to_table, relationship.to_column),
            }
            if declared == pair:
                return True

        directions = ((left_table, right_table), (right_table, left_table))
        for table_name, other_table in directions:
            table = self._registry.get_table(table_name)
            if table is None:
                continue
            for foreign_key in table.foreign_keys:
                declared = {
                    (table_name, foreign_key.column),
                    (foreign_key.references_table, foreign_key.references_column),
                }
                if declared == pair and foreign_key.references_table == other_table:
                    return True

        return False

    @staticmethod
    def _literals_of(comparison: exp.Expression) -> list[str]:
        """Extract the string literals a column is being compared against.

        Binary comparisons hold their right-hand side on ``expression``, while an
        ``IN`` list holds its values on ``expressions``.
        """
        candidates: list[exp.Expression] = []

        right_hand_side = comparison.args.get("expression")
        if right_hand_side is not None:
            candidates.append(right_hand_side)
        candidates.extend(comparison.args.get("expressions") or [])

        return [
            literal.this
            for candidate in candidates
            for literal in candidate.find_all(exp.Literal)
            if literal.is_string
        ]

    @staticmethod
    def _collect_output_aliases(statement: exp.Expression) -> set[str]:
        """Collect SELECT-list aliases so ORDER BY can reference them."""
        aliases: set[str] = set()
        for select in statement.find_all(exp.Select):
            for projection in select.expressions:
                if isinstance(projection, exp.Alias) and projection.alias:
                    aliases.add(projection.alias.lower())
        return aliases

    def _suggest_columns(self, table_name: str) -> str:
        """Render a short list of the table's real columns as a repair hint."""
        table = self._registry.get_table(table_name)
        if table is None:
            return ""
        names = [column.name for column in table.columns][:10]
        return f"Columns on {table_name}: {', '.join(names)}."
