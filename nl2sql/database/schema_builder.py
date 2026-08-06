"""Deriving physical DDL from Knowledge Base metadata."""

from __future__ import annotations

from nl2sql.dialects import SQLITE, SQLDialect
from nl2sql.exceptions import KnowledgeBaseError
from nl2sql.knowledge_base.models import TableMetadata
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.logging_config import get_logger

logger = get_logger(__name__)


def build_create_table(table: TableMetadata, dialect: SQLDialect = SQLITE) -> str:
    """Render the ``CREATE TABLE`` statement for one table."""
    lines: list[str] = []

    for column in table.columns:
        parts = [f"    {column.name}", dialect.storage_type(column.data_type)]
        if not column.nullable:
            parts.append("NOT NULL")
        lines.append(" ".join(parts))

    if table.primary_key:
        lines.append(f"    PRIMARY KEY ({', '.join(table.primary_key)})")

    for foreign_key in table.foreign_keys:
        lines.append(
            f"    FOREIGN KEY ({foreign_key.column}) REFERENCES "
            f"{foreign_key.references_table}({foreign_key.references_column})"
        )

    body = ",\n".join(lines)
    return f"CREATE TABLE IF NOT EXISTS {table.name} (\n{body}\n)"


def build_indexes(table: TableMetadata) -> list[str]:
    """Render an index for every foreign key and every non-``created_at`` timestamp."""
    statements: list[str] = []

    for foreign_key in table.foreign_keys:
        statements.append(
            f"CREATE INDEX IF NOT EXISTS idx_{table.name}_{foreign_key.column} "
            f"ON {table.name}({foreign_key.column})"
        )

    for column in table.columns:
        if column.role.value == "timestamp" and column.name != "created_at":
            statements.append(
                f"CREATE INDEX IF NOT EXISTS idx_{table.name}_{column.name} "
                f"ON {table.name}({column.name})"
            )

    return statements


def order_tables_by_dependency(registry: KnowledgeBaseRegistry) -> list[TableMetadata]:
    """Return tables ordered so a table is always created after its parents.

    Raises:
        KnowledgeBaseError: if the foreign keys contain a cycle.
    """
    remaining = {table.name: table for table in registry.tables}
    ordered: list[TableMetadata] = []
    resolved: set[str] = set()

    while remaining:
        ready = [
            table
            for table in remaining.values()
            if all(
                foreign_key.references_table in resolved
                or foreign_key.references_table == table.name
                for foreign_key in table.foreign_keys
            )
        ]

        if not ready:
            raise KnowledgeBaseError(
                "Foreign keys form a cycle; cannot determine a creation order for: "
                f"{sorted(remaining)}"
            )

        for table in sorted(ready, key=lambda item: item.name):
            ordered.append(table)
            resolved.add(table.name)
            del remaining[table.name]

    return ordered


def build_schema_statements(
    registry: KnowledgeBaseRegistry, dialect: SQLDialect = SQLITE
) -> list[str]:
    """Render every DDL statement needed to materialise the Knowledge Base schema."""
    statements: list[str] = []

    for table in order_tables_by_dependency(registry):
        statements.append(build_create_table(table, dialect))
        statements.extend(build_indexes(table))

    logger.debug(
        "Built %d DDL statement(s) from the Knowledge Base for %s",
        len(statements),
        dialect.name,
    )
    return statements
