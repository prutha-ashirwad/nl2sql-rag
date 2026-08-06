"""Building and populating the demo analytics database."""

from __future__ import annotations

from sqlalchemy import Engine, text

from nl2sql.database.demo_data import build_demo_dataset
from nl2sql.database.schema_builder import (
    build_schema_statements,
    order_tables_by_dependency,
)
from nl2sql.dialects import SQLITE, SQLDialect
from nl2sql.knowledge_base.models import TableMetadata
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.logging_config import get_logger

logger = get_logger(__name__)

_INSERT_BATCH_SIZE = 500


def create_schema(
    engine: Engine, registry: KnowledgeBaseRegistry, dialect: SQLDialect = SQLITE
) -> None:
    """Create every table and index declared in the Knowledge Base."""
    statements = build_schema_statements(registry, dialect)

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

    logger.info("Schema created: %d statement(s) applied", len(statements))


def drop_schema(engine: Engine, registry: KnowledgeBaseRegistry) -> None:
    """Drop every declared table, children before parents."""
    ordered = order_tables_by_dependency(registry)

    with engine.begin() as connection:
        for table in reversed(ordered):
            connection.execute(text(f"DROP TABLE IF EXISTS {table.name}"))

    logger.info("Schema dropped: %d table(s) removed", len(ordered))


def _adapt_rows(
    table: TableMetadata, rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Cast demo values to the Python types the declared columns actually hold.

    The dataset is written the way SQLite stores things, where a boolean is just the
    integer 0 or 1. Postgres declares a real ``BOOLEAN`` and refuses the integer
    outright, so the cast belongs here — where the declared type is known — rather
    than in the dataset, which has no idea which engine it is being loaded into.
    """
    booleans = {
        column.name
        for column in table.columns
        if column.data_type.strip().upper() == "BOOLEAN"
    }
    if not booleans:
        return rows

    return [
        {
            name: bool(value) if name in booleans and value is not None else value
            for name, value in row.items()
        }
        for row in rows
    ]


def _insert_rows(
    engine: Engine, table_name: str, rows: list[dict[str, object]]
) -> None:
    """Insert ``rows`` into ``table_name`` using named parameter binding."""
    if not rows:
        return

    columns = list(rows[0].keys())
    placeholders = ", ".join(f":{column}" for column in columns)
    statement = text(
        f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    )

    with engine.begin() as connection:
        for start in range(0, len(rows), _INSERT_BATCH_SIZE):
            connection.execute(statement, rows[start : start + _INSERT_BATCH_SIZE])


def seed_database(
    engine: Engine,
    registry: KnowledgeBaseRegistry,
    *,
    recreate: bool = True,
    dialect: SQLDialect = SQLITE,
) -> dict[str, int]:
    """Build the demo database and populate it with the demo dataset.

    ``recreate`` drops existing tables first, so seeding is idempotent. ``dialect``
    must match ``engine``; a mismatch produces columns of the wrong type rather than
    an error.

    Returns:
        Row counts keyed by table name.
    """
    if recreate:
        drop_schema(engine, registry)

    create_schema(engine, registry, dialect)

    dataset = build_demo_dataset()

    counts: dict[str, int] = {}
    # Insert in dependency order so foreign key enforcement never rejects a row.
    for table in order_tables_by_dependency(registry):
        rows = _adapt_rows(table, dataset.tables.get(table.name, []))
        _insert_rows(engine, table.name, rows)
        counts[table.name] = len(rows)
        if rows:
            logger.debug("Inserted %d row(s) into %s", len(rows), table.name)

    logger.info(
        "Database seeded: %d row(s) across %d table(s)",
        sum(counts.values()),
        len([name for name, count in counts.items() if count]),
    )
    return counts
