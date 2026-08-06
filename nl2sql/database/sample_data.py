"""Generating sample rows for a table from its Knowledge Base metadata."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from nl2sql.knowledge_base.models import ColumnMetadata, ColumnRole, TableMetadata
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.logging_config import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 500

# Timestamps land inside this window, ending now.
_HISTORY_DAYS = 90

# Plausible magnitudes for a measure, chosen by its declared unit.
_UNIT_RANGES: dict[str, tuple[int, int]] = {
    "percent": (0, 100),
    "percentage": (0, 100),
    "ms": (1, 5_000),
    "milliseconds": (1, 5_000),
    "seconds": (1, 3_600),
    "days": (1, 365),
    "bytes": (1_024, 10_000_000),
    "count": (0, 500),
}
_DEFAULT_MEASURE_RANGE = (1, 1_000)


@dataclass(frozen=True, slots=True)
class SampleDataResult:
    """The outcome of one attempt to populate a table."""

    ok: bool
    message: str
    rows_written: int = 0


def _timestamp(rng: random.Random, now: datetime) -> str:
    """A moment inside the history window, as an ISO string."""
    moment = now - timedelta(
        seconds=rng.randint(0, _HISTORY_DAYS * 24 * 3_600),
    )
    return moment.replace(microsecond=0).isoformat(sep="T")


def _measure(rng: random.Random, column: ColumnMetadata) -> int:
    """A number sized to the column's declared unit."""
    low, high = _UNIT_RANGES.get(
        (column.unit or "").strip().lower(), _DEFAULT_MEASURE_RANGE
    )
    return rng.randint(low, high)


def _text_value(rng: random.Random, column: ColumnMetadata, index: int) -> str:
    """A readable stand-in for a text column with no declared value set."""
    if column.example_values:
        return str(rng.choice(list(column.example_values)))

    stem = column.name.removesuffix("_name").removesuffix("_code").replace("_", " ")
    if column.role == ColumnRole.DESCRIPTIVE:
        return f"Sample {stem} entry {index}."
    return f"{stem.title().replace(' ', '')}-{index:04d}"


def _value_for(
    column: ColumnMetadata,
    *,
    index: int,
    rng: random.Random,
    now: datetime,
    parent_keys: dict[str, list[object]],
) -> object:
    """Invent one value for ``column``, honouring whatever the metadata declares."""
    if column.name in parent_keys:
        # A foreign key must point at a row that exists.
        return rng.choice(parent_keys[column.name])

    if column.is_primary_key:
        return index

    if column.allowed_values:
        return rng.choice(list(column.allowed_values))

    if column.role == ColumnRole.TIMESTAMP:
        return _timestamp(rng, now)

    if column.role == ColumnRole.FLAG:
        return rng.choice([0, 1])

    if column.role == ColumnRole.MEASURE:
        return _measure(rng, column)

    declared = column.data_type.strip().upper()
    if declared.startswith(("INT", "BIGINT", "SMALLINT")):
        return rng.randint(1, 1_000)
    if declared.startswith("BOOL"):
        return rng.choice([0, 1])
    if declared.startswith(("TIMESTAMP", "DATETIME", "DATE")):
        return _timestamp(rng, now)

    return _text_value(rng, column, index)


def _parent_keys(
    engine: Engine, table: TableMetadata
) -> tuple[dict[str, list[object]], list[str]]:
    """Read the values each foreign key column is allowed to take.

    Returns:
        The usable keys by column name, and the names of any parents that are empty.
    """
    keys: dict[str, list[object]] = {}
    empty: list[str] = []

    with engine.connect() as connection:
        for foreign_key in table.foreign_keys:
            rows = connection.execute(
                text(
                    f"SELECT DISTINCT {foreign_key.references_column} "
                    f"FROM {foreign_key.references_table} "
                    f"WHERE {foreign_key.references_column} IS NOT NULL"
                )
            ).fetchall()
            values = [row[0] for row in rows]
            if values:
                keys[foreign_key.column] = values
            else:
                empty.append(foreign_key.references_table)

    return keys, empty


def count_rows(engine: Engine, table_name: str) -> int:
    """Return how many rows ``table_name`` currently holds."""
    with engine.connect() as connection:
        return int(
            connection.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar_one()
        )


def generate_rows(
    table: TableMetadata,
    *,
    count: int,
    parent_keys: dict[str, list[object]],
    seed: int = 0,
    now: datetime | None = None,
) -> list[dict[str, object]]:
    """Build ``count`` synthetic rows for ``table``.

    ``parent_keys`` maps each foreign key column to the values it may take. ``now``
    ends the timestamp window and defaults to the current UTC time.
    """
    rng = random.Random(seed)
    moment = now or datetime.now(UTC)

    return [
        {
            column.name: _value_for(
                column, index=index, rng=rng, now=moment, parent_keys=parent_keys
            )
            for column in table.columns
        }
        for index in range(1, count + 1)
    ]


def populate_sample_data(
    engine: Engine,
    registry: KnowledgeBaseRegistry,
    table_name: str,
    *,
    rows: int = 50,
    seed: int = 0,
    replace: bool = False,
) -> SampleDataResult:
    """Fill ``table_name`` with synthetic rows derived from its metadata.

    A table that already holds rows is left alone unless ``replace`` is set, which
    deletes them first.
    """
    table = registry.get_table(table_name)
    if table is None:
        return SampleDataResult(
            ok=False, message=f"{table_name} is not declared in the Knowledge Base."
        )

    try:
        existing = count_rows(engine, table_name)
    except SQLAlchemyError as exc:
        if "no such table" in str(exc).lower():
            return SampleDataResult(
                ok=False,
                message=(
                    f"{table_name} is not in the database yet — run "
                    f"'Create missing tables' first."
                ),
            )
        return SampleDataResult(
            ok=False, message=f"Could not read {table_name}: {exc}"
        )

    if existing and not replace:
        return SampleDataResult(
            ok=False,
            message=(
                f"{table_name} already holds {existing:,} row(s). "
                "Tick 'replace existing rows' to overwrite them."
            ),
        )

    keys, empty_parents = _parent_keys(engine, table)
    if empty_parents:
        return SampleDataResult(
            ok=False,
            message=(
                f"{table_name} points at {', '.join(sorted(set(empty_parents)))}, "
                "which has no rows to reference. Populate that first."
            ),
        )

    generated = generate_rows(table, count=rows, parent_keys=keys, seed=seed)
    columns = [column.name for column in table.columns]
    placeholders = ", ".join(f":{name}" for name in columns)
    statement = text(
        f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    )

    with engine.begin() as connection:
        if replace:
            connection.execute(text(f"DELETE FROM {table_name}"))
        for start in range(0, len(generated), _BATCH_SIZE):
            connection.execute(statement, generated[start : start + _BATCH_SIZE])

    logger.info("Sample data written: %d row(s) into %s", len(generated), table_name)
    return SampleDataResult(
        ok=True,
        message=f"Wrote {len(generated):,} sample row(s) into {table_name}.",
        rows_written=len(generated),
    )


def missing_tables(engine: Engine, registry: KnowledgeBaseRegistry) -> list[str]:
    """Every table the Knowledge Base declares that the database does not have yet."""
    existing = set(inspect(engine).get_table_names())
    return [table.name for table in registry.tables if table.name not in existing]


def empty_tables(engine: Engine, registry: KnowledgeBaseRegistry) -> list[str]:
    """Every declared table that exists in the database but holds no rows."""
    found: list[str] = []
    for table in registry.tables:
        try:
            if count_rows(engine, table.name) == 0:
                found.append(table.name)
        except SQLAlchemyError:
            # Declared but not built yet — nothing to populate.
            continue
    return found
