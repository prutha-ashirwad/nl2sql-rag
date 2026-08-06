"""Tests for generating sample rows from Knowledge Base metadata.

The generator exists so a newly added table can be queried end to end without
leaving the interface. What matters is that the rows it invents are *usable*: a
filter on a declared value must match something, a foreign key must point at a row
that exists, and a real table must never be silently overwritten.
"""

from __future__ import annotations

from sqlalchemy import Engine, text

from nl2sql.database.sample_data import (
    count_rows,
    empty_tables,
    generate_rows,
    populate_sample_data,
)
from nl2sql.knowledge_base.models import ColumnRole
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry

TARGET = "observation_metrics"


def parent_keys_for(
    engine: Engine, registry: KnowledgeBaseRegistry, table_name: str
) -> dict[str, list[object]]:
    table = registry.get_table(table_name)
    assert table is not None
    keys: dict[str, list[object]] = {}
    with engine.connect() as connection:
        for foreign_key in table.foreign_keys:
            rows = connection.execute(
                text(
                    f"SELECT DISTINCT {foreign_key.references_column} "
                    f"FROM {foreign_key.references_table}"
                )
            ).fetchall()
            keys[foreign_key.column] = [row[0] for row in rows]
    return keys


class TestGeneratedValues:
    def test_one_row_per_requested_count(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        table = registry.get_table("observations")
        assert table is not None
        rows = generate_rows(table, count=7, parent_keys={})

        assert len(rows) == 7
        assert all(set(row) == {c.name for c in table.columns} for row in rows)

    def test_the_primary_key_is_unique(self, registry: KnowledgeBaseRegistry) -> None:
        table = registry.get_table("observations")
        assert table is not None
        key = table.primary_key[0]
        rows = generate_rows(table, count=40, parent_keys={})

        assert len({row[key] for row in rows}) == 40

    def test_declared_values_are_the_only_ones_used(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        # This is the property that makes a generated table answerable: a question
        # filtering on a declared value has to match something.
        table = registry.get_table("observations")
        assert table is not None
        enums = [c for c in table.columns if c.allowed_values]
        assert enums, "expected observations to declare at least one value set"

        rows = generate_rows(table, count=60, parent_keys={})
        for column in enums:
            produced = {row[column.name] for row in rows}
            assert produced <= set(column.allowed_values)

    def test_foreign_keys_only_reference_supplied_parents(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        table = registry.get_table("observations")
        assert table is not None
        rows = generate_rows(table, count=30, parent_keys={"device_id": [7, 8, 9]})

        assert {row["device_id"] for row in rows} <= {7, 8, 9}

    def test_timestamps_are_recent_and_ordered_text(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        table = registry.get_table("observations")
        assert table is not None
        stamps = [c for c in table.columns if c.role == ColumnRole.TIMESTAMP]
        assert stamps

        rows = generate_rows(table, count=20, parent_keys={})
        for row in rows:
            value = str(row[stamps[0].name])
            # ISO text sorts chronologically, which is what the SQL comparisons rely on.
            assert value.startswith("20") and "T" in value

    def test_the_same_seed_produces_the_same_rows(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        table = registry.get_table("observations")
        assert table is not None
        first = generate_rows(table, count=5, parent_keys={}, seed=3)
        second = generate_rows(table, count=5, parent_keys={}, seed=3)

        assert first == second


class TestPopulate:
    def test_an_unknown_table_is_refused(
        self, seeded_engine: Engine, registry: KnowledgeBaseRegistry
    ) -> None:
        result = populate_sample_data(seeded_engine, registry, "not_a_table")

        assert not result.ok
        assert "not declared" in result.message

    def test_a_populated_table_is_not_overwritten_by_default(
        self, seeded_engine: Engine, registry: KnowledgeBaseRegistry
    ) -> None:
        before = count_rows(seeded_engine, "observations")
        result = populate_sample_data(seeded_engine, registry, "observations", rows=5)

        assert not result.ok
        assert "already holds" in result.message
        assert count_rows(seeded_engine, "observations") == before

    def test_rows_land_and_satisfy_their_foreign_keys(
        self, seeded_engine: Engine, registry: KnowledgeBaseRegistry
    ) -> None:
        result = populate_sample_data(
            seeded_engine, registry, TARGET, rows=25, replace=True
        )
        assert result.ok, result.message
        assert result.rows_written == 25

        table = registry.get_table(TARGET)
        assert table is not None
        with seeded_engine.connect() as connection:
            for foreign_key in table.foreign_keys:
                orphans = connection.execute(
                    text(
                        f"SELECT COUNT(*) FROM {TARGET} child "
                        f"LEFT JOIN {foreign_key.references_table} parent "
                        f"  ON child.{foreign_key.column} "
                        f"   = parent.{foreign_key.references_column} "
                        f"WHERE parent.{foreign_key.references_column} IS NULL"
                    )
                ).scalar_one()
                assert orphans == 0, f"{foreign_key.column} produced orphan rows"

    def test_replace_swaps_the_rows_rather_than_appending(
        self, seeded_engine: Engine, registry: KnowledgeBaseRegistry
    ) -> None:
        populate_sample_data(seeded_engine, registry, TARGET, rows=12, replace=True)
        populate_sample_data(seeded_engine, registry, TARGET, rows=9, replace=True)

        assert count_rows(seeded_engine, TARGET) == 9

    def test_empty_tables_are_discoverable(
        self, seeded_engine: Engine, registry: KnowledgeBaseRegistry
    ) -> None:
        with seeded_engine.begin() as connection:
            connection.execute(text(f"DELETE FROM {TARGET}"))

        assert TARGET in empty_tables(seeded_engine, registry)
