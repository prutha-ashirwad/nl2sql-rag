"""Invariants that keep a newly added table a data-only change.

The system is built so that a table can be added by writing YAML, with no change to
the agent. That only holds while the surrounding pieces keep up with the Knowledge
Base: the join planner walks declared relationships, and the demo database is built
from the declared tables. A table added without them still loads, still appears in
the schema browser and still gets retrieved — it simply produces wrong or empty
answers, with nothing to signal why.

These tests turn that silent failure into a failing build.
"""

from __future__ import annotations

from nl2sql.database.demo_data import build_demo_dataset
from nl2sql.knowledge_base.models import KnowledgeBase


def _declared_edges(knowledge_base: KnowledgeBase) -> set[tuple[str, str, str, str]]:
    """Every relationship, recorded in both directions.

    The join graph is undirected, so a relationship declared from either end of a
    foreign key satisfies it.
    """
    edges: set[tuple[str, str, str, str]] = set()
    for relationship in knowledge_base.relationships:
        edges.add(
            (
                relationship.from_table,
                relationship.from_column,
                relationship.to_table,
                relationship.to_column,
            )
        )
        edges.add(
            (
                relationship.to_table,
                relationship.to_column,
                relationship.from_table,
                relationship.from_column,
            )
        )
    return edges


class TestEveryForeignKeyIsJoinable:
    """A physical link with no declared relationship is invisible to the planner."""

    def test_no_foreign_key_is_missing_its_relationship(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        declared = _declared_edges(knowledge_base)

        orphaned = [
            f"{table.name}.{key.column} -> "
            f"{key.references_table}.{key.references_column}"
            for table in knowledge_base.tables
            for key in table.foreign_keys
            if (table.name, key.column, key.references_table, key.references_column)
            not in declared
        ]

        # Without an edge the planner does not fail — it routes around the gap
        # through a longer path, quietly answering a different question.
        assert orphaned == [], (
            "These foreign keys have no relationship in relationships.yaml, so the "
            f"join planner cannot traverse them: {orphaned}"
        )

    def test_relationships_reference_real_columns(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        columns = {
            (table.name, column.name)
            for table in knowledge_base.tables
            for column in table.columns
        }

        dangling = [
            relationship.name
            for relationship in knowledge_base.relationships
            if (relationship.from_table, relationship.from_column) not in columns
            or (relationship.to_table, relationship.to_column) not in columns
        ]

        assert dangling == []


class TestEveryTableHasDemoData:
    """A declared table with no generator seeds empty and answers every question 0."""

    def test_no_declared_table_seeds_empty(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        seeded = set(build_demo_dataset().tables)

        missing = sorted({table.name for table in knowledge_base.tables} - seeded)

        assert missing == [], (
            "These tables are declared in the Knowledge Base but have no demo data "
            f"generator, so they seed to zero rows: {missing}"
        )

    def test_demo_data_declares_no_unknown_table(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        declared = {table.name for table in knowledge_base.tables}

        unknown = sorted(set(build_demo_dataset().tables) - declared)

        # Seeding a table the Knowledge Base does not describe means the schema and
        # the data have drifted apart.
        assert unknown == []

    def test_generated_rows_only_use_declared_columns(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        dataset = build_demo_dataset()

        mismatches: list[str] = []
        for table in knowledge_base.tables:
            rows = dataset.tables.get(table.name)
            if not rows:
                continue
            declared = {column.name for column in table.columns}
            for column in set(rows[0]) - declared:
                mismatches.append(f"{table.name}.{column}")

        assert sorted(mismatches) == []
