"""Tests for Knowledge Base loading, validation and indexing."""

from __future__ import annotations

import pytest

from nl2sql.exceptions import KnowledgeBaseError
from nl2sql.knowledge_base.loader import load_knowledge_base
from nl2sql.knowledge_base.models import (
    Cardinality,
    ColumnMetadata,
    ColumnRole,
    KnowledgeBase,
    RelationshipMetadata,
    TableMetadata,
)

# The brief asks for metadata covering roughly twenty tables and at least five
# major relationships; these assertions keep the delivered KB above that bar.
MINIMUM_TABLES = 20
MINIMUM_RELATIONSHIPS = 5


class TestKnowledgeBaseContent:
    """The bundled Knowledge Base meets the documented coverage requirements."""

    def test_declares_enough_tables(self, knowledge_base: KnowledgeBase) -> None:
        assert len(knowledge_base.tables) >= MINIMUM_TABLES

    def test_declares_enough_relationships(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        assert len(knowledge_base.relationships) >= MINIMUM_RELATIONSHIPS

    def test_every_table_is_fully_documented(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        for table in knowledge_base.tables:
            assert table.description.strip(), f"{table.name} has no description"
            assert table.business_definition.strip(), f"{table.name} has no definition"
            assert table.grain.strip(), f"{table.name} has no declared grain"
            assert table.primary_key, f"{table.name} has no primary key"
            assert table.columns, f"{table.name} has no columns"

    def test_every_column_is_described(self, knowledge_base: KnowledgeBase) -> None:
        for table in knowledge_base.tables:
            for column in table.columns:
                assert column.description.strip(), (
                    f"{table.name}.{column.name} has no description"
                )

    def test_ships_sql_rules_and_examples(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        assert knowledge_base.rules
        assert knowledge_base.glossary
        assert knowledge_base.examples

    def test_primary_keys_are_marked_on_their_columns(
        self, knowledge_base: KnowledgeBase
    ) -> None:
        for table in knowledge_base.tables:
            for key_column in table.primary_key:
                column = table.column(key_column)
                assert column is not None and column.is_primary_key


class TestValidation:
    """Malformed Knowledge Base content is rejected at load time."""

    def test_rejects_missing_directory(self, tmp_path) -> None:
        with pytest.raises(KnowledgeBaseError, match="not found"):
            load_knowledge_base(tmp_path / "does-not-exist")

    def test_rejects_unknown_foreign_key_target(self) -> None:
        table = TableMetadata(
            name="orders",
            description="Orders.",
            business_definition="An order.",
            domain="sales",
            grain="one row per order",
            primary_key=["order_id"],
            columns=[
                ColumnMetadata(
                    name="order_id",
                    data_type="INTEGER",
                    description="Key.",
                    role=ColumnRole.IDENTIFIER,
                    is_primary_key=True,
                ),
                ColumnMetadata(
                    name="customer_id",
                    data_type="INTEGER",
                    description="Customer.",
                    role=ColumnRole.IDENTIFIER,
                ),
            ],
            foreign_keys=[
                {
                    "column": "customer_id",
                    "references_table": "customers",
                    "references_column": "customer_id",
                }
            ],
        )

        with pytest.raises(ValueError, match="unknown table"):
            KnowledgeBase(
                tables=[table], relationships=[], rules=[], glossary=[], examples=[]
            )

    def test_rejects_relationship_to_unknown_column(self) -> None:
        table = TableMetadata(
            name="orders",
            description="Orders.",
            business_definition="An order.",
            domain="sales",
            grain="one row per order",
            primary_key=["order_id"],
            columns=[
                ColumnMetadata(
                    name="order_id",
                    data_type="INTEGER",
                    description="Key.",
                    role=ColumnRole.IDENTIFIER,
                    is_primary_key=True,
                )
            ],
        )
        relationship = RelationshipMetadata(
            name="bad",
            from_table="orders",
            from_column="missing_column",
            to_table="orders",
            to_column="order_id",
            cardinality=Cardinality.MANY_TO_ONE,
        )

        with pytest.raises(ValueError, match="unknown column"):
            KnowledgeBase(
                tables=[table],
                relationships=[relationship],
                rules=[],
                glossary=[],
                examples=[],
            )

    def test_rejects_uppercase_table_name(self) -> None:
        with pytest.raises(ValueError, match="snake_case"):
            TableMetadata(
                name="Orders",
                description="Orders.",
                business_definition="An order.",
                domain="sales",
                grain="one row per order",
                columns=[],
            )

    def test_rejects_value_synonym_for_undeclared_value(self) -> None:
        with pytest.raises(ValueError, match="not in allowed_values"):
            ColumnMetadata(
                name="status",
                data_type="VARCHAR(16)",
                description="Status.",
                allowed_values=["OPEN"],
                value_synonyms={"CLOSED": ["done"]},
            )

    def test_rejects_metric_without_alias(self) -> None:
        from nl2sql.knowledge_base.models import BusinessTerm

        with pytest.raises(ValueError, match="metric_alias"):
            BusinessTerm(
                term="failure rate",
                definition="Rate of failures.",
                metric_expression="COUNT(*)",
            )

    def test_rejects_unknown_yaml_keys(self) -> None:
        with pytest.raises(ValueError):
            ColumnMetadata(
                name="status",
                data_type="VARCHAR(16)",
                description="Status.",
                discription="typo",  # noqa: F821 - deliberately misspelled
            )
