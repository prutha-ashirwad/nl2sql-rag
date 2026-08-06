"""Turning Knowledge Base entities into retrievable documents, one per entity."""

from __future__ import annotations

from nl2sql.knowledge_base.models import (
    BusinessTerm,
    ExampleQuery,
    RelationshipMetadata,
    SQLRule,
    TableMetadata,
)
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.retrieval.base import Document, DocumentKind


def _format_column(column) -> str:  # noqa: ANN001 - ColumnMetadata, kept local
    """Render one column as a single searchable, human-readable line."""
    parts = [f"- {column.name} ({column.data_type}): {column.description}"]

    if column.is_primary_key:
        parts.append("[primary key]")
    if not column.nullable:
        parts.append("[not null]")
    if column.unit:
        parts.append(f"[unit: {column.unit}]")
    if column.allowed_values:
        parts.append(f"[allowed values: {', '.join(column.allowed_values)}]")
    if column.synonyms:
        parts.append(f"[also called: {', '.join(column.synonyms)}]")
    if column.value_synonyms:
        phrasings = "; ".join(
            f"{value} = {', '.join(synonyms)}"
            for value, synonyms in column.value_synonyms.items()
        )
        parts.append(f"[everyday wording: {phrasings}]")

    return " ".join(parts)


def build_table_document(table: TableMetadata) -> Document:
    """Build the retrievable document for one table."""
    lines = [
        f"TABLE: {table.name}",
        f"Domain: {table.domain}",
        f"Description: {table.description}",
        f"Business definition: {table.business_definition}",
        f"Grain: {table.grain}",
    ]

    if table.synonyms:
        lines.append(f"Also known as: {', '.join(table.synonyms)}")
    if table.primary_key:
        lines.append(f"Primary key: {', '.join(table.primary_key)}")

    lines.append("Columns:")
    lines.extend(_format_column(column) for column in table.columns)

    if table.foreign_keys:
        lines.append("Foreign keys:")
        lines.extend(
            f"- {table.name}.{fk.column} -> "
            f"{fk.references_table}.{fk.references_column}: {fk.description}"
            for fk in table.foreign_keys
        )

    if table.default_filters:
        lines.append(f"Default filters: {'; '.join(table.default_filters)}")

    return Document(
        id=f"table::{table.name}",
        kind=DocumentKind.TABLE,
        text="\n".join(lines),
        tables=(table.name,),
        metadata={"domain": table.domain, "table": table.name},
    )


def build_relationship_document(relationship: RelationshipMetadata) -> Document:
    """Build the retrievable document for one relationship."""
    text = "\n".join(
        [
            f"RELATIONSHIP: {relationship.name}",
            f"Join: {relationship.from_table}.{relationship.from_column} -> "
            f"{relationship.to_table}.{relationship.to_column}",
            f"Cardinality: {relationship.cardinality.value}",
            f"Recommended join type: {relationship.join_type}",
            f"Description: {relationship.description}",
            f"Business meaning: {relationship.business_meaning}",
        ]
    )

    return Document(
        id=f"relationship::{relationship.name}",
        kind=DocumentKind.RELATIONSHIP,
        text=text,
        tables=(relationship.from_table, relationship.to_table),
        metadata={"join_type": relationship.join_type},
    )


def build_rule_document(rule: SQLRule) -> Document:
    """Build the retrievable document for one SQL generation rule."""
    lines = [f"SQL RULE {rule.id} ({rule.category}): {rule.rule}"]

    if rule.rationale:
        lines.append(f"Rationale: {rule.rationale}")
    if rule.example:
        lines.append(f"Example: {rule.example}")

    return Document(
        id=f"rule::{rule.id}",
        kind=DocumentKind.RULE,
        text="\n".join(lines),
        tables=tuple(rule.applies_to),
        metadata={"category": rule.category, "rule_id": rule.id},
    )


def build_glossary_document(term: BusinessTerm) -> Document:
    """Build the retrievable document for one glossary term."""
    lines = [f"BUSINESS TERM: {term.term}", f"Definition: {term.definition}"]

    if term.synonyms:
        lines.append(f"Synonyms: {', '.join(term.synonyms)}")
    if term.maps_to_tables:
        lines.append(f"Relevant tables: {', '.join(term.maps_to_tables)}")
    if term.maps_to_columns:
        lines.append(f"Relevant columns: {', '.join(term.maps_to_columns)}")
    if term.sql_hint:
        lines.append(f"SQL hint: {term.sql_hint}")

    return Document(
        id=f"glossary::{term.term}",
        kind=DocumentKind.GLOSSARY,
        text="\n".join(lines),
        tables=tuple(term.maps_to_tables),
        metadata={"term": term.term},
    )


def build_example_document(example: ExampleQuery) -> Document:
    """Build the retrievable document for one curated example query."""
    lines = [f"EXAMPLE QUESTION: {example.question}"]

    if example.explanation:
        lines.append(f"Reasoning: {example.explanation}")
    lines.append(f"SQL:\n{example.sql.strip()}")

    return Document(
        id=f"example::{example.id}",
        kind=DocumentKind.EXAMPLE,
        text="\n".join(lines),
        tables=tuple(example.tables_used),
        metadata={"example_id": example.id, "question": example.question},
    )


def build_documents(registry: KnowledgeBaseRegistry) -> list[Document]:
    """Build the full retrievable corpus for a Knowledge Base."""
    documents: list[Document] = []

    documents.extend(build_table_document(table) for table in registry.tables)
    documents.extend(
        build_relationship_document(relationship)
        for relationship in registry.relationships
    )
    documents.extend(build_rule_document(rule) for rule in registry.rules)
    documents.extend(build_glossary_document(term) for term in registry.glossary)
    documents.extend(build_example_document(example) for example in registry.examples)

    return documents
