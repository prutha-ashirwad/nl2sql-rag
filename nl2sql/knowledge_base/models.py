"""Typed models describing the contents of the Knowledge Base."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base model that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Cardinality(str, Enum):
    """Cardinality of a relationship between two tables."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class ColumnRole(str, Enum):
    """How a column is typically used when composing a query."""

    IDENTIFIER = "identifier"
    DIMENSION = "dimension"
    MEASURE = "measure"
    TIMESTAMP = "timestamp"
    FLAG = "flag"
    DESCRIPTIVE = "descriptive"


class ColumnMetadata(StrictModel):
    """A single column, including the semantics needed to reason about it."""

    name: str
    data_type: str
    description: str
    role: ColumnRole = ColumnRole.DESCRIPTIVE
    nullable: bool = True
    is_primary_key: bool = False
    unit: str | None = None
    allowed_values: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    example_values: list[str] = Field(default_factory=list)
    value_synonyms: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Everyday wording for each allowed value, e.g. FAILED -> "
            "['failure', 'failing', 'error']. Lets a question phrased in business "
            "language resolve to the exact stored literal."
        ),
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.islower() or " " in value:
            raise ValueError(f"Column name must be lower snake_case: {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_value_synonyms(self) -> ColumnMetadata:
        unknown = set(self.value_synonyms) - set(self.allowed_values)
        if unknown:
            raise ValueError(
                f"Column {self.name!r} declares value synonyms for values that are "
                f"not in allowed_values: {sorted(unknown)}"
            )
        return self

    @property
    def is_enumerated(self) -> bool:
        """True when the column only accepts a fixed set of values."""
        return bool(self.allowed_values)


class ForeignKeyMetadata(StrictModel):
    """A foreign key declared on the owning table."""

    column: str
    references_table: str
    references_column: str
    description: str = ""


class TableMetadata(StrictModel):
    """Full metadata for one physical table."""

    name: str
    schema_name: str = "public"
    description: str
    business_definition: str
    domain: str
    grain: str = Field(
        description="What a single row represents, e.g. 'one monitoring observation'."
    )
    preferred_alias: str | None = Field(
        default=None,
        description=(
            "Alias to use for this table in generated SQL. Declaring one for the "
            "frequently queried tables keeps output readable; the rest are derived "
            "automatically."
        ),
    )
    columns: list[ColumnMetadata]
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyMetadata] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    default_filters: list[str] = Field(
        default_factory=list,
        description="SQL predicates that should normally be applied to this table.",
    )
    tags: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.islower() or " " in value:
            raise ValueError(f"Table name must be lower snake_case: {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_keys_exist(self) -> TableMetadata:
        column_names = {column.name for column in self.columns}

        missing_pk = set(self.primary_key) - column_names
        if missing_pk:
            raise ValueError(
                f"Table {self.name!r} declares unknown primary key column(s): "
                f"{sorted(missing_pk)}"
            )

        for foreign_key in self.foreign_keys:
            if foreign_key.column not in column_names:
                raise ValueError(
                    f"Table {self.name!r} declares a foreign key on unknown column "
                    f"{foreign_key.column!r}"
                )
        return self

    @property
    def qualified_name(self) -> str:
        """Table name as it should appear in generated SQL."""
        return self.name

    def column(self, name: str) -> ColumnMetadata | None:
        """Return the column with ``name``, or ``None`` when absent."""
        return next((col for col in self.columns if col.name == name), None)

    def columns_with_role(self, role: ColumnRole) -> list[ColumnMetadata]:
        """Return every column that plays ``role`` in this table."""
        return [column for column in self.columns if column.role is role]


class RelationshipMetadata(StrictModel):
    """A curated, business-meaningful join between two tables."""

    name: str
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    cardinality: Cardinality
    join_type: str = "INNER"
    description: str = ""
    business_meaning: str = ""
    traversal_cost: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Relative preference when several join paths connect the same pair of "
            "tables. Leave at 1.0 for the natural business hierarchy and raise it "
            "for incidental links, so path finding prefers the meaningful route."
        ),
    )

    @field_validator("join_type")
    @classmethod
    def _validate_join_type(cls, value: str) -> str:
        allowed = {"INNER", "LEFT", "RIGHT", "FULL"}
        normalised = value.strip().upper()
        if normalised not in allowed:
            raise ValueError(f"join_type must be one of {sorted(allowed)}: {value!r}")
        return normalised

    def to_join_clause(self, from_alias: str, to_alias: str) -> str:
        """Render this relationship as a SQL ``JOIN`` clause."""
        return (
            f"{self.join_type} JOIN {self.to_table} {to_alias} "
            f"ON {from_alias}.{self.from_column} = {to_alias}.{self.to_column}"
        )


class SQLRule(StrictModel):
    """A SQL generation rule that constrains how queries must be written."""

    id: str
    category: str
    rule: str
    rationale: str = ""
    applies_to: list[str] = Field(
        default_factory=list,
        description="Table names this rule is scoped to; empty means global.",
    )
    example: str | None = None

    def applies_to_tables(self, table_names: set[str]) -> bool:
        """True when the rule is global or targets one of ``table_names``."""
        if not self.applies_to:
            return True
        return bool(set(self.applies_to) & table_names)


class BusinessTerm(StrictModel):
    """A domain term mapped onto concrete schema elements."""

    term: str
    definition: str
    synonyms: list[str] = Field(default_factory=list)
    maps_to_tables: list[str] = Field(default_factory=list)
    maps_to_columns: list[str] = Field(default_factory=list)
    sql_hint: str | None = None
    metric_expression: str | None = Field(
        default=None,
        description=(
            "A complete aggregate expression, written with fully qualified column "
            "names. Declaring one turns the term into a computable metric, so "
            "'success rate' produces a ratio rather than a count. Table names are "
            "rewritten to the query's aliases before the expression is emitted."
        ),
    )
    metric_alias: str | None = Field(
        default=None, description="Result column name for ``metric_expression``."
    )

    @model_validator(mode="after")
    def _validate_metric(self) -> BusinessTerm:
        if self.metric_expression and not self.metric_alias:
            raise ValueError(
                f"Glossary term {self.term!r} declares a metric_expression but no "
                f"metric_alias to name the resulting column"
            )
        return self

    @property
    def is_metric(self) -> bool:
        """True when this term can be computed as an aggregate expression."""
        return bool(self.metric_expression)


class ExampleQuery(StrictModel):
    """A curated natural-language question paired with its correct SQL."""

    id: str
    question: str
    sql: str
    explanation: str = ""
    tables_used: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class KnowledgeBase(StrictModel):
    """The complete, validated Knowledge Base."""

    tables: list[TableMetadata]
    relationships: list[RelationshipMetadata]
    rules: list[SQLRule]
    glossary: list[BusinessTerm]
    examples: list[ExampleQuery]

    @model_validator(mode="after")
    def _validate_references(self) -> KnowledgeBase:
        """Ensure every declared reference points at something that exists."""
        table_index: dict[str, TableMetadata] = {t.name: t for t in self.tables}

        duplicates = len(self.tables) - len(table_index)
        if duplicates:
            raise ValueError(f"Knowledge Base declares {duplicates} duplicate table(s)")

        for table in self.tables:
            for foreign_key in table.foreign_keys:
                self._assert_column_exists(
                    table_index,
                    foreign_key.references_table,
                    foreign_key.references_column,
                    context=f"foreign key {table.name}.{foreign_key.column}",
                )

        for relationship in self.relationships:
            self._assert_column_exists(
                table_index,
                relationship.from_table,
                relationship.from_column,
                context=f"relationship {relationship.name!r} (from)",
            )
            self._assert_column_exists(
                table_index,
                relationship.to_table,
                relationship.to_column,
                context=f"relationship {relationship.name!r} (to)",
            )

        for example in self.examples:
            unknown = set(example.tables_used) - set(table_index)
            if unknown:
                raise ValueError(
                    f"Example {example.id!r} references unknown table(s): "
                    f"{sorted(unknown)}"
                )

        return self

    @staticmethod
    def _assert_column_exists(
        table_index: dict[str, TableMetadata],
        table_name: str,
        column_name: str,
        *,
        context: str,
    ) -> None:
        table = table_index.get(table_name)
        if table is None:
            raise ValueError(f"{context} references unknown table {table_name!r}")
        if table.column(column_name) is None:
            raise ValueError(
                f"{context} references unknown column {table_name}.{column_name}"
            )

    def summary(self) -> dict[str, Any]:
        """Counts of each Knowledge Base entity."""
        return {
            "tables": len(self.tables),
            "columns": sum(len(table.columns) for table in self.tables),
            "relationships": len(self.relationships),
            "rules": len(self.rules),
            "glossary_terms": len(self.glossary),
            "example_queries": len(self.examples),
        }
