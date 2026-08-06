"""In-memory indexes over the Knowledge Base.

Built once at start-up and read-only afterwards, so it is safe to share across
requests and threads.
"""

from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass
from functools import cached_property

from nl2sql.knowledge_base.models import (
    BusinessTerm,
    Cardinality,
    ColumnMetadata,
    ColumnRole,
    ExampleQuery,
    KnowledgeBase,
    RelationshipMetadata,
    SQLRule,
    TableMetadata,
)
from nl2sql.logging_config import get_logger

logger = get_logger(__name__)

_LEXICON_STOP_WORDS = frozenset(
    {"the", "a", "an", "of", "and", "or", "in", "on", "by", "to", "for", "is"}
)


def _singular(name: str) -> str:
    """Return the singular form of a table name for label-column lookups."""
    return name[:-1] if name.endswith("s") and not name.endswith("ss") else name


@dataclass(frozen=True, slots=True)
class JoinStep:
    """One hop in a join path, already oriented in the direction of travel."""

    relationship: RelationshipMetadata
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    join_type: str
    preserves_grain: bool
    """True when the hop cannot multiply rows on the source side."""

    def to_sql(self, source_alias: str, target_alias: str) -> str:
        """Render the hop as a SQL ``JOIN`` clause."""
        return (
            f"{self.join_type} JOIN {self.target_table} {target_alias} "
            f"ON {source_alias}.{self.source_column} = "
            f"{target_alias}.{self.target_column}"
        )


@dataclass(frozen=True, slots=True)
class EnumValueMatch:
    """A literal value traced back to the column that accepts it."""

    table: str
    column: str
    value: str
    from_synonym: bool = False
    """True when the phrase is a curated ``value_synonyms`` entry."""


@dataclass(frozen=True, slots=True)
class TableMatch:
    """A table matched from a phrase, with how strong the evidence was."""

    table: str
    score: float


@dataclass(frozen=True, slots=True)
class DimensionMatch:
    """A concrete column that results can meaningfully be grouped by."""

    table: str
    column: str


# Evidence weights for phrase-to-table matching.
_WEIGHT_TABLE_NAME = 3.0
_WEIGHT_TABLE_SYNONYM = 2.5
_WEIGHT_COLUMN_SYNONYM = 1.5
_WEIGHT_GLOSSARY_TERM = 1.0

# The weakest evidence that still counts as the question *naming* a table.
NAMED_TABLE_EVIDENCE = _WEIGHT_TABLE_SYNONYM


class KnowledgeBaseRegistry:
    """Queryable indexes over a loaded :class:`KnowledgeBase`."""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._kb = knowledge_base
        self._tables: dict[str, TableMetadata] = {
            table.name: table for table in knowledge_base.tables
        }
        self._aliases: dict[str, str] = self._build_aliases()
        self._adjacency: dict[str, list[JoinStep]] = self._build_join_graph()
        logger.debug(
            "Registry built over %d tables and %d relationships",
            len(self._tables),
            len(knowledge_base.relationships),
        )

    # -- Basic accessors ------------------------------------------------------

    @property
    def knowledge_base(self) -> KnowledgeBase:
        """The underlying Knowledge Base."""
        return self._kb

    @property
    def table_names(self) -> list[str]:
        """Every table name, in declaration order."""
        return list(self._tables)

    @property
    def tables(self) -> list[TableMetadata]:
        """Every table's metadata."""
        return list(self._tables.values())

    @property
    def relationships(self) -> list[RelationshipMetadata]:
        """Every declared relationship."""
        return list(self._kb.relationships)

    @property
    def rules(self) -> list[SQLRule]:
        """Every SQL generation rule."""
        return list(self._kb.rules)

    @property
    def glossary(self) -> list[BusinessTerm]:
        """Every business glossary term."""
        return list(self._kb.glossary)

    @property
    def examples(self) -> list[ExampleQuery]:
        """Every curated example query."""
        return list(self._kb.examples)

    def get_table(self, name: str) -> TableMetadata | None:
        """Return metadata for ``name``, or ``None`` when it is not declared."""
        return self._tables.get(name.lower())

    def has_table(self, name: str) -> bool:
        """True when ``name`` is a declared table."""
        return name.lower() in self._tables

    def get_column(self, table: str, column: str) -> ColumnMetadata | None:
        """Return metadata for ``table.column``, or ``None`` when absent."""
        table_metadata = self.get_table(table)
        return table_metadata.column(column.lower()) if table_metadata else None

    def alias_for(self, table: str) -> str:
        """Return the stable short alias assigned to ``table``."""
        return self._aliases.get(table.lower(), table.lower()[:2])

    # -- Derived indexes ------------------------------------------------------

    @cached_property
    def _term_lexicon(self) -> dict[str, dict[str, float]]:
        """Map lowercase phrases onto the tables they identify, with a weight.

        Each entry holds the strongest weight seen for that phrase and table.
        """
        lexicon: dict[str, dict[str, float]] = {}

        def register(phrase: str, table_name: str, weight: float) -> None:
            key = phrase.strip().lower()
            if not key or key in _LEXICON_STOP_WORDS:
                return
            targets = lexicon.setdefault(key, {})
            targets[table_name] = max(targets.get(table_name, 0.0), weight)

        for table in self._kb.tables:
            register(table.name, table.name, _WEIGHT_TABLE_NAME)
            register(table.name.replace("_", " "), table.name, _WEIGHT_TABLE_NAME)
            register(_singular(table.name), table.name, _WEIGHT_TABLE_NAME)
            for synonym in table.synonyms:
                register(synonym, table.name, _WEIGHT_TABLE_SYNONYM)
            for column in table.columns:
                for synonym in column.synonyms:
                    register(synonym, table.name, _WEIGHT_COLUMN_SYNONYM)

        for term in self._kb.glossary:
            for table_name in term.maps_to_tables:
                register(term.term, table_name, _WEIGHT_GLOSSARY_TERM)
                for synonym in term.synonyms:
                    register(synonym, table_name, _WEIGHT_GLOSSARY_TERM)

        return lexicon

    @cached_property
    def _dimension_lexicon(self) -> dict[str, list[DimensionMatch]]:
        """Map phrases onto columns that results can sensibly be grouped by.

        Narrower than the table lexicon: measure phrases such as "failure count"
        must not resolve here, or they would be grouped by instead of aggregated.
        """
        lexicon: dict[str, list[DimensionMatch]] = {}

        def register(phrase: str, table: str, column: str) -> None:
            key = phrase.strip().lower()
            if not key or key in _LEXICON_STOP_WORDS:
                return
            match = DimensionMatch(table=table, column=column)
            entries = lexicon.setdefault(key, [])
            if match not in entries:
                entries.append(match)

        for table in self._kb.tables:
            label = self.label_column(table.name)
            if label is not None:
                register(table.name, table.name, label)
                register(table.name.replace("_", " "), table.name, label)
                register(_singular(table.name), table.name, label)
                for synonym in table.synonyms:
                    register(synonym, table.name, label)

            for column in table.columns_with_role(ColumnRole.DIMENSION):
                register(column.name, table.name, column.name)
                register(column.name.replace("_", " "), table.name, column.name)
                for synonym in column.synonyms:
                    register(synonym, table.name, column.name)

        for term in self._kb.glossary:
            for table_name in term.maps_to_tables:
                label = self.label_column(table_name)
                if label is None:
                    continue
                register(term.term, table_name, label)
                for synonym in term.synonyms:
                    register(synonym, table_name, label)

        return lexicon

    def label_column(self, table_name: str) -> str | None:
        """Return the column that names a row of ``table_name``.

        ``None`` when there is no naming column, as on fact tables.
        """
        table = self.get_table(table_name)
        if table is None:
            return None

        singular = _singular(table.name)
        for candidate in (f"{singular}_name", f"{singular}_code", f"{singular}_key"):
            if table.column(candidate) is not None:
                return candidate

        for column in table.columns:
            if column.name.endswith("_name"):
                return column.name

        return None

    @cached_property
    def _enum_index(self) -> dict[str, list[EnumValueMatch]]:
        """Map lowercase phrases onto the enumerated column values they denote."""
        index: dict[str, list[EnumValueMatch]] = {}

        def register(
            phrase: str, table: str, column: str, value: str, *, from_synonym: bool
        ) -> None:
            match = EnumValueMatch(
                table=table, column=column, value=value, from_synonym=from_synonym
            )
            entries = index.setdefault(phrase.strip().lower(), [])
            if match not in entries:
                entries.append(match)

        for table in self._kb.tables:
            for column in table.columns:
                for value in column.allowed_values:
                    register(
                        value, table.name, column.name, value, from_synonym=False
                    )
                    for synonym in column.value_synonyms.get(value, []):
                        register(
                            synonym, table.name, column.name, value, from_synonym=True
                        )

        return index

    def resolve_tables_for_phrase(self, phrase: str) -> list[TableMatch]:
        """Return the tables a business phrase refers to, strongest match first."""
        matches = self._term_lexicon.get(phrase.strip().lower(), {})
        return [
            TableMatch(table=table, score=score)
            for table, score in sorted(
                matches.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    def resolve_dimension_for_phrase(self, phrase: str) -> list[DimensionMatch]:
        """Return the groupable columns a phrase refers to."""
        return list(self._dimension_lexicon.get(phrase.strip().lower(), []))

    @cached_property
    def _metric_lexicon(self) -> dict[str, BusinessTerm]:
        """Map phrases onto the glossary terms that declare a computable metric."""
        lexicon: dict[str, BusinessTerm] = {}

        for term in self._kb.glossary:
            if not term.is_metric:
                continue
            for phrase in (term.term, *term.synonyms):
                lexicon.setdefault(phrase.strip().lower(), term)

        return lexicon

    def resolve_metric_for_phrase(self, phrase: str) -> BusinessTerm | None:
        """Return the metric a phrase names, if the glossary declares one."""
        return self._metric_lexicon.get(phrase.strip().lower())

    def tables_in_expression(self, expression: str) -> list[str]:
        """Return the declared tables a metric expression references."""
        return [
            name
            for name in self._tables
            if re.search(rf"\b{re.escape(name)}\.", expression)
        ]

    def resolve_enum_value(self, value: str) -> list[EnumValueMatch]:
        """Return every column that declares ``value`` as an allowed value."""
        return list(self._enum_index.get(value.strip().lower(), []))

    def timestamp_columns(self, table: str) -> list[ColumnMetadata]:
        """Return the timestamp columns of ``table``, event time before bookkeeping."""
        table_metadata = self.get_table(table)
        if table_metadata is None:
            return []
        columns = table_metadata.columns_with_role(ColumnRole.TIMESTAMP)
        return sorted(columns, key=lambda column: column.name in {"created_at"})

    def primary_timestamp_column(self, table: str) -> ColumnMetadata | None:
        """Return the column a time-window filter should be applied to."""
        columns = self.timestamp_columns(table)
        return columns[0] if columns else None

    def rules_for_tables(self, table_names: set[str]) -> list[SQLRule]:
        """Return the rules that apply to at least one of ``table_names``."""
        return [rule for rule in self._kb.rules if rule.applies_to_tables(table_names)]

    # -- Join graph -----------------------------------------------------------

    def _build_aliases(self) -> dict[str, str]:
        """Assign a short, stable and unique alias to every table.

        Declared ``preferred_alias`` values are claimed first; the rest are derived
        from the table name's initials, with a numeric suffix on collision.
        """
        aliases: dict[str, str] = {}
        taken: set[str] = set()

        for table in self._tables.values():
            if table.preferred_alias and table.preferred_alias not in taken:
                taken.add(table.preferred_alias)
                aliases[table.name] = table.preferred_alias

        for name in self._tables:
            if name in aliases:
                continue

            parts = name.split("_")
            base = "".join(part[0] for part in parts if part) or name[:1]

            alias = base
            suffix = 1
            while alias in taken:
                suffix += 1
                alias = f"{base}{suffix}"

            taken.add(alias)
            aliases[name] = alias

        return aliases

    def _build_join_graph(self) -> dict[str, list[JoinStep]]:
        """Build a bidirectional adjacency list from declared relationships.

        The declared join type constrains only the forward direction, so reverse
        hops are rendered as INNER.
        """
        adjacency: dict[str, list[JoinStep]] = {name: [] for name in self._tables}

        # Travelling the declared direction of these moves towards the "one" side.
        forward_preserves = {Cardinality.MANY_TO_ONE, Cardinality.ONE_TO_ONE}
        reverse_preserves = {Cardinality.ONE_TO_MANY, Cardinality.ONE_TO_ONE}

        for relationship in self._kb.relationships:
            adjacency[relationship.from_table].append(
                JoinStep(
                    relationship=relationship,
                    source_table=relationship.from_table,
                    source_column=relationship.from_column,
                    target_table=relationship.to_table,
                    target_column=relationship.to_column,
                    join_type=relationship.join_type,
                    preserves_grain=relationship.cardinality in forward_preserves,
                )
            )
            adjacency[relationship.to_table].append(
                JoinStep(
                    relationship=relationship,
                    source_table=relationship.to_table,
                    source_column=relationship.to_column,
                    target_table=relationship.from_table,
                    target_column=relationship.from_column,
                    join_type="INNER",
                    preserves_grain=relationship.cardinality in reverse_preserves,
                )
            )

        return adjacency

    def find_join_path(
        self, from_table: str, to_table: str, *, preserve_grain: bool = False
    ) -> list[JoinStep] | None:
        """Return the cheapest declared join path between two tables.

        Cost is the relationships' ``traversal_cost``; ties break on hop count and
        then table name, so the result is identical on every run.

        Args:
            from_table: Table to start from.
            to_table: Table to reach.
            preserve_grain: When true, only consider hops that cannot multiply rows.

        Returns:
            The ordered hops, an empty list when the tables are the same, or ``None``
            when no declared path connects them.
        """
        source = from_table.lower()
        target = to_table.lower()

        if source == target:
            return []
        if source not in self._adjacency or target not in self._adjacency:
            return None

        # (total cost, hop count, table, path) — the tie-breakers keep heap ordering
        # total, so no two entries ever compare equal.
        queue: list[tuple[float, int, str, list[JoinStep]]] = [(0.0, 0, source, [])]
        best_cost: dict[str, float] = {source: 0.0}

        while queue:
            cost, hops, current, path = heapq.heappop(queue)

            if current == target:
                return path
            if cost > best_cost.get(current, math.inf):
                continue

            for step in sorted(
                self._adjacency[current], key=lambda item: item.target_table
            ):
                if preserve_grain and not step.preserves_grain:
                    continue

                next_cost = cost + step.relationship.traversal_cost
                if next_cost >= best_cost.get(step.target_table, math.inf):
                    continue

                best_cost[step.target_table] = next_cost
                heapq.heappush(
                    queue,
                    (next_cost, hops + 1, step.target_table, [*path, step]),
                )

        return None

    def build_join_plan(
        self,
        base_table: str,
        required_tables: list[str],
        *,
        preserve_grain: bool = False,
    ) -> tuple[list[JoinStep], list[str]]:
        """Compute the joins needed to reach every table in ``required_tables``.

        Args:
            base_table: Table the query is anchored on.
            required_tables: Tables that must be reachable, in priority order.
            preserve_grain: When true, refuse paths that would duplicate base rows.

        Returns:
            A tuple of (ordered join steps, names of tables that are unreachable).
        """
        # A list, not a set: path selection must not depend on set iteration order.
        joined: list[str] = [base_table.lower()]
        steps: list[JoinStep] = []
        unreachable: list[str] = []

        for target in required_tables:
            target = target.lower()
            if target in joined:
                continue

            best_path = self._shortest_path_from_joined(
                joined, target, preserve_grain=preserve_grain
            )
            if best_path is None:
                unreachable.append(target)
                logger.debug(
                    "No %sjoin path from %s to %s",
                    "grain-preserving " if preserve_grain else "",
                    base_table,
                    target,
                )
                continue

            for step in best_path:
                if step.target_table not in joined:
                    steps.append(step)
                    joined.append(step.target_table)

        return steps, unreachable

    def _shortest_path_from_joined(
        self, sources: list[str], target: str, *, preserve_grain: bool
    ) -> list[JoinStep] | None:
        """Return the shortest path reaching ``target`` from any already-joined table.

        Sources are considered in join order, so the base table wins ties.
        """
        best: list[JoinStep] | None = None
        best_cost = math.inf

        for source in sources:
            path = self.find_join_path(source, target, preserve_grain=preserve_grain)
            if path is None:
                continue
            cost = sum(step.relationship.traversal_cost for step in path)
            if cost < best_cost:
                best, best_cost = path, cost

        return best
