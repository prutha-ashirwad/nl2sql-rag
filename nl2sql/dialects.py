"""Per-engine SQL differences — time arithmetic and column types — in one registry."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nl2sql.analysis.question_analyzer import TimeUnit, TimeWindow

# Only exact conversions belong here; a month is not a fixed number of days.
_EXACT_CONVERSIONS: dict[TimeUnit, tuple[int, TimeUnit]] = {
    TimeUnit.WEEK: (7, TimeUnit.DAY),
}

_PARAMETRIC_TYPE_PATTERN = re.compile(r"^([A-Z ]+?)\s*(\([^)]*\))?$")

# Types whose declared precision is worth carrying across.
_TAKES_PARAMETERS = frozenset(
    {"VARCHAR", "CHAR", "NVARCHAR", "DECIMAL", "NUMERIC", "NUMBER"}
)


@dataclass(frozen=True, slots=True)
class SQLDialect:
    """How to render engine-sensitive SQL for one database."""

    name: str
    current_timestamp: str
    relative_timestamp_template: str
    """Format string taking ``quantity`` and ``unit`` and yielding a past timestamp."""

    unit_names: dict[TimeUnit, str] = field(default_factory=dict)
    """Per-unit keyword overrides; units absent here use the enum's own value."""

    supported_units: frozenset[TimeUnit] = field(
        default_factory=lambda: frozenset(TimeUnit)
    )

    type_map: dict[str, str] = field(default_factory=dict)
    """Declared type to engine type. Types absent here are already portable."""

    # -- Time ------------------------------------------------------------------

    def normalise_unit(self, quantity: int, unit: TimeUnit) -> tuple[int, TimeUnit]:
        """Convert ``quantity`` and ``unit`` into something the dialect supports."""
        if unit in self.supported_units:
            return quantity, unit

        conversion = _EXACT_CONVERSIONS.get(unit)
        if conversion is None:
            return quantity, unit

        multiplier, target_unit = conversion
        return quantity * multiplier, target_unit

    def timestamp_at_offset(self, quantity: int, unit: TimeUnit) -> str:
        """Render an expression for "``quantity`` ``unit`` ago"."""
        quantity, unit = self.normalise_unit(quantity, unit)
        keyword = self.unit_names.get(unit, unit.value)
        return self.relative_timestamp_template.format(
            quantity=quantity, unit=keyword
        )

    def time_window_predicates(
        self, qualified_column: str, window: TimeWindow
    ) -> list[str]:
        """Render the predicates that restrict ``qualified_column`` to ``window``."""
        lower_bound = self.timestamp_at_offset(window.quantity, window.unit)
        predicates = [f"{qualified_column} >= {lower_bound}"]

        if window.include_upper_bound:
            predicates.append(f"{qualified_column} < {self.current_timestamp}")

        return predicates

    # -- Types -----------------------------------------------------------------

    def storage_type(self, declared_type: str) -> str:
        """Map a Knowledge Base column type onto this engine's own type."""
        match = _PARAMETRIC_TYPE_PATTERN.match(declared_type.strip().upper())
        if match is None:
            return declared_type.strip().upper()

        base, parameters = match.group(1).strip(), match.group(2)
        target = self.type_map.get(base, base)

        if parameters and target in _TAKES_PARAMETERS:
            return f"{target}{parameters}"
        return target


# SQLite has no week modifier, and an unrecognised one evaluates to NULL, not an error.
SQLITE = SQLDialect(
    name="sqlite",
    current_timestamp="datetime('now')",
    relative_timestamp_template="datetime('now', '-{quantity} {unit}')",
    supported_units=frozenset(
        {
            TimeUnit.MINUTE,
            TimeUnit.HOUR,
            TimeUnit.DAY,
            TimeUnit.MONTH,
            TimeUnit.YEAR,
        }
    ),
    type_map={
        "BIGINT": "INTEGER",
        "SMALLINT": "INTEGER",
        "BOOLEAN": "INTEGER",
        "VARCHAR": "TEXT",
        "CHAR": "TEXT",
        "NVARCHAR": "TEXT",
        "DATE": "TEXT",
        "TIMESTAMP": "TEXT",
        "DATETIME": "TEXT",
        "DECIMAL": "REAL",
        "NUMERIC": "REAL",
        "FLOAT": "REAL",
        "DOUBLE": "REAL",
    },
)

POSTGRES = SQLDialect(
    name="postgres",
    current_timestamp="NOW()",
    relative_timestamp_template="NOW() - INTERVAL '{quantity} {unit}'",
    type_map={
        "DATETIME": "TIMESTAMP",
        "DOUBLE": "DOUBLE PRECISION",
    },
)

# MySQL TIMESTAMP tops out in 2038, so a declared timestamp maps to DATETIME.
MYSQL = SQLDialect(
    name="mysql",
    current_timestamp="NOW()",
    relative_timestamp_template="DATE_SUB(NOW(), INTERVAL {quantity} {unit})",
    unit_names={
        TimeUnit.MINUTE: "MINUTE",
        TimeUnit.HOUR: "HOUR",
        TimeUnit.DAY: "DAY",
        TimeUnit.WEEK: "WEEK",
        TimeUnit.MONTH: "MONTH",
        TimeUnit.YEAR: "YEAR",
    },
    type_map={
        "BOOLEAN": "TINYINT(1)",
        "TIMESTAMP": "DATETIME",
        "DOUBLE": "DOUBLE",
    },
)

_DIALECTS: dict[str, SQLDialect] = {
    dialect.name: dialect for dialect in (SQLITE, POSTGRES, MYSQL)
}

# DuckDB implements Postgres syntax closely enough to share its rendering.
_DIALECT_ALIASES: dict[str, str] = {
    "postgresql": "postgres",
    "mariadb": "mysql",
    "duckdb": "postgres",
}


def get_dialect(name: str) -> SQLDialect:
    """Return the dialect named ``name``, falling back to SQLite."""
    return _DIALECTS.get(canonical_dialect_name(name), SQLITE)


def canonical_dialect_name(name: str) -> str:
    """Resolve an alias onto the name the registry uses."""
    key = name.strip().lower()
    return _DIALECT_ALIASES.get(key, key)


def is_supported_dialect(name: str) -> bool:
    """True when ``name`` names a dialect the registry knows."""
    return canonical_dialect_name(name) in _DIALECTS


def dialect_names() -> list[str]:
    """Every dialect that can be selected, for a picker or a help message."""
    return sorted(_DIALECTS)
