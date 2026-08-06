"""Tests for the per-engine SQL registry.

Two kinds of divergence are covered: how a relative time window is written, and how a
declared column type is spelled. Both fail quietly rather than loudly when they are
wrong — SQLite evaluates an unknown date modifier to NULL, and a mistyped column
accepts the wrong values instead of rejecting them — so they are asserted explicitly.
"""

from __future__ import annotations

import pytest

from nl2sql.analysis.question_analyzer import TimeUnit, TimeWindow
from nl2sql.database.schema_builder import build_create_table
from nl2sql.dialects import (
    MYSQL,
    POSTGRES,
    SQLITE,
    canonical_dialect_name,
    dialect_names,
    get_dialect,
    is_supported_dialect,
)
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry


class TestDialectLookup:
    def test_every_registered_dialect_resolves(self) -> None:
        for name in dialect_names():
            assert get_dialect(name).name == name

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [("postgresql", "postgres"), ("mariadb", "mysql"), ("duckdb", "postgres")],
    )
    def test_aliases_resolve(self, alias: str, expected: str) -> None:
        assert get_dialect(alias).name == expected
        assert canonical_dialect_name(alias) == expected

    def test_an_unknown_name_falls_back_to_sqlite(self) -> None:
        # Never raises: an unrecognised setting degrades to the engine that backs the
        # bundled demo database rather than taking the system down.
        assert get_dialect("teradata").name == "sqlite"

    def test_but_an_unknown_name_is_reported_as_unsupported(self) -> None:
        # Callers validating user input need to tell "SQLite on purpose" apart from
        # "SQLite because you typed something I did not recognise".
        assert is_supported_dialect("teradata") is False
        assert is_supported_dialect("postgresql") is True


class TestTimeRendering:
    @pytest.mark.parametrize(
        ("dialect", "fragment"),
        [
            (SQLITE, "datetime('now', '-24 hours')"),
            (POSTGRES, "NOW() - INTERVAL '24 hours'"),
            (MYSQL, "DATE_SUB(NOW(), INTERVAL 24 HOUR)"),
        ],
    )
    def test_each_engine_gets_its_own_syntax(self, dialect, fragment: str) -> None:  # noqa: ANN001
        assert dialect.timestamp_at_offset(24, TimeUnit.HOUR) == fragment

    def test_sqlite_converts_weeks_it_cannot_express(self) -> None:
        # SQLite accepts '-2 weeks' and evaluates the whole expression to NULL, so a
        # query filtered on it returns nothing and reports no error.
        assert SQLITE.timestamp_at_offset(2, TimeUnit.WEEK) == (
            "datetime('now', '-14 days')"
        )

    def test_an_engine_that_understands_weeks_keeps_them(self) -> None:
        assert "2 weeks" in POSTGRES.timestamp_at_offset(2, TimeUnit.WEEK)

    def test_a_closed_window_gets_an_upper_bound(self) -> None:
        window = TimeWindow(1, TimeUnit.DAY, "yesterday", include_upper_bound=True)

        predicates = POSTGRES.time_window_predicates("o.observed_at", window)

        # "Yesterday" excludes today, so the window is bounded at both ends.
        assert len(predicates) == 2
        assert predicates[1].endswith("NOW()")


class TestTypeRendering:
    @pytest.mark.parametrize(
        ("declared", "sqlite", "postgres", "mysql"),
        [
            ("BIGINT", "INTEGER", "BIGINT", "BIGINT"),
            ("VARCHAR(64)", "TEXT", "VARCHAR(64)", "VARCHAR(64)"),
            ("DECIMAL(5,2)", "REAL", "DECIMAL(5,2)", "DECIMAL(5,2)"),
            ("BOOLEAN", "INTEGER", "BOOLEAN", "TINYINT(1)"),
            ("TIMESTAMP", "TEXT", "TIMESTAMP", "DATETIME"),
        ],
    )
    def test_one_declaration_renders_per_engine(
        self, declared: str, sqlite: str, postgres: str, mysql: str
    ) -> None:
        assert SQLITE.storage_type(declared) == sqlite
        assert POSTGRES.storage_type(declared) == postgres
        assert MYSQL.storage_type(declared) == mysql

    def test_a_width_is_dropped_where_it_would_be_meaningless(self) -> None:
        # SQLite has no sized text type; TEXT(64) would merely look like one.
        assert SQLITE.storage_type("VARCHAR(255)") == "TEXT"

    def test_an_unknown_type_passes_through(self) -> None:
        # A type no dialect remaps is already portable, or is one the author meant
        # literally. Either way, inventing a substitute would be worse.
        assert POSTGRES.storage_type("jsonb") == "JSONB"


class TestGeneratedDDL:
    def test_the_same_table_materialises_on_every_engine(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        table = registry.get_table("observations")
        assert table is not None

        sqlite_ddl = build_create_table(table, SQLITE)
        postgres_ddl = build_create_table(table, POSTGRES)

        # One Knowledge Base declaration, two engines, no second schema definition.
        assert "observed_at TEXT" in sqlite_ddl
        assert "observed_at TIMESTAMP" in postgres_ddl
        assert "PRIMARY KEY (observation_id)" in sqlite_ddl
        assert "PRIMARY KEY (observation_id)" in postgres_ddl

    def test_declared_widths_survive_on_an_engine_that_has_them(
        self, registry: KnowledgeBaseRegistry
    ) -> None:
        table = registry.get_table("observations")
        assert table is not None

        assert "VARCHAR(16)" in build_create_table(table, POSTGRES)
