"""Point the agent at a database, and materialise the Knowledge Base into it."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from nl2sql.database.engine import build_engine, check_connection
from nl2sql.database.sample_data import missing_tables
from nl2sql.database.seed import create_schema
from nl2sql.dialects import canonical_dialect_name, dialect_names, get_dialect
from nl2sql.exceptions import ConfigurationError, NL2SQLError
from nl2sql.ui.state import Workspace


def _render_connection(workspace: Workspace) -> None:
    """Test and switch the database connection."""
    st.caption(
        "Any engine SQLAlchemy can reach. Two things have to agree: the **URL** says "
        "where the data lives, the **dialect** says which SQL to write for it — "
        "SQLite date functions are a syntax error on Postgres, and Postgres intervals "
        "evaluate to `NULL` on SQLite rather than failing."
    )

    st.info(
        f"Currently querying **{workspace.describe_database()}**, "
        f"generating **{workspace.settings.sql_dialect}** SQL."
    )

    url_column, dialect_column = st.columns([3, 1])
    database_url = url_column.text_input(
        "Connection URL",
        placeholder="postgresql://user:password@localhost:5432/observability",
        help="A SQLAlchemy URL. The password is never shown back.",
    )
    dialect = dialect_column.selectbox(
        "SQL dialect",
        dialect_names(),
        index=dialect_names().index(
            canonical_dialect_name(workspace.settings.sql_dialect)
        ),
    )

    test_column, switch_column = st.columns(2)

    if test_column.button("Test connection", width="stretch"):
        if not database_url.strip():
            st.warning("Enter a connection URL to test.")
        else:
            try:
                backend = check_connection(database_url.strip())
            except ConfigurationError as exc:
                st.error(f"Could not connect: {exc}")
            else:
                st.success(
                    f"Connected to {backend}. Nothing has changed yet — "
                    "press *Use this database* to switch."
                )

    if switch_column.button(
        "Use this database", type="primary", width="stretch"
    ):
        if not database_url.strip():
            st.warning("Enter a connection URL to switch to.")
            return
        try:
            workspace.switch_database(database_url.strip(), dialect)
        except (ConfigurationError, NL2SQLError, SQLAlchemyError) as exc:
            st.error(f"Could not switch: {exc} The previous connection is still in use.")
        else:
            st.success(f"Now querying {workspace.describe_database()}.")
            st.rerun()


def _render_schema_sync(workspace: Workspace) -> None:
    """Create declared tables that the database does not have yet."""
    st.caption(
        "Adding a table to the Knowledge Base teaches the **agent** about it. It does "
        "not create it in the **database** — so a question about a table that exists "
        "only in the Knowledge Base fails when the query runs. This closes that gap."
    )
    st.caption(
        "`CREATE TABLE IF NOT EXISTS`, so nothing is dropped and no existing row is "
        "touched."
    )

    if not st.button("Create missing tables", width="content"):
        return

    registry = workspace.registry
    try:
        engine = build_engine(workspace.settings.database_url)
        pending = missing_tables(engine, registry)
        create_schema(engine, registry, get_dialect(workspace.settings.sql_dialect))
    except (NL2SQLError, SQLAlchemyError) as exc:
        st.error(f"Could not update the database: {exc}")
        return

    if pending:
        st.success(f"Created {len(pending)} table(s): {', '.join(pending)}")
    else:
        st.info(
            f"Nothing to build — all {len(registry.tables)} declared tables already "
            "exist."
        )


def _render_contents(workspace: Workspace) -> None:
    """Show how many rows each declared table currently holds."""
    if not st.button("Count the rows", width="content"):
        return

    from sqlalchemy import text

    registry = workspace.registry
    try:
        engine = build_engine(workspace.settings.database_url)
        with engine.connect() as connection:
            counts = []
            for table in registry.tables:
                try:
                    total = connection.execute(
                        text(f"SELECT COUNT(*) FROM {table.name}")  # noqa: S608
                    ).scalar_one()
                except SQLAlchemyError:
                    total = None
                counts.append(
                    {
                        "Table": table.name,
                        "Rows": "missing" if total is None else f"{total:,}",
                    }
                )
    except (NL2SQLError, SQLAlchemyError) as exc:
        st.error(f"Could not read the database: {exc}")
        return

    st.dataframe(pd.DataFrame(counts), width="stretch", hide_index=True)


def render(workspace: Workspace) -> None:
    """Draw the Database page."""
    st.header("Database")

    connect_tab, build_tab, contents_tab = st.tabs(
        ["Connection", "Build the schema", "What is in it"]
    )

    with connect_tab:
        _render_connection(workspace)
    with build_tab:
        _render_schema_sync(workspace)
    with contents_tab:
        _render_contents(workspace)
