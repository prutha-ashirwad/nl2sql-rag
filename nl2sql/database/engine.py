"""Database engine construction."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError

from nl2sql.exceptions import ConfigurationError
from nl2sql.logging_config import get_logger

logger = get_logger(__name__)

_DRIVER_HINTS: dict[str, str] = {
    "postgresql": "pip install psycopg2-binary",
    "mysql": "pip install PyMySQL",
    "mssql": "pip install pyodbc",
    "oracle": "pip install oracledb",
}


def build_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Create a SQLAlchemy engine for ``database_url``.

    For SQLite the parent directory is created if missing and foreign key
    enforcement is switched on, since SQLite leaves it off by default.
    """
    url = make_url(database_url)

    if url.get_backend_name() == "sqlite" and url.database:
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(database_url, echo=echo, future=True)

    if url.get_backend_name() == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    logger.debug(
        "Database engine created for %s", url.render_as_string(hide_password=True)
    )
    return engine


def describe_database_url(database_url: str) -> str:
    """Render a connection string with the password masked.

    An unparseable URL is returned unchanged, so the reader sees the actual typo.
    """
    try:
        return make_url(database_url).render_as_string(hide_password=True)
    except ArgumentError:
        return database_url


def check_connection(database_url: str) -> str:
    """Open ``database_url`` and run a trivial query against it.

    Returns:
        The backend name, for example ``postgresql``.

    Raises:
        ConfigurationError: if the database cannot be reached.
    """
    safe_url = describe_database_url(database_url)

    try:
        url = make_url(database_url)
    except ArgumentError as exc:
        raise ConfigurationError(
            f"{safe_url} is not a valid database URL: {exc}"
        ) from exc

    try:
        engine = create_engine(database_url, future=True)
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        finally:
            engine.dispose()
    except ModuleNotFoundError as exc:
        hint = _DRIVER_HINTS.get(url.get_backend_name(), "")
        remedy = f" Install its driver: {hint}." if hint else ""
        raise ConfigurationError(
            f"No driver installed for {url.get_backend_name()}.{remedy}"
        ) from exc
    except SQLAlchemyError as exc:
        raise ConfigurationError(f"Could not connect to {safe_url}: {exc}") from exc

    logger.info("Connection verified: %s", safe_url)
    return url.get_backend_name()
