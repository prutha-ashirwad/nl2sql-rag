"""Safe, validated editing of the Knowledge Base.

Every change is applied to a staged copy first and only written back once it loads
cleanly, so a rejected edit leaves the live Knowledge Base unchanged.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nl2sql.exceptions import KnowledgeBaseError
from nl2sql.knowledge_base.loader import TABLES_DIRECTORY, load_knowledge_base
from nl2sql.logging_config import get_logger

logger = get_logger(__name__)

# The single-file sections, in the order they are offered in the picker.
_SINGLE_FILE_SECTIONS: tuple[str, ...] = (
    "relationships.yaml",
    "sql_rules.yaml",
    "business_glossary.yaml",
    "example_queries.yaml",
)

# Keeps `tables/inventory.yaml` distinct from a top-level `inventory.yaml`.
TABLES_PREFIX = f"{TABLES_DIRECTORY}/"


@dataclass(frozen=True, slots=True)
class AuthoringResult:
    """The outcome of validating or committing one edit."""

    ok: bool
    """True when the edit is valid, and — for a commit — was written."""

    message: str
    """One-line summary suitable for showing directly to a user."""

    detail: str = ""
    """The full validator output, when the edit was rejected."""

    summary: dict[str, int] = field(default_factory=dict)
    """Entity counts the Knowledge Base would have after the edit."""

    delta: dict[str, int] = field(default_factory=dict)
    """Change in each entity count relative to the Knowledge Base on disk."""

    @property
    def changed(self) -> bool:
        """True when the edit alters at least one entity count."""
        return any(self.delta.values())


def _is_safe_section(name: str) -> bool:
    """True when ``name`` addresses a Knowledge Base file and nothing else.

    Section names arrive from the web interface, so ``../../etc/passwd`` must not
    resolve to a writable path.
    """
    if not name.endswith((".yaml", ".yml")):
        return False
    if name in _SINGLE_FILE_SECTIONS:
        return True
    if not name.startswith(TABLES_PREFIX):
        return False

    stem = name[len(TABLES_PREFIX) :]
    return bool(stem) and "/" not in stem and stem not in {".", ".."}


class KnowledgeBaseEditor:
    """Reads and writes Knowledge Base files, refusing to persist invalid ones."""

    def __init__(self, kb_path: Path) -> None:
        self._kb_path = Path(kb_path).expanduser().resolve()
        if not self._kb_path.is_dir():
            raise KnowledgeBaseError(
                f"Knowledge Base directory not found: {self._kb_path}"
            )

    @property
    def kb_path(self) -> Path:
        """The directory being edited."""
        return self._kb_path

    def sections(self) -> list[str]:
        """Every editable file, table files first and then the shared sections."""
        tables_directory = self._kb_path / TABLES_DIRECTORY
        table_files = (
            sorted(
                f"{TABLES_PREFIX}{path.name}"
                for path in tables_directory.iterdir()
                if path.suffix in {".yaml", ".yml"} and path.is_file()
            )
            if tables_directory.is_dir()
            else []
        )
        present = [
            name for name in _SINGLE_FILE_SECTIONS if (self._kb_path / name).is_file()
        ]
        return table_files + present

    def resolve(self, section: str) -> Path:
        """Return the on-disk path for ``section``.

        Raises:
            KnowledgeBaseError: if ``section`` does not name a Knowledge Base file.
        """
        if not _is_safe_section(section):
            raise KnowledgeBaseError(f"Not a Knowledge Base file: {section}")

        if section.startswith(TABLES_PREFIX):
            return self._kb_path / TABLES_DIRECTORY / section[len(TABLES_PREFIX) :]
        return self._kb_path / section

    def read(self, section: str) -> str:
        """Return the current text of ``section``, or the empty string when new."""
        path = self.resolve(section)
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def current_summary(self) -> dict[str, int]:
        """Entity counts for the Knowledge Base as it currently stands on disk."""
        return load_knowledge_base(self._kb_path).summary()

    def validate(self, section: str, text: str) -> AuthoringResult:
        """Check what ``text`` would do to the Knowledge Base, without writing it.

        Args:
            section: The file the text belongs to, as returned by :meth:`sections`.
            text: The complete replacement contents of that file.
        """
        return self._evaluate(section, text, commit=False)

    def commit(self, section: str, text: str) -> AuthoringResult:
        """Validate ``text`` and, only if it passes, write it to disk."""
        return self._evaluate(section, text, commit=True)

    def delete(self, section: str) -> AuthoringResult:
        """Remove a table file, provided the Knowledge Base still validates without it."""
        if not section.startswith(TABLES_PREFIX):
            return AuthoringResult(
                ok=False,
                message="Only table files can be deleted.",
                detail=(
                    f"{section} is one of the shared sections. To empty it, edit it "
                    "to hold an empty list instead."
                ),
            )

        path = self.resolve(section)
        if not path.is_file():
            return AuthoringResult(ok=False, message=f"{section} does not exist.")

        staged, staged_path = self._stage(section)
        try:
            staged_path.unlink(missing_ok=True)
            outcome = self._load_staged(staged)
            if outcome is not None:
                return AuthoringResult(
                    ok=False,
                    message="Cannot delete: something else still references it.",
                    detail=outcome,
                )

            before = self.current_summary()
            path.unlink()
            after = self.current_summary()
        finally:
            shutil.rmtree(staged.parent, ignore_errors=True)

        logger.info("Knowledge Base file deleted: %s", section)
        return AuthoringResult(
            ok=True,
            message=f"Deleted {section}.",
            summary=after,
            delta={key: after[key] - before[key] for key in after},
        )

    def _evaluate(self, section: str, text: str, *, commit: bool) -> AuthoringResult:
        """Stage ``text``, load the result, and optionally persist it."""
        try:
            path = self.resolve(section)
        except KnowledgeBaseError as exc:
            return AuthoringResult(ok=False, message=str(exc))

        # Parse before staging, so the error points at the text rather than a temp path.
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as exc:
            return AuthoringResult(
                ok=False, message="That is not valid YAML.", detail=str(exc)
            )

        before = self.current_summary()
        staged, staged_path = self._stage(section)
        try:
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_text(text, encoding="utf-8")

            failure = self._load_staged(staged)
            if failure is not None:
                return AuthoringResult(
                    ok=False,
                    message="The Knowledge Base would not load with this change.",
                    detail=failure,
                )

            after = load_knowledge_base(staged).summary()
        finally:
            shutil.rmtree(staged.parent, ignore_errors=True)

        delta = {key: after[key] - before.get(key, 0) for key in after}

        if not commit:
            return AuthoringResult(
                ok=True,
                message="Valid — this change is safe to save.",
                summary=after,
                delta=delta,
            )

        self._write_atomically(path, text)
        logger.info("Knowledge Base updated: %s (%s)", section, _render_delta(delta))
        return AuthoringResult(
            ok=True,
            message=f"Saved {section}.",
            summary=after,
            delta=delta,
        )

    def _stage(self, section: str) -> tuple[Path, Path]:
        """Copy the Knowledge Base to a temporary directory.

        Returns:
            The staged Knowledge Base root, and the staged path of ``section``.
        """
        holder = Path(tempfile.mkdtemp(prefix="nl2sql-kb-"))
        staged = holder / "data"
        shutil.copytree(self._kb_path, staged)

        relative = (
            Path(TABLES_DIRECTORY) / section[len(TABLES_PREFIX) :]
            if section.startswith(TABLES_PREFIX)
            else Path(section)
        )
        return staged, staged / relative

    @staticmethod
    def _load_staged(staged: Path) -> str | None:
        """Load a staged Knowledge Base, returning the error text on failure."""
        try:
            load_knowledge_base(staged)
        except KnowledgeBaseError as exc:
            return str(exc)
        return None

    @staticmethod
    def _write_atomically(path: Path, text: str) -> None:
        """Replace ``path`` with ``text`` in a single filesystem operation."""
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with open(descriptor, "w", encoding="utf-8") as stream:
                stream.write(text)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def _render_delta(delta: dict[str, int]) -> str:
    """Render the non-zero entity count changes as a short phrase."""
    moved = [
        f"{key} {value:+d}" for key, value in sorted(delta.items()) if value
    ]
    return ", ".join(moved) if moved else "no count change"


# --- Building a table from a filled-in form ----------------------------------

COLUMN_ROLES: tuple[str, ...] = (
    "identifier",
    "dimension",
    "measure",
    "timestamp",
    "flag",
    "descriptive",
)

COLUMN_TYPES: tuple[str, ...] = (
    "INTEGER",
    "TEXT",
    "TIMESTAMP",
    "DATE",
    "BOOLEAN",
    "BIGINT",
    "SMALLINT",
)

# Shown in the form so the role choice is not a guess.
ROLE_GUIDANCE = {
    "identifier": "a key — joined on, not grouped by",
    "dimension": "something to group or filter by",
    "measure": "a number to sum, average or count",
    "timestamp": "makes 'in the last 24 hours' work on this table",
    "flag": "a true/false marker",
    "descriptive": "free text, shown but not aggregated",
}


@dataclass(frozen=True, slots=True)
class ColumnDraft:
    """One column as collected from the form."""

    name: str
    data_type: str = "TEXT"
    description: str = ""
    role: str = "dimension"
    primary_key: bool = False
    allowed_values: tuple[str, ...] = ()
    """The complete set of values this column may hold."""


@dataclass(frozen=True, slots=True)
class TableDraft:
    """A table as collected from the form, before it becomes YAML."""

    name: str
    description: str = ""
    business_definition: str = ""
    domain: str = "inventory"
    grain: str = ""
    synonyms: tuple[str, ...] = ()
    columns: tuple[ColumnDraft, ...] = ()
    link_column: str = ""
    link_table: str = ""
    link_to_column: str = ""

    def problems(self) -> list[str]:
        """Return the reasons this draft is not ready, in the order worth fixing."""
        found: list[str] = []

        if not self.name.strip():
            found.append("Give the table a name.")
        elif not self.name.islower() or " " in self.name:
            found.append(
                f"Table names are lower_snake_case — try "
                f"'{_snake_case(self.name)}' instead of '{self.name}'."
            )

        named = [column for column in self.columns if column.name.strip()]
        if not named:
            found.append("Add at least one column.")

        for column in named:
            if column.role not in COLUMN_ROLES:
                found.append(
                    f"Column '{column.name}' has role '{column.role}', which is not "
                    f"one of: {', '.join(COLUMN_ROLES)}."
                )
            if not column.description.strip():
                found.append(
                    f"Describe column '{column.name}' — the retriever searches "
                    "these descriptions, so an undescribed column is hard to find."
                )

        for column in named:
            if (
                column.role.strip().lower() == "dimension"
                and column.data_type.strip().upper() == "TEXT"
                and not column.allowed_values
            ):
                found.append(
                    f"List the allowed values for '{column.name}' (e.g. GOLD, SILVER) "
                    "— without them a filter literal is guessed, which is how a "
                    "correct-looking query returns no rows."
                )

        if named and not any(column.primary_key for column in named):
            found.append("Tick the primary key on exactly one column.")

        if self.link_table and not self.link_column:
            found.append(
                f"Choose which column links to {self.link_table}, or clear the link."
            )
        if self.link_column and not self.link_table:
            found.append(f"Choose the table '{self.link_column}' points at.")
        if self.link_column and self.link_column not in {c.name for c in named}:
            found.append(
                f"The link column '{self.link_column}' is not one of this table's "
                "columns."
            )

        return found

    @property
    def primary_key(self) -> list[str]:
        """The columns marked as the primary key."""
        return [c.name for c in self.columns if c.primary_key and c.name.strip()]

    def to_yaml(self) -> str:
        """Render the draft as the contents of a table file."""
        table: dict[str, Any] = {
            "name": self.name.strip(),
            "description": self.description.strip() or f"Records for {self.name}.",
            "business_definition": (
                self.business_definition.strip()
                or self.description.strip()
                or f"Business records held in {self.name}."
            ),
            "domain": self.domain.strip() or "inventory",
            "grain": self.grain.strip() or f"One row per {self.name} record.",
            "preferred_alias": _alias_for(self.name),
        }
        if self.synonyms:
            table["synonyms"] = list(self.synonyms)
        if self.primary_key:
            table["primary_key"] = self.primary_key

        if self.link_column and self.link_table:
            table["foreign_keys"] = [
                {
                    "column": self.link_column,
                    "references_table": self.link_table,
                    "references_column": self.link_to_column or self.link_column,
                    "description": (
                        f"The {self.link_table} row this record belongs to."
                    ),
                }
            ]

        table["columns"] = [
            {
                "name": column.name.strip(),
                "data_type": column.data_type.strip().upper() or "TEXT",
                "description": (
                    column.description.strip() or f"The {column.name} value."
                ),
                "role": column.role.strip().lower(),
                **(
                    {"allowed_values": list(column.allowed_values)}
                    if column.allowed_values
                    else {}
                ),
                **({"is_primary_key": True, "nullable": False} if column.primary_key
                   else {}),
            }
            for column in self.columns
            if column.name.strip()
        ]

        header = f"# {self.name} — added from the Knowledge Base tab.\n"
        return header + yaml.safe_dump(
            {"tables": [table]}, sort_keys=False, default_flow_style=False, width=88
        )

    def relationship_yaml(self) -> str | None:
        """Render the join this table declares, or ``None`` when it links to nothing."""
        if not (self.link_column and self.link_table):
            return None

        entry = {
            "name": f"{self.name}_to_{self.link_table}",
            "from_table": self.name.strip(),
            "from_column": self.link_column,
            "to_table": self.link_table,
            "to_column": self.link_to_column or self.link_column,
            "cardinality": "many_to_one",
            "join_type": "INNER",
            "description": (
                f"Each {self.name} row belongs to one {self.link_table} row."
            ),
        }
        # A fragment appended to an existing list, so indent it here.
        body = yaml.safe_dump(
            [entry], sort_keys=False, default_flow_style=False, width=88
        )
        return "".join(f"  {line}\n" for line in body.splitlines())


def _snake_case(value: str) -> str:
    """Best-effort conversion of a typed name into lower_snake_case."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()
    return re.sub(r"_+", "_", cleaned) or "my_table"


def _alias_for(table_name: str) -> str:
    """Derive a short alias from the table name, as the registry would."""
    parts = [part for part in _snake_case(table_name).split("_") if part]
    if not parts:
        return "t"
    if len(parts) == 1:
        return parts[0][:3]
    return "".join(part[0] for part in parts)[:4]


# --- Starter templates -------------------------------------------------------

TABLE_TEMPLATE = """\
tables:
  - name: my_new_table
    description: One line on what this table holds.
    business_definition: >
      How the business talks about this table. The retriever searches this text, so
      write the words a user would actually type.
    domain: inventory
    grain: One row per ...
    preferred_alias: mnt
    synonyms: [my table, the new table]
    primary_key: [my_new_table_id]
    # default_filters: ["mnt.is_active = 1"]
    # tags: [reference]
    foreign_keys:
      - column: device_id
        references_table: devices
        references_column: device_id
        description: The device this row belongs to.
    columns:
      - name: my_new_table_id
        data_type: INTEGER
        description: Surrogate key for the row.
        role: identifier
        is_primary_key: true
        nullable: false
      - name: device_id
        data_type: INTEGER
        description: Device this row belongs to.
        role: identifier
        nullable: false
      - name: status
        data_type: TEXT
        description: Current status of the row.
        role: dimension
        allowed_values: [ACTIVE, RETIRED]
        value_synonyms:
          ACTIVE: [live, in use]
          RETIRED: [decommissioned, retired]
      - name: recorded_at
        data_type: TIMESTAMP
        description: When the row was recorded.
        role: timestamp

# Column roles: identifier | dimension | measure | timestamp | flag | descriptive
# A `timestamp` role is what makes "in the last 24 hours" resolvable on this table.
# `allowed_values` is what lets a filter literal be generated without guessing.
"""

RELATIONSHIP_SNIPPET = """\
  - name: my_new_table_to_devices
    from_table: my_new_table
    from_column: device_id
    to_table: devices
    to_column: device_id
    cardinality: many_to_one
    join_type: INNER
    description: Each row belongs to exactly one device.
    business_meaning: Ties the new records to the hardware they describe.
    # traversal_cost: 1.0
"""

RULE_SNIPPET = """\
  - id: my_new_rule
    category: filtering
    rule: >
      State the rule as an instruction the generator can follow directly.
    rationale: Why the rule exists — this is shown to the model as justification.
    applies_to: [my_new_table]
    # example: SELECT ... WHERE ...
"""
