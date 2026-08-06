"""Loading the Knowledge Base from YAML.

Expected layout of the Knowledge Base directory::

    data/
      tables/*.yaml          -> {"tables": [...]}   (any number of files)
      relationships.yaml     -> {"relationships": [...]}
      sql_rules.yaml         -> {"rules": [...]}
      business_glossary.yaml -> {"glossary": [...]}
      example_queries.yaml   -> {"examples": [...]}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from nl2sql.exceptions import KnowledgeBaseError
from nl2sql.knowledge_base.models import KnowledgeBase
from nl2sql.logging_config import get_logger

logger = get_logger(__name__)

TABLES_DIRECTORY = "tables"

_SINGLE_FILE_SECTIONS: dict[str, str] = {
    "relationships": "relationships.yaml",
    "rules": "sql_rules.yaml",
    "glossary": "business_glossary.yaml",
    "examples": "example_queries.yaml",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML document, returning an empty mapping for an empty file."""
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise KnowledgeBaseError(f"Malformed YAML in {path}: {exc}") from exc
    except OSError as exc:
        raise KnowledgeBaseError(f"Could not read {path}: {exc}") from exc

    if content is None:
        return {}
    if not isinstance(content, dict):
        raise KnowledgeBaseError(
            f"{path} must contain a YAML mapping at the top level, "
            f"got {type(content).__name__}"
        )
    return content


def _load_table_definitions(tables_directory: Path) -> list[dict[str, Any]]:
    """Merge every ``tables`` list found under the tables directory."""
    if not tables_directory.is_dir():
        raise KnowledgeBaseError(
            f"Knowledge Base is missing its '{TABLES_DIRECTORY}' directory: "
            f"{tables_directory}"
        )

    yaml_files = sorted(
        path
        for path in tables_directory.iterdir()
        if path.suffix in {".yaml", ".yml"} and path.is_file()
    )
    if not yaml_files:
        raise KnowledgeBaseError(f"No table definitions found in {tables_directory}")

    definitions: list[dict[str, Any]] = []
    for path in yaml_files:
        document = _read_yaml(path)
        tables = document.get("tables", [])
        if not isinstance(tables, list):
            raise KnowledgeBaseError(f"'tables' in {path} must be a list")
        logger.debug("Loaded %d table definition(s) from %s", len(tables), path.name)
        definitions.extend(tables)

    return definitions


def _load_section(kb_path: Path, key: str, filename: str) -> list[dict[str, Any]]:
    """Load one optional single-file section such as ``rules``."""
    path = kb_path / filename
    if not path.is_file():
        logger.warning("Optional Knowledge Base file not found, skipping: %s", filename)
        return []

    document = _read_yaml(path)
    entries = document.get(key, [])
    if not isinstance(entries, list):
        raise KnowledgeBaseError(f"'{key}' in {path} must be a list")
    logger.debug("Loaded %d %s entr(ies) from %s", len(entries), key, filename)
    return entries


def load_knowledge_base(kb_path: Path) -> KnowledgeBase:
    """Load and validate the complete Knowledge Base from ``kb_path``.

    Raises:
        KnowledgeBaseError: if any file is missing, malformed, or references an
            entity that does not exist.
    """
    kb_path = Path(kb_path).expanduser().resolve()
    if not kb_path.is_dir():
        raise KnowledgeBaseError(f"Knowledge Base directory not found: {kb_path}")

    payload: dict[str, Any] = {
        "tables": _load_table_definitions(kb_path / TABLES_DIRECTORY)
    }
    for key, filename in _SINGLE_FILE_SECTIONS.items():
        payload[key] = _load_section(kb_path, key, filename)

    try:
        knowledge_base = KnowledgeBase.model_validate(payload)
    except PydanticValidationError as exc:
        raise KnowledgeBaseError(
            f"Knowledge Base failed validation:\n{exc}"
        ) from exc
    except ValueError as exc:
        # Raised by the cross-reference model validators.
        raise KnowledgeBaseError(f"Knowledge Base failed validation: {exc}") from exc

    logger.info("Knowledge Base loaded: %s", knowledge_base.summary())
    return knowledge_base
