"""Tests for editing the Knowledge Base through the authoring API.

The editor is exposed in the web interface, so its safety property is the thing worth
testing hardest: a rejected edit must leave the Knowledge Base on disk exactly as it
was. Every rejection case below asserts that, not just the error message.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from nl2sql.config import DEFAULT_KB_PATH
from nl2sql.exceptions import KnowledgeBaseError
from nl2sql.knowledge_base.authoring import (
    RELATIONSHIP_SNIPPET,
    RULE_SNIPPET,
    TABLE_TEMPLATE,
    ColumnDraft,
    KnowledgeBaseEditor,
    TableDraft,
)
from nl2sql.knowledge_base.loader import load_knowledge_base

NEW_TABLE = "tables/capacity.yaml"


@pytest.fixture
def kb_copy(tmp_path: Path) -> Path:
    """A throwaway copy of the real Knowledge Base, safe to write to."""
    destination = tmp_path / "data"
    shutil.copytree(DEFAULT_KB_PATH, destination)
    return destination


@pytest.fixture
def editor(kb_copy: Path) -> KnowledgeBaseEditor:
    return KnowledgeBaseEditor(kb_copy)


def fingerprint(kb_path: Path) -> dict[str, str]:
    """Map every Knowledge Base file to its contents, for unchanged-ness assertions."""
    return {
        str(path.relative_to(kb_path)): path.read_text(encoding="utf-8")
        for path in sorted(kb_path.rglob("*.yaml"))
    }


class TestSections:
    def test_lists_table_files_and_shared_sections(
        self, editor: KnowledgeBaseEditor
    ) -> None:
        sections = editor.sections()
        assert "tables/observability.yaml" in sections
        assert "relationships.yaml" in sections
        assert "sql_rules.yaml" in sections

    def test_reads_an_existing_file(self, editor: KnowledgeBaseEditor) -> None:
        assert "tables:" in editor.read("tables/observability.yaml")

    def test_a_file_that_does_not_exist_yet_reads_as_empty(
        self, editor: KnowledgeBaseEditor
    ) -> None:
        assert editor.read(NEW_TABLE) == ""


class TestPathSafety:
    """A section name arrives from a browser, so it must not address the filesystem."""

    @pytest.mark.parametrize(
        "section",
        [
            "../../../etc/passwd",
            "tables/../../escape.yaml",
            "/etc/passwd",
            "tables/nested/deep.yaml",
            "not_a_yaml_file.txt",
        ],
    )
    def test_traversal_is_refused(
        self, editor: KnowledgeBaseEditor, section: str
    ) -> None:
        with pytest.raises(KnowledgeBaseError):
            editor.resolve(section)

    def test_a_refused_path_is_never_written(
        self, editor: KnowledgeBaseEditor, kb_copy: Path
    ) -> None:
        before = fingerprint(kb_copy)
        result = editor.commit("../escape.yaml", TABLE_TEMPLATE)

        assert not result.ok
        assert fingerprint(kb_copy) == before


class TestValidation:
    def test_the_starter_template_is_valid(
        self, editor: KnowledgeBaseEditor
    ) -> None:
        result = editor.validate(NEW_TABLE, TABLE_TEMPLATE)

        assert result.ok
        assert result.delta["tables"] == 1
        assert result.changed

    def test_validation_does_not_write(
        self, editor: KnowledgeBaseEditor, kb_copy: Path
    ) -> None:
        before = fingerprint(kb_copy)
        assert editor.validate(NEW_TABLE, TABLE_TEMPLATE).ok
        assert fingerprint(kb_copy) == before

    def test_malformed_yaml_is_rejected(
        self, editor: KnowledgeBaseEditor, kb_copy: Path
    ) -> None:
        before = fingerprint(kb_copy)
        result = editor.commit(NEW_TABLE, "tables: [ unclosed")

        assert not result.ok
        assert "YAML" in result.message
        assert fingerprint(kb_copy) == before

    def test_a_foreign_key_to_an_unknown_table_is_rejected(
        self, editor: KnowledgeBaseEditor, kb_copy: Path
    ) -> None:
        before = fingerprint(kb_copy)
        text = TABLE_TEMPLATE.replace("references_table: devices", "references_table: x")
        result = editor.commit(NEW_TABLE, text)

        assert not result.ok
        assert fingerprint(kb_copy) == before

    def test_an_unknown_field_is_rejected(
        self, editor: KnowledgeBaseEditor, kb_copy: Path
    ) -> None:
        before = fingerprint(kb_copy)
        text = TABLE_TEMPLATE.replace(
            "    domain: inventory", "    domain: inventory\n    nonsense: true"
        )
        result = editor.commit(NEW_TABLE, text)

        assert not result.ok
        assert fingerprint(kb_copy) == before

    def test_breaking_an_existing_file_is_rejected(
        self, editor: KnowledgeBaseEditor, kb_copy: Path
    ) -> None:
        # Emptying the tables file orphans every relationship that references it.
        before = fingerprint(kb_copy)
        result = editor.commit("tables/observability.yaml", "tables: []")

        assert not result.ok
        assert fingerprint(kb_copy) == before


class TestCommit:
    def test_a_new_table_lands_in_the_knowledge_base(
        self, editor: KnowledgeBaseEditor, kb_copy: Path
    ) -> None:
        before = editor.current_summary()
        result = editor.commit(NEW_TABLE, TABLE_TEMPLATE)

        assert result.ok
        after = load_knowledge_base(kb_copy).summary()
        assert after["tables"] == before["tables"] + 1
        assert NEW_TABLE in editor.sections()

    def test_the_new_table_is_queryable_metadata(
        self, editor: KnowledgeBaseEditor, kb_copy: Path
    ) -> None:
        editor.commit(NEW_TABLE, TABLE_TEMPLATE)
        knowledge_base = load_knowledge_base(kb_copy)

        assert "my_new_table" in {table.name for table in knowledge_base.tables}

    def test_a_relationship_snippet_appends_cleanly(
        self, editor: KnowledgeBaseEditor
    ) -> None:
        editor.commit(NEW_TABLE, TABLE_TEMPLATE)
        text = editor.read("relationships.yaml").rstrip() + "\n" + RELATIONSHIP_SNIPPET
        result = editor.commit("relationships.yaml", text)

        assert result.ok, result.detail
        assert result.delta["relationships"] == 1

    def test_a_rule_snippet_appends_cleanly(
        self, editor: KnowledgeBaseEditor
    ) -> None:
        editor.commit(NEW_TABLE, TABLE_TEMPLATE)
        text = editor.read("sql_rules.yaml").rstrip() + "\n" + RULE_SNIPPET
        result = editor.commit("sql_rules.yaml", text)

        assert result.ok, result.detail
        assert result.delta["rules"] == 1


class TestTableDraft:
    """The form path: plain fields in, valid Knowledge Base YAML out."""

    @staticmethod
    def complete() -> TableDraft:
        return TableDraft(
            name="capacity_forecasts",
            description="Projected utilisation for a device.",
            business_definition="Capacity planning output per device.",
            domain="inventory",
            grain="One row per device per horizon.",
            synonyms=("capacity forecast",),
            columns=(
                ColumnDraft("forecast_id", "INTEGER", "Key.", "identifier", True),
                ColumnDraft("device_id", "INTEGER", "The device.", "identifier"),
                ColumnDraft("peak", "INTEGER", "Peak percent.", "measure"),
            ),
            link_column="device_id",
            link_table="devices",
            link_to_column="device_id",
        )

    def test_a_complete_draft_has_no_problems(self) -> None:
        assert self.complete().problems() == []

    def test_the_generated_yaml_validates(self, editor: KnowledgeBaseEditor) -> None:
        result = editor.validate(NEW_TABLE, self.complete().to_yaml())

        assert result.ok, result.detail
        assert result.delta["tables"] == 1
        assert result.delta["columns"] == 3

    def test_the_relationship_appends_and_validates(
        self, editor: KnowledgeBaseEditor
    ) -> None:
        draft = self.complete()
        editor.commit(NEW_TABLE, draft.to_yaml())

        fragment = draft.relationship_yaml()
        assert fragment is not None
        text = editor.read("relationships.yaml").rstrip() + "\n" + fragment
        result = editor.commit("relationships.yaml", text)

        assert result.ok, result.detail
        assert result.delta["relationships"] == 1

    def test_no_link_means_no_relationship(self) -> None:
        draft = TableDraft(name="standalone", columns=(ColumnDraft("id"),))
        assert draft.relationship_yaml() is None

    def test_a_description_containing_yaml_syntax_survives(
        self, editor: KnowledgeBaseEditor
    ) -> None:
        # Templating this by hand would produce a file that no longer parses.
        hostile = 'a: colon, a "quote", a #hash and a - dash'
        draft = TableDraft(
            name="tricky",
            description=hostile,
            business_definition=hostile,
            columns=(ColumnDraft("tricky_id", "INTEGER", hostile, "identifier", True),),
        )
        result = editor.validate("tables/tricky.yaml", draft.to_yaml())

        assert result.ok, result.detail

    @pytest.mark.parametrize(
        ("draft", "expected"),
        [
            (TableDraft(name=""), "name"),
            (TableDraft(name="Capacity Forecasts"), "lower_snake_case"),
            (TableDraft(name="ok", columns=()), "column"),
            (
                TableDraft(name="ok", columns=(ColumnDraft("id", description="x"),)),
                "primary key",
            ),
            (
                TableDraft(
                    name="ok",
                    columns=(ColumnDraft("id", description="x", primary_key=True),),
                    link_table="devices",
                ),
                "which column",
            ),
        ],
    )
    def test_problems_are_reported_in_plain_english(
        self, draft: TableDraft, expected: str
    ) -> None:
        assert any(expected in problem for problem in draft.problems())

    def test_an_undescribed_column_is_flagged(self) -> None:
        draft = TableDraft(
            name="ok", columns=(ColumnDraft("id", primary_key=True),)
        )
        assert any("Describe column" in problem for problem in draft.problems())

    def test_an_unknown_role_is_flagged(self) -> None:
        draft = TableDraft(
            name="ok",
            columns=(
                ColumnDraft("id", description="x", role="nonsense", primary_key=True),
            ),
        )
        assert any("nonsense" in problem for problem in draft.problems())


class TestDelete:
    def test_a_table_file_can_be_removed(
        self, editor: KnowledgeBaseEditor, kb_copy: Path
    ) -> None:
        editor.commit(NEW_TABLE, TABLE_TEMPLATE)
        assert editor.delete(NEW_TABLE).ok

        assert not (kb_copy / "tables" / "capacity.yaml").exists()
        remaining = {table.name for table in load_knowledge_base(kb_copy).tables}
        assert "my_new_table" not in remaining

    def test_a_file_other_tables_depend_on_cannot_be_removed(
        self, editor: KnowledgeBaseEditor, kb_copy: Path
    ) -> None:
        before = fingerprint(kb_copy)
        result = editor.delete("tables/observability.yaml")

        assert not result.ok
        assert fingerprint(kb_copy) == before

    def test_shared_sections_are_not_deletable(
        self, editor: KnowledgeBaseEditor
    ) -> None:
        result = editor.delete("relationships.yaml")

        assert not result.ok
        assert "table files" in result.message.lower()
