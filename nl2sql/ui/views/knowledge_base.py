"""Browse the Knowledge Base, and extend it without leaving the browser."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nl2sql.knowledge_base.authoring import AuthoringResult, ColumnDraft, TableDraft
from nl2sql.ui import components
from nl2sql.ui.state import Workspace

COLUMN_ROLES = ["dimension", "measure", "identifier", "timestamp", "descriptive", "flag"]

_STARTER_COLUMNS = pd.DataFrame(
    [
        {"Column": "", "Type": "INTEGER", "What it holds": "", "Role": "identifier",
         "Allowed values": "", "Primary key": True},
        {"Column": "", "Type": "VARCHAR(64)", "What it holds": "", "Role": "dimension",
         "Allowed values": "", "Primary key": False},
    ]
)


def _report(result: AuthoringResult) -> None:
    """Show the outcome of an authoring action."""
    if result.ok:
        st.success(result.message)
    else:
        st.error(result.message)
    if result.detail:
        st.caption(result.detail)


def _render_browser(workspace: Workspace) -> None:
    """Show what the agent knows."""
    registry = workspace.registry

    tables_tab, joins_tab, rules_tab, glossary_tab = st.tabs(
        ["Tables", "Relationships", "SQL rules", "Business glossary"]
    )

    with tables_tab:
        st.caption(
            "Every subject the agent can answer about. Ask by name, or combine them — "
            "*failures by site*, *incidents per team*."
        )
        st.dataframe(
            components.schema_dataframe(registry),
            width="stretch",
            hide_index=True,
        )

    with joins_tab:
        st.caption(
            "Foreign keys describe the physical link; these add the intent — why two "
            "tables are joined and how the join must be written."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "From": f"{item.from_table}.{item.from_column}",
                        "To": f"{item.to_table}.{item.to_column}",
                        "Join": item.join_type,
                        "Cardinality": item.cardinality,
                        "Why": item.description,
                    }
                    for item in registry.relationships
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    with rules_tab:
        st.caption(
            "Retrieved into the prompt *and* enforced by the validator, so they are "
            "constraints rather than suggestions."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ID": rule.id,
                        "Category": rule.category,
                        "Applies to": ", ".join(rule.applies_to) or "all tables",
                        "Rule": rule.rule,
                        "Why": rule.rationale or "—",
                    }
                    for rule in registry.knowledge_base.rules
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    with glossary_tab:
        st.caption(
            "Domain wording, and the metrics computable from it. A metric expression "
            "is what lets *success rate* reach `observations` without naming it."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Term": term.term,
                        "Means": term.definition,
                        "Also called": ", ".join(term.synonyms) or "—",
                        "Computable": "yes" if term.is_metric else "no",
                    }
                    for term in registry.knowledge_base.glossary
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def _draft_from_form(
    name: str,
    domain: str,
    description: str,
    business: str,
    grain: str,
    synonyms: str,
    columns: pd.DataFrame,
    link_column: str,
    link_table: str,
    link_target: str,
) -> TableDraft:
    """Turn the form's raw values into a validatable draft."""
    drafts: list[ColumnDraft] = []
    for _, row in columns.iterrows():
        column_name = str(row.get("Column", "") or "").strip()
        if not column_name:
            continue
        drafts.append(
            ColumnDraft(
                name=column_name,
                data_type=str(row.get("Type", "") or "").strip() or "TEXT",
                description=str(row.get("What it holds", "") or "").strip(),
                role=str(row.get("Role", "") or "dimension").strip(),
                allowed_values=tuple(
                    value.strip()
                    for value in str(row.get("Allowed values", "") or "").split(",")
                    if value.strip()
                ),
                primary_key=bool(row.get("Primary key", False)),
            )
        )

    return TableDraft(
        name=name,
        domain=domain,
        description=description,
        business_definition=business,
        grain=grain,
        synonyms=tuple(item.strip() for item in synonyms.split(",") if item.strip()),
        columns=tuple(drafts),
        link_column=link_column,
        link_table=link_table,
        link_to_column=link_target,
    )


def _render_add_table(workspace: Workspace) -> None:
    """The guided form for declaring a new table."""
    st.caption(
        "Describe the table in plain English. The YAML is generated, validated and "
        "only written if it is valid — and the agent picks it up on the next question."
    )

    table_names = ["(none)"] + list(workspace.registry.table_names)

    with st.form("add_table"):
        name_column, domain_column = st.columns(2)
        name = name_column.text_input("Table name", placeholder="capacity_forecasts",
                                      help="lower_snake_case, as in the database.")
        domain = domain_column.text_input("Subject area", placeholder="capacity",
                                          help="Groups it with related tables.")

        description = st.text_input(
            "What does it hold?", placeholder="Projected utilisation per interface."
        )
        business = st.text_area(
            "Business definition",
            placeholder="What one row means to someone who does this job.",
            height=70,
        )

        grain_column, synonyms_column = st.columns(2)
        grain = grain_column.text_input(
            "One row is…", placeholder="one row per interface per forecast window"
        )
        synonyms = synonyms_column.text_input(
            "Also called", placeholder="capacity forecast, utilisation projection",
            help="Comma separated. This is how questions reach the table by other names.",
        )

        st.markdown("**Columns**")
        columns = st.data_editor(
            _STARTER_COLUMNS,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Role": st.column_config.SelectboxColumn(options=COLUMN_ROLES),
                "Primary key": st.column_config.CheckboxColumn(),
            },
            key="new_table_columns",
        )

        st.markdown("**Link it to an existing table** — needed before it can be joined.")
        link_a, link_b, link_c = st.columns(3)
        link_column = link_a.text_input("This table's column", placeholder="interface_id")
        link_table = link_b.selectbox("References table", table_names)
        link_target = link_c.text_input("Referenced column", placeholder="interface_id")

        preview = st.form_submit_button("Preview the YAML")
        submitted = st.form_submit_button("Add this table", type="primary")

    if not (preview or submitted):
        return

    draft = _draft_from_form(
        name, domain, description, business, grain, synonyms, columns,
        link_column, "" if link_table == "(none)" else link_table, link_target,
    )

    problems = draft.problems()
    if problems:
        st.error("Not valid yet:\n\n" + "\n".join(f"- {item}" for item in problems))
        return

    if preview:
        st.code(draft.to_yaml(), language="yaml")
        return

    editor = workspace.editor
    assert editor is not None  # noqa: S101 - set when the workspace is built

    result = editor.commit(f"tables/{draft.name}.yaml", draft.to_yaml())
    _report(result)
    if not result.ok:
        return

    # Written second: a relationship naming an unsaved table leaves the KB unloadable.
    relationship = draft.relationship_yaml()
    if relationship:
        existing = editor.read("relationships.yaml").rstrip()
        _report(editor.commit("relationships.yaml", existing + "\n" + relationship))

    workspace.reload()
    st.cache_resource.clear()
    st.success(f"`{draft.name}` is live — ask a question about it.")


def _render_editor(workspace: Workspace) -> None:
    """Direct YAML editing, with the same validation as the form."""
    editor = workspace.editor
    assert editor is not None  # noqa: S101 - set when the workspace is built

    st.caption(
        "The Knowledge Base as it sits on disk. Nothing is written until it loads "
        "cleanly, so a bad edit cannot take the agent down."
    )

    section = st.selectbox("File", editor.sections())
    text = st.text_area("Contents", value=editor.read(section), height=420)

    check_column, save_column = st.columns(2)
    if check_column.button("Check without saving", width="stretch"):
        _report(editor.validate(section, text))

    if save_column.button("Save", type="primary", width="stretch"):
        result = editor.commit(section, text)
        _report(result)
        if result.ok:
            workspace.reload()
            st.cache_resource.clear()


def render(workspace: Workspace) -> None:
    """Draw the Knowledge Base page."""
    st.header("Knowledge Base")
    st.caption(
        "Everything the agent knows about the database. None of it is compiled into "
        "the code, so **adding a table is an edit here, not a release**."
    )

    components.coverage_chips(workspace.registry, workspace.ready_model_count)
    st.divider()

    browse_tab, add_tab, edit_tab = st.tabs(
        ["Browse", "Add a table", "Edit files directly"]
    )

    with browse_tab:
        _render_browser(workspace)
    with add_tab:
        _render_add_table(workspace)
    with edit_tab:
        _render_editor(workspace)
