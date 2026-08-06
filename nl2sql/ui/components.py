"""Shared renderers used by more than one page."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from nl2sql.graph.state import TraceEvent
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.pipeline import NL2SQLAnswer

MAX_DISPLAY_ROWS = 200

STEP_TITLES: dict[str, str] = {
    "analyze_question": "Understood the question",
    "retrieve_context": "Retrieved Knowledge Base context",
    "generate_sql": "Generated SQL",
    "validate_sql": "Validated against the Knowledge Base",
    "repair_sql": "Repaired the query",
    "execute_sql": "Ran the query",
    "finalize": "Composed the answer",
}

STEP_ICONS: dict[str, str] = {
    "analyze_question": "🧭",
    "retrieve_context": "📚",
    "generate_sql": "✍️",
    "validate_sql": "🛡️",
    "repair_sql": "🔧",
    "execute_sql": "⚡",
    "finalize": "✅",
}


def step_title(node: str) -> str:
    """Plain-language name for a workflow node."""
    return STEP_TITLES.get(node, node.replace("_", " ").capitalize())


def step_icon(node: str) -> str:
    """Icon for a workflow node."""
    return STEP_ICONS.get(node, "•")


def render_outcome(answer: NL2SQLAnswer) -> None:
    """Show whether the question was answered, and how it went."""
    if not answer.succeeded:
        reason = answer.errors[-1] if answer.errors else answer.answer
        st.error(f"**No answer for that one** — {reason}")
        return

    if answer.row_count == 0:
        st.warning(
            "**Nothing matched.** The query ran and returned no rows. "
            "Try a wider time range or a different filter."
        )
        return

    plural = "" if answer.row_count == 1 else "s"

    # Queries carry a LIMIT only when the question asked for one, so the row cap is
    # what bounds everything else. Saying so is the difference between a partial
    # answer and one that reads as complete.
    if answer.truncated:
        st.warning(
            f"**Showing the first {answer.row_count} row{plural}** — more matched. "
            "Narrow the question, or ask for a specific number such as *top 20*."
        )
        return

    st.success(f"**{answer.row_count} result{plural}** — answered from live data.")


def render_metrics(answer: NL2SQLAnswer, elapsed_seconds: float) -> None:
    """Headline numbers for one run."""
    columns = st.columns(4)
    columns[0].metric("Rows returned", f"{answer.row_count:,}")
    columns[1].metric("Time taken", f"{elapsed_seconds:.1f}s")
    columns[2].metric("Tables read", len(answer.tables_used))
    columns[3].metric(
        "Repair attempts",
        answer.repair_attempts,
        help="How many times the query was rewritten after failing validation.",
    )

    if answer.tables_used:
        st.caption("Read from: " + ", ".join(f"`{name}`" for name in answer.tables_used))


def render_sql(answer: NL2SQLAnswer) -> None:
    """Show the generated query."""
    if not answer.sql:
        st.info("No query was generated for this question.")
        return
    st.code(answer.sql, language="sql")


def results_dataframe(answer: NL2SQLAnswer) -> pd.DataFrame:
    """Build a display frame with human-readable column names."""
    if not answer.rows:
        return pd.DataFrame()

    frame = pd.DataFrame(answer.rows[:MAX_DISPLAY_ROWS])
    frame.columns = [str(name).replace("_", " ").title() for name in frame.columns]
    return frame


def render_results(answer: NL2SQLAnswer) -> None:
    """Show the result rows, with the full set offered as a download."""
    frame = results_dataframe(answer)
    if frame.empty:
        return

    st.dataframe(frame, width="stretch", hide_index=True)

    if answer.row_count > MAX_DISPLAY_ROWS:
        st.caption(
            f"Showing the first {MAX_DISPLAY_ROWS:,} of {answer.row_count:,} rows — "
            "download for the rest."
        )

    st.download_button(
        "Download all results (CSV)",
        data=pd.DataFrame(answer.rows).to_csv(index=False).encode("utf-8"),
        file_name="nl2sql-results.csv",
        mime="text/csv",
    )


def _format_detail(value: Any) -> str:
    """Render one trace detail compactly enough to sit in a table."""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "—"
    if value is None:
        return "—"
    return str(value)


def render_step_timeline(trace: list[TraceEvent]) -> None:
    """Draw the workflow as an ordered list of steps with their timings."""
    if not trace:
        st.info("No steps were recorded for this question.")
        return

    total = sum(event.duration_ms for event in trace) or 1.0

    for event in trace:
        share = event.duration_ms / total
        left, right = st.columns([5, 1])
        left.markdown(
            f"{step_icon(event.node)} **{step_title(event.node)}** — {event.summary}"
        )
        right.markdown(f"`{event.duration_ms:7.1f} ms`")
        st.progress(min(share, 1.0))


def render_step_details(trace: list[TraceEvent]) -> None:
    """Draw every step with the details the node recorded."""
    if not trace:
        st.info("Ask a question to see the workflow run.")
        return

    for position, event in enumerate(trace, start=1):
        with st.expander(
            f"{position}. {step_icon(event.node)} {step_title(event.node)} "
            f"— {event.duration_ms:.1f} ms",
            expanded=position <= 2,
        ):
            st.markdown(f"**{event.summary}**")
            st.caption(f"Workflow node: `{event.node}`")

            if not event.details:
                continue

            st.dataframe(
                pd.DataFrame(
                    [
                        {"Detail": key.replace("_", " ").title(),
                         "Value": _format_detail(value)}
                        for key, value in event.details.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )


def render_validation(answer: NL2SQLAnswer) -> None:
    """Show what validation found, if anything."""
    if answer.validation_errors:
        st.error("**Validation errors**\n\n" + "\n".join(
            f"- {message}" for message in answer.validation_errors
        ))
    if answer.validation_warnings:
        st.warning("**Validation warnings**\n\n" + "\n".join(
            f"- {message}" for message in answer.validation_warnings
        ))
    if not answer.validation_errors and not answer.validation_warnings and answer.sql:
        st.success("Passed every check: read-only, real tables and columns, "
                   "bound aliases, row limit applied.")


def schema_dataframe(registry: KnowledgeBaseRegistry) -> pd.DataFrame:
    """One row per declared table, in plain language."""
    return pd.DataFrame(
        [
            {
                "Subject": table.name.replace("_", " ").title(),
                "Table": table.name,
                "Area": table.domain.replace("_", " ").title(),
                "What it holds": table.description,
                "One row is": table.grain,
                "Columns": len(table.columns),
            }
            for table in registry.tables
        ]
    )


def coverage_chips(registry: KnowledgeBaseRegistry, ready_models: int) -> None:
    """Knowledge Base coverage as a row of metrics."""
    summary = registry.knowledge_base.summary()
    columns = st.columns(6)
    columns[0].metric("Tables", summary["tables"])
    columns[1].metric("Columns", summary["columns"])
    columns[2].metric("Relationships", summary["relationships"])
    columns[3].metric("SQL rules", summary["rules"])
    columns[4].metric("Examples", summary["example_queries"])
    columns[5].metric("Models ready", ready_models)
