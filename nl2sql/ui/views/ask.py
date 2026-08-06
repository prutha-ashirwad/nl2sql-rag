"""Ask a question and see the answer, the SQL, and how it was reached."""

from __future__ import annotations

import streamlit as st

from nl2sql.exceptions import NL2SQLError
from nl2sql.ui import components
from nl2sql.ui.state import Workspace

EXAMPLE_GROUPS: dict[str, list[str]] = {
    "Look up records": [
        "Show all failed observations in the last 24 hours.",
        "Show open critical alerts that nobody has acknowledged.",
    ],
    "Rank and compare": [
        "List interfaces with the highest failure count.",
        "Which devices had the most failures in production?",
        "Show the top 5 sites by failure count in the last week.",
    ],
    "Summarise": [
        "Display observations grouped by environment.",
        "Break down failures by failure reason.",
        "What is the success rate per environment over the last 7 days?",
        "How many incidents does each team own?",
    ],
}

# Keyed rather than given a `value=`, which a rerun would reset on every interaction.
QUESTION_KEY = "question"

# Set when an example button asks for the question to be run, not merely loaded.
SUBMIT_KEY = "submit_question"


def _load_example(question: str) -> None:
    """Put ``question`` in the box and queue it to run.

    Must stay a callback: assigning a widget's own key from the script body once
    that widget exists raises ``StreamlitAPIException``.
    """
    st.session_state[QUESTION_KEY] = question
    st.session_state[SUBMIT_KEY] = True


def _render_examples() -> None:
    """Offer worked examples; clicking one asks it straight away."""
    st.caption("Or start from an example:")

    for heading, questions in EXAMPLE_GROUPS.items():
        st.markdown(f"**{heading}**")
        columns = st.columns(len(questions))
        for column, question in zip(columns, questions, strict=True):
            column.button(
                question,
                key=f"example::{question}",
                width="stretch",
                on_click=_load_example,
                args=(question,),
            )


def render(workspace: Workspace) -> None:
    """Draw the Ask page."""
    st.header("Ask the database")
    st.caption(
        "Ask in plain English. The agent retrieves the relevant schema from the "
        "Knowledge Base, writes the SQL, checks it, then runs it."
    )

    st.session_state.setdefault(QUESTION_KEY, "")

    question = st.text_area(
        "Your question",
        key=QUESTION_KEY,
        placeholder="e.g. Show all failed observations in the last 24 hours.",
        height=90,
    )

    asked = st.button("Ask", type="primary")
    _render_examples()

    # Queued rather than answered in the callback: callbacks run before the page draws.
    asked = asked or st.session_state.pop(SUBMIT_KEY, False)

    if not asked:
        if workspace.last_run is not None:
            st.divider()
            st.caption("Showing the most recent answer.")
            _render_run(workspace)
        return

    if not question.strip():
        st.warning("Type a question first.")
        return

    engine = st.session_state.get("engine", workspace.default_engine())
    try:
        with st.spinner(f"Asking {engine}…"):
            workspace.ask(question, engine)
    except NL2SQLError as exc:
        st.error(f"Could not answer that: {exc}")
        return

    st.divider()
    _render_run(workspace)


def _render_run(workspace: Workspace) -> None:
    """Render the most recent run."""
    record = workspace.last_run
    if record is None:
        return

    answer = record.answer

    components.render_outcome(answer)
    components.render_metrics(answer, record.elapsed_seconds)
    components.render_results(answer)

    st.subheader("Generated SQL")
    components.render_sql(answer)

    st.subheader("Agent execution")
    st.caption(
        f"Answered by **{record.engine}** in {len(answer.trace)} workflow steps. "
        "Open the Agent Execution page for the full detail of each one."
    )
    components.render_step_timeline(answer.trace)
