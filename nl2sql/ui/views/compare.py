"""Run one question through every engine and put the answers side by side."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from nl2sql.exceptions import NL2SQLError
from nl2sql.pipeline import NL2SQLAnswer
from nl2sql.ui.state import PLANNER_LABEL, Workspace


def render(workspace: Workspace) -> None:
    """Draw the Compare engines page."""
    st.header("Compare engines")

    engines = workspace.available
    st.caption(
        f"Runs one question through all {len(engines)} available engines — including "
        "the zero-cost Knowledge Base planner — and shows what each produced."
    )

    default_question = (
        workspace.last_run.answer.question if workspace.last_run else
        "List interfaces with the highest failure count."
    )
    question = st.text_input("Question", value=default_question)

    if not st.button("Run the comparison", type="primary"):
        return

    if not question.strip():
        st.warning("Type a question first.")
        return

    results: list[tuple[str, NL2SQLAnswer, float]] = []
    progress = st.progress(0.0, text="Starting…")

    for position, (label, engine) in enumerate(engines.items(), start=1):
        progress.progress(position / len(engines), text=f"Asking {label}…")
        started = time.perf_counter()
        try:
            # Tagged so one comparison stays reconstructable in LangSmith afterwards.
            answer = engine.answer(
                question.strip(), tags=["comparison", f"engine:{label}"]
            )
        except NL2SQLError as exc:
            st.error(f"{label} failed: {exc}")
            continue
        results.append((label, answer, time.perf_counter() - started))

    progress.empty()

    if not results:
        st.error("No engine produced an answer.")
        return

    st.subheader("Side by side")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Engine": label,
                    "Answered": "yes" if answer.succeeded else "no",
                    "Rows": answer.row_count,
                    "Seconds": round(seconds, 2),
                    "Tokens": answer.tokens_used or 0,
                    "Tables read": ", ".join(answer.tables_used) or "—",
                }
                for label, answer, seconds in results
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("What each one wrote")
    for label, answer, _ in results:
        with st.expander(label, expanded=label == PLANNER_LABEL):
            st.code(answer.sql or "-- no query produced", language="sql")
