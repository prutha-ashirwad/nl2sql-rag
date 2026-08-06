"""The LangGraph workflow, step by step, for the most recent question."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nl2sql.graph.state import TraceEvent
from nl2sql.pipeline import NL2SQLAnswer
from nl2sql.ui import components, workflow_diagram
from nl2sql.ui.state import Workspace

# Ordered as the graph runs them.
NODE_GUIDE: list[dict[str, str]] = [
    {
        "Node": "analyze",
        "What it does": "Classifies the question and resolves its wording onto tables, "
        "columns and declared values.",
        "Then what": "→ retrieve if the question is answerable, otherwise → finalize.",
    },
    {
        "Node": "retrieve",
        "What it does": "RAG step. Ranks Knowledge Base documents by BM25, TF-IDF "
        "vectors and glossary synonyms, fuses the three, then expands along declared "
        "join paths.",
        "Then what": "→ generate if any schema matched, otherwise → finalize.",
    },
    {
        "Node": "generate",
        "What it does": "Writes SQL from the retrieved context — via the model, or via "
        "the deterministic Knowledge Base planner.",
        "Then what": "→ validate if a query came back, otherwise → finalize.",
    },
    {
        "Node": "validate",
        "What it does": "Parses the SQL and checks it against the Knowledge Base: real "
        "tables, real columns, bound aliases, read-only, row limit.",
        "Then what": "→ execute if valid, → repair if not, → finalize when execution "
        "is off or the repair budget is spent.",
    },
    {
        "Node": "repair",
        "What it does": "Feeds the validation errors back for another attempt. The only "
        "cycle in the graph, and bounded by MAX_REPAIR_ATTEMPTS.",
        "Then what": "→ validate, so a rewrite is never trusted without re-checking.",
    },
    {
        "Node": "execute",
        "What it does": "Runs the query read-only, with a row cap and a timeout.",
        "Then what": "→ finalize.",
    },
    {
        "Node": "finalize",
        "What it does": "Composes the answer, including when an earlier node bailed out.",
        "Then what": "→ END. Every path lands here, so a caller always gets a formed "
        "response.",
    },
]


def _render_workflow(trace: list[TraceEvent] | None) -> None:
    """Draw the graph, highlighting the path this run took."""
    visited = {event.node for event in trace} if trace else None

    st.markdown(workflow_diagram.render(visited), unsafe_allow_html=True)
    if visited is None:
        st.caption(
            "The compiled LangGraph state machine. Ask a question and this diagram "
            "highlights the path that question actually took."
        )
    else:
        skipped = [
            node for node in workflow_diagram.NODES if node not in visited
        ]
        st.caption(
            "Highlighted nodes are the ones this run entered"
            + (
                f"; {', '.join(name.split('_')[0] for name in skipped)} "
                "did not run."
                if skipped
                else " — every node in the graph."
            )
        )

    with st.expander("What each node does, and what makes it branch"):
        st.markdown(
            "\n\n".join(
                f"**`{entry['Node']}`** — {entry['What it does']}  \n"
                f"<span style='color:#94A3B8'>{entry['Then what']}</span>"
                for entry in NODE_GUIDE
            ),
            unsafe_allow_html=True,
        )


def _render_retrieved_context(answer: NL2SQLAnswer) -> None:
    """Show what the RAG step pulled out of the Knowledge Base, and how it scored."""
    documents = answer.retrieved_documents

    if not documents:
        st.info("No Knowledge Base documents were retrieved for this question.")
        return

    st.caption(
        "Ranked by fused relevance across three signals — keyword, vector and the "
        "curated glossary. This is the context the SQL was written from."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Rank": position,
                    "Document": document["id"],
                    "Kind": document["kind"],
                    "Score": round(document["score"], 4),
                    "Signals": document["retriever"],
                }
                for position, document in enumerate(documents, start=1)
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def render(workspace: Workspace) -> None:
    """Draw the Agent Execution page."""
    st.header("Agent execution")
    st.caption(
        "Every question runs through a declared LangGraph state machine. Each node "
        "records what it decided, so an answer can be audited rather than trusted."
    )

    record = workspace.last_run
    _render_workflow(record.answer.trace if record is not None else None)

    if record is None:
        st.info("Ask a question first — this page shows how the last one was answered.")
        return

    answer = record.answer

    st.divider()
    st.subheader("The question just asked")
    st.markdown(f"> {answer.question}")

    columns = st.columns(4)
    columns[0].metric("Engine", record.engine.split(" ")[0])
    columns[1].metric("Steps run", len(answer.trace))
    columns[2].metric("Total time", f"{record.elapsed_seconds:.2f}s")
    columns[3].metric(
        "Tokens used",
        f"{answer.tokens_used:,}" if answer.tokens_used else "—",
        help="Model tokens. The Knowledge Base planner uses none.",
    )

    st.divider()
    step_tab, context_tab, validation_tab, sql_tab = st.tabs(
        ["Steps", "Retrieved context", "Validation", "SQL"]
    )

    with step_tab:
        st.markdown("#### Where the time went")
        components.render_step_timeline(answer.trace)
        st.markdown("#### What each step decided")
        components.render_step_details(answer.trace)

    with context_tab:
        _render_retrieved_context(answer)

    with validation_tab:
        st.caption(
            "Generated SQL is checked against the same Knowledge Base that produced "
            "the prompt, which is what makes the repair loop converge rather than guess."
        )
        components.render_validation(answer)
        if answer.repair_attempts:
            st.info(
                f"The query was rewritten {answer.repair_attempts} time(s) after "
                "failing validation."
            )

    with sql_tab:
        components.render_sql(answer)
        if answer.explanation:
            st.markdown("**How the query was built**")
            st.write(answer.explanation)
