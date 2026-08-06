"""Streamlit entry point: ``streamlit run nl2sql/ui/app.py``."""

from __future__ import annotations

import streamlit as st

from nl2sql.logging_config import configure_logging
from nl2sql.tracing import configure_tracing, is_tracing_active
from nl2sql.ui import clipboard, theme
from nl2sql.ui.state import PLANNER_LABEL, Workspace, get_workspace
from nl2sql.ui.views import agent_execution, ask, compare, database, knowledge_base

PAGES = {
    "Ask": ask.render,
    "Agent Execution": agent_execution.render,
    "Knowledge Base": knowledge_base.render,
    "Compare Engines": compare.render,
    "Database": database.render,
}

PAGE_ICONS = {
    "Ask": "💬",
    "Agent Execution": "🔍",
    "Knowledge Base": "📚",
    "Compare Engines": "⚖️",
    "Database": "🗄️",
}


def _render_sidebar(workspace: Workspace) -> str:
    """Draw the sidebar and return the page to show."""
    with st.sidebar:
        st.title("NL2SQL")
        st.caption("LangGraph · RAG · Knowledge Base")

        page = st.radio(
            "Page",
            list(PAGES),
            format_func=lambda name: f"{PAGE_ICONS[name]}  {name}",
            label_visibility="collapsed",
        )

        st.divider()
        st.subheader("Query writer")
        labels = list(workspace.engines)
        available = workspace.available
        default = workspace.default_engine()

        engine = st.selectbox(
            "Engine",
            labels,
            index=labels.index(default),
            label_visibility="collapsed",
            help=(
                "Which engine writes the SQL. Models whose key is missing are listed "
                "with the variable to set, so what is absent is visible."
            ),
        )
        # Unavailable models stay in the picker but must not be dispatched to.
        st.session_state["engine"] = engine if engine in available else default
        if engine not in available:
            st.warning("That model has no key configured — using the planner instead.")

        st.divider()
        st.subheader("Retrieval")
        settings = workspace.settings
        st.caption("**BM25** keywords · **dense** vectors · **glossary** synonyms")
        st.caption(
            f"Fused by RRF · top {settings.retrieval_top_k} documents · "
            f"lexical weight {settings.lexical_weight}"
        )
        if settings.retrieval_expand_joins:
            st.caption("Join paths expanded from the Knowledge Base.")

        # Read-only: the backend is chosen at start-up, so showing what is actually
        # loaded is the only honest reading — a missing key silently means TF-IDF.
        st.caption(f"Dense backend: `{workspace.describe_retrieval()}`")
        st.caption(
            "Change it with `EMBEDDING_PROVIDER` (tfidf · openai) and "
            "`VECTOR_STORE` (memory · faiss) in `.env`, then restart."
        )

        st.divider()
        st.subheader("Environment")
        st.caption(f"Database: `{workspace.describe_database()}`")
        st.caption(f"Dialect: `{settings.sql_dialect}`")
        st.caption(
            "LangSmith tracing: " + ("on" if is_tracing_active() else "off")
        )
        if workspace.ready_model_count == 0:
            st.info(
                "No model key configured — every answer comes from the Knowledge Base "
                "planner. That is a supported mode, not a broken one."
            )

    return page


def main() -> None:
    """Configure the page and run the selected view."""
    st.set_page_config(
        page_title="Intelligent NL2SQL",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.apply()
    clipboard.install()

    settings_configured = st.session_state.get("_configured", False)
    if not settings_configured:
        # Once per process: reconfiguring on every rerun would stack log handlers.
        configure_logging("INFO")
        configure_tracing(get_workspace().settings)
        st.session_state["_configured"] = True

    workspace = get_workspace()
    page = _render_sidebar(workspace)

    st.session_state.setdefault("engine", PLANNER_LABEL)
    PAGES[page](workspace)


main()
