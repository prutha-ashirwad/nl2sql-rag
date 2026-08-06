"""Visual polish applied on top of the Streamlit theme.

The palette itself lives in ``.streamlit/config.toml``; Streamlit reads it before the
first widget is drawn, so it cannot be set from Python.
"""

from __future__ import annotations

import streamlit as st

_STYLES = """
<style>
:root {
    --nl2sql-border: rgba(148, 163, 184, 0.16);
    --nl2sql-surface: #141922;
    --nl2sql-accent: #6D8CFF;
    --nl2sql-muted: #94A3B8;
}

/* Streamlit reserves room for a toolbar that is minimised here. */
[data-testid="stAppViewBlockContainer"] {
    padding-top: 2.6rem;
    max-width: 1400px;
}

/* --- Sidebar ---------------------------------------------------------- */

[data-testid="stSidebar"] {
    border-right: 1px solid var(--nl2sql-border);
}
[data-testid="stSidebar"] h1 {
    font-size: 1.45rem;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}
[data-testid="stSidebar"] h3 {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--nl2sql-muted);
}
[data-testid="stSidebar"] hr {
    margin: 1.1rem 0;
    border-color: var(--nl2sql-border);
}
/* The page picker reads as navigation, not as a form control. */
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.28rem 0.5rem;
    border-radius: 7px;
    transition: background 120ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(109, 140, 255, 0.10);
}

/* --- Headings --------------------------------------------------------- */

h1, h2, h3 {
    letter-spacing: -0.021em;
}

/* --- Buttons ---------------------------------------------------------- */

.stButton button {
    border-radius: 9px;
    border: 1px solid var(--nl2sql-border);
    font-weight: 500;
    transition: border-color 130ms ease, background 130ms ease, transform 130ms ease;
}
.stButton button:hover {
    border-color: var(--nl2sql-accent);
    transform: translateY(-1px);
}
.stButton button[kind="primary"] {
    border-color: transparent;
    box-shadow: 0 1px 14px rgba(109, 140, 255, 0.28);
    padding-left: 2.1rem;
    padding-right: 2.1rem;
    font-weight: 600;
}
/* Example questions are suggestions; they should not compete with Ask. */
.stButton button[kind="secondary"] {
    background: var(--nl2sql-surface);
    color: #CBD5E1;
    font-size: 0.87rem;
    text-align: left;
    min-height: 3.1rem;
}

/* --- Metrics ---------------------------------------------------------- */

[data-testid="stMetric"] {
    background: var(--nl2sql-surface);
    border: 1px solid var(--nl2sql-border);
    border-radius: 11px;
    padding: 0.85rem 1rem;
}
/* Uppercasing widens the label, so it has to be allowed to wrap — Streamlit
   otherwise clips it to an ellipsis and "Relationships" reads as "Relationshi…". */
[data-testid="stMetricLabel"] {
    overflow: visible;
    white-space: normal;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    white-space: normal;
    overflow-wrap: anywhere;
    color: var(--nl2sql-muted);
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-variant-numeric: tabular-nums;
}

/* --- SQL and code ----------------------------------------------------- */

[data-testid="stCode"] {
    border: 1px solid var(--nl2sql-border);
    border-radius: 10px;
}
[data-testid="stCode"] pre {
    background: #0D1118 !important;
    font-size: 0.855rem;
    line-height: 1.62;
}

/* --- Tabs, tables, expanders ------------------------------------------ */

[data-testid="stTabs"] [role="tablist"] {
    gap: 1.4rem;
    border-bottom: 1px solid var(--nl2sql-border);
}
[data-testid="stTabs"] [role="tab"] {
    font-weight: 500;
}
[data-testid="stExpander"] details {
    border: 1px solid var(--nl2sql-border);
    border-radius: 10px;
    background: rgba(20, 25, 34, 0.55);
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--nl2sql-border);
    border-radius: 10px;
}

/* --- Inputs ----------------------------------------------------------- */

[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    border-radius: 9px;
    font-size: 0.95rem;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--nl2sql-accent);
}

/* Captions carry most of the explanatory text, so they need to stay legible. */
[data-testid="stCaptionContainer"] p {
    color: var(--nl2sql-muted);
    line-height: 1.55;
}
</style>
"""


def apply() -> None:
    """Inject the stylesheet. Safe to call on every rerun."""
    st.markdown(_STYLES, unsafe_allow_html=True)
