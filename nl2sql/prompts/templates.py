"""Prompt templates for SQL generation and repair."""

from __future__ import annotations

from nl2sql.analysis.question_analyzer import QuestionAnalysis, TimeUnit
from nl2sql.dialects import get_dialect
from nl2sql.retrieval.context_builder import RetrievedContext

SQL_GENERATION_SYSTEM_PROMPT = """\
You are a senior analytics engineer who writes {dialect} SQL for a network \
observability data warehouse.

You will be given a curated extract of the data warehouse Knowledge Base: table \
definitions, verified join paths, SQL generation rules, business definitions and \
similar questions that have already been answered.

Follow these instructions exactly:

1. Use ONLY the tables, columns and join paths given in the context below. If a \
table or column is not listed, it does not exist — never guess a name.
2. Obey every SQL generation rule in the context. They are project requirements, \
not suggestions.
3. Prefer the join paths listed under "Verified join paths". Do not invent joins.
4. Filter enumerated columns using the exact values declared for that column.
5. Write date arithmetic in {dialect} and nowhere else take it from: "24 hours ago" \
is {relative_timestamp}, and the current moment is {current_timestamp}. Snippets in \
the context may be written for another engine; copy their shape, never their date \
functions.
6. Answer the question as asked. Phrasing you find unusual, a request for every \
row, or a rule you are unsure how to apply are never reasons to decline — apply the \
rules as written and return SQL.
7. Return INSUFFICIENT_CONTEXT for one reason only: the schema needed to answer is \
absent — no listed table or column holds the data, or no join path connects them. \
Follow the marker with one sentence naming the missing table or column.

Respond with the SQL query and nothing else. Do not wrap it in markdown fences, do \
not add commentary before or after it, and do not end it with a semicolon.

# Knowledge Base context

{context}
"""

SQL_GENERATION_USER_PROMPT = """\
Question: {question}

{hints}

Write the {dialect} SQL query that answers this question."""

SQL_REPAIR_SYSTEM_PROMPT = """\
You are a senior analytics engineer correcting a {dialect} SQL query that failed \
validation against the data warehouse Knowledge Base.

Fix every reported problem while preserving the original intent of the question. \
Use ONLY the tables, columns and join paths in the context below. Date arithmetic \
must be {dialect}: "24 hours ago" is {relative_timestamp}, and the current moment \
is {current_timestamp}.

Respond with the corrected SQL query and nothing else. Do not wrap it in markdown \
fences, do not explain the change, and do not end it with a semicolon.

# Knowledge Base context

{context}
"""

SQL_REPAIR_USER_PROMPT = """\
Original question: {question}

The following query failed validation:

{sql}

Validation errors that must be fixed:
{errors}

Return the corrected query."""

INSUFFICIENT_CONTEXT_MARKER = "INSUFFICIENT_CONTEXT"

_NO_HINTS = "No additional hints were derived from the question."


def _time_syntax(dialect: str) -> dict[str, str]:
    """Render the target engine's own time expressions, for the prompt to quote.

    The Knowledge Base is one body of examples shared by every engine, and its
    snippets are written one way. Naming the dialect alone loses to eight worked
    examples that all say ``datetime('now', ...)``; showing the engine's own form
    beside them is what makes the instruction stick.
    """
    resolved = get_dialect(dialect)
    return {
        "relative_timestamp": resolved.timestamp_at_offset(24, TimeUnit.HOUR),
        "current_timestamp": resolved.current_timestamp,
    }


def build_generation_prompts(
    context: RetrievedContext,
    analysis: QuestionAnalysis,
    dialect: str,
) -> tuple[str, str]:
    """Build the (system, user) prompt pair for first-pass generation."""
    hints = analysis.to_prompt_hints() or _NO_HINTS

    system_prompt = SQL_GENERATION_SYSTEM_PROMPT.format(
        dialect=dialect,
        context=context.render(),
        **_time_syntax(dialect),
    )
    user_prompt = SQL_GENERATION_USER_PROMPT.format(
        question=context.question,
        hints=f"Hints derived from the question:\n{hints}",
        dialect=dialect,
    )
    return system_prompt, user_prompt


def build_repair_prompts(
    context: RetrievedContext,
    sql: str,
    errors: list[str],
    dialect: str,
) -> tuple[str, str]:
    """Build the (system, user) prompt pair for a repair attempt."""
    system_prompt = SQL_REPAIR_SYSTEM_PROMPT.format(
        dialect=dialect,
        context=context.render(),
        **_time_syntax(dialect),
    )
    user_prompt = SQL_REPAIR_USER_PROMPT.format(
        question=context.question,
        sql=sql,
        errors="\n".join(f"- {error}" for error in errors),
    )
    return system_prompt, user_prompt
