"""Command line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from nl2sql.config import get_settings
from nl2sql.database.engine import build_engine
from nl2sql.database.seed import seed_database
from nl2sql.dialects import get_dialect
from nl2sql.evaluation.models import Verdict
from nl2sql.evaluation.runner import evaluate as run_evaluation
from nl2sql.evaluation.runner import load_cases
from nl2sql.exceptions import NL2SQLError
from nl2sql.knowledge_base.loader import load_knowledge_base
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.logging_config import configure_logging
from nl2sql.pipeline import NL2SQLAnswer, NL2SQLPipeline
from nl2sql.tracing import configure_tracing

app = typer.Typer(
    name="nl2sql",
    help="Convert natural-language questions into validated SQL.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

DEMO_QUESTIONS = (
    "Show all failed observations in the last 24 hours.",
    "List interfaces with the highest failure count.",
    "Display observations grouped by environment.",
    "What is the success rate per environment over the last 7 days?",
    "Which devices had the most failures in production?",
    "Break down failures by failure reason.",
    "Show the top 5 sites by failure count in the last week.",
)


def _render_answer(answer: NL2SQLAnswer, *, show_trace: bool, max_rows: int) -> None:
    """Print an answer as formatted terminal output."""
    console.print(Panel(answer.question, title="Question", border_style="cyan"))

    if answer.sql:
        console.print(
            Panel(
                Syntax(answer.sql, "sql", theme="ansi_dark", word_wrap=True),
                title=f"Generated SQL ({answer.generator})",
                border_style="green" if answer.succeeded else "yellow",
            )
        )

    if answer.answer:
        console.print(f"[bold]Summary:[/bold] {answer.answer}\n")

    if answer.validation_errors:
        console.print("[bold red]Validation errors[/bold red]")
        for error in answer.validation_errors:
            console.print(f"  • {error}")
        console.print()

    if answer.validation_warnings:
        console.print("[bold yellow]Validation warnings[/bold yellow]")
        for warning in answer.validation_warnings:
            console.print(f"  • {warning}")
        console.print()

    if answer.rows:
        table = Table(show_header=True, header_style="bold magenta", box=None)
        for column in answer.columns:
            table.add_column(str(column))
        for row in answer.rows[:max_rows]:
            table.add_row(
                *["" if value is None else str(value) for value in row.values()]
            )
        console.print(table)
        if answer.row_count > max_rows:
            console.print(f"[dim]… showing {max_rows} of {answer.row_count} rows[/dim]")
        console.print()

    if show_trace:
        trace_table = Table(title="Workflow trace", box=None, header_style="bold blue")
        trace_table.add_column("Step")
        trace_table.add_column("Summary")
        trace_table.add_column("Duration", justify="right")
        for event in answer.trace:
            trace_table.add_row(event.node, event.summary, f"{event.duration_ms:.1f} ms")
        console.print(trace_table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    show_trace: Annotated[
        bool, typer.Option("--trace", help="Show each workflow step.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the full answer as JSON.")
    ] = False,
    max_rows: Annotated[
        int, typer.Option("--max-rows", help="Rows to display in the table.")
    ] = 20,
) -> None:
    """Answer a single natural-language question."""
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(settings)

    try:
        pipeline = NL2SQLPipeline.create(settings)
        answer = pipeline.answer(question)
    except NL2SQLError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(json.dumps(answer.to_dict(), default=str))
    else:
        _render_answer(answer, show_trace=show_trace, max_rows=max_rows)

    raise typer.Exit(code=0 if answer.succeeded else 2)


@app.command()
def seed(
    recreate: Annotated[
        bool, typer.Option("--recreate/--keep", help="Drop existing tables first.")
    ] = True,
) -> None:
    """Build the demo database from the Knowledge Base and populate it."""
    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        knowledge_base = load_knowledge_base(settings.knowledge_base_path)
        registry = KnowledgeBaseRegistry(knowledge_base)
        engine = build_engine(settings.database_url)
        counts = seed_database(
            engine,
            registry,
            recreate=recreate,
            dialect=get_dialect(settings.sql_dialect),
        )
    except NL2SQLError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Seeded tables", box=None, header_style="bold magenta")
    table.add_column("Table")
    table.add_column("Rows", justify="right")
    for name, count in sorted(counts.items(), key=lambda item: -item[1]):
        table.add_row(name, f"{count:,}")

    console.print(table)
    console.print(f"\n[green]Database ready at[/green] {settings.database_url}")


@app.command()
def inspect() -> None:
    """Show a summary of the loaded Knowledge Base."""
    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        knowledge_base = load_knowledge_base(settings.knowledge_base_path)
    except NL2SQLError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    registry = KnowledgeBaseRegistry(knowledge_base)
    summary = knowledge_base.summary()

    overview = Table(title="Knowledge Base", box=None, header_style="bold magenta")
    overview.add_column("Entity")
    overview.add_column("Count", justify="right")
    for key, value in summary.items():
        overview.add_row(key.replace("_", " ").title(), str(value))
    console.print(overview)
    console.print()

    tables = Table(title="Tables", box=None, header_style="bold blue")
    tables.add_column("Table")
    tables.add_column("Domain")
    tables.add_column("Alias")
    tables.add_column("Columns", justify="right")
    tables.add_column("Grain")
    for table in registry.tables:
        tables.add_row(
            table.name,
            table.domain,
            registry.alias_for(table.name),
            str(len(table.columns)),
            table.grain,
        )
    console.print(tables)


@app.command()
def demo(
    show_trace: Annotated[
        bool, typer.Option("--trace", help="Show each workflow step.")
    ] = False,
) -> None:
    """Run the bundled example questions end to end."""
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(settings)

    try:
        pipeline = NL2SQLPipeline.create(settings)
    except NL2SQLError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    succeeded = 0
    for index, question in enumerate(DEMO_QUESTIONS, start=1):
        console.rule(f"[bold]{index}/{len(DEMO_QUESTIONS)}[/bold]")
        answer = pipeline.answer(question)
        _render_answer(answer, show_trace=show_trace, max_rows=5)
        succeeded += int(answer.succeeded)

    console.rule()
    style = "green" if succeeded == len(DEMO_QUESTIONS) else "yellow"
    console.print(
        f"[bold {style}]{succeeded}/{len(DEMO_QUESTIONS)} questions answered "
        f"with a valid query.[/bold {style}]"
    )

    raise typer.Exit(code=0 if succeeded == len(DEMO_QUESTIONS) else 2)


@app.command()
def evaluate(
    cases_path: Annotated[
        Path | None,
        typer.Option("--cases", help="Case file. Defaults to evaluation/cases.yaml."),
    ] = None,
    show_failures: Annotated[
        bool, typer.Option("--failures/--no-failures", help="Print each failing case.")
    ] = True,
) -> None:
    """Measure answer accuracy against the held-out evaluation set."""
    settings = get_settings()
    configure_logging("WARNING")

    try:
        cases = load_cases(cases_path)
        report = run_evaluation(cases, settings=settings)
    except (NL2SQLError, OSError, ValueError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"\n[bold]{report.engine}[/bold] — {len(cases)} cases\n")

    breakdown = Table(box=None, header_style="bold magenta")
    breakdown.add_column("Category")
    breakdown.add_column("Passed", justify="right")
    breakdown.add_column("Cases", justify="right")
    breakdown.add_column("", justify="right")
    for category, (hit, run) in report.by_category().items():
        share = hit / run if run else 0.0
        colour = "green" if hit == run else "yellow" if share >= 0.5 else "red"
        breakdown.add_row(
            category, f"[{colour}]{hit}[/{colour}]", str(run), f"{share:6.0%}"
        )
    console.print(breakdown)

    verdicts = Table(box=None, header_style="bold blue")
    verdicts.add_column("Verdict")
    verdicts.add_column("Cases", justify="right")
    for verdict in Verdict:
        count = report.count(verdict)
        if count:
            verdicts.add_row(verdict.value.replace("_", " "), str(count))
    console.print()
    console.print(verdicts)

    if show_failures:
        failures = [o for o in report.outcomes if not o.passed]
        if failures:
            console.print()
            for outcome in failures:
                console.print(
                    f"[yellow]{outcome.case.id}[/yellow] {outcome.case.question}\n"
                    f"  [dim]{outcome.verdict.value}: {outcome.detail}[/dim]"
                )

    for outcome in report.broken_cases:
        console.print(
            f"[bold red]Broken case[/bold red] {outcome.case.id}: {outcome.detail}"
        )

    console.print()
    style = "green" if report.accuracy == 1.0 else "yellow"
    summary = (
        f"[bold {style}]Execution accuracy {report.passed}/{report.total} "
        f"({report.accuracy:.0%})[/bold {style}]"
    )
    if report.table_recall is not None:
        summary += f"  ·  required-table recall {report.table_recall:.0%}"
    console.print(summary)

    raise typer.Exit(code=0 if not report.broken_cases else 2)


def main() -> None:
    """Console script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover - module executed as a script
    sys.exit(app())
