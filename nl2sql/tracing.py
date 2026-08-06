"""Optional LangSmith tracing.

Payloads are summarised before upload: LangGraph traces the whole state at every
node, which is large enough to saturate the background uploader and drop runs.
"""

from __future__ import annotations

import atexit
import os
from typing import TYPE_CHECKING, Any

from nl2sql.config import LLMProvider, Settings
from nl2sql.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - import kept out of the untraced path
    from langchain_core.tracers import LangChainTracer

logger = get_logger(__name__)

# Read by the LangSmith SDK when it builds a client.
_TRACING_ENABLED_VAR = "LANGSMITH_TRACING"
_API_KEY_VAR = "LANGSMITH_API_KEY"
_PROJECT_VAR = "LANGSMITH_PROJECT"
_ENDPOINT_VAR = "LANGSMITH_ENDPOINT"

# How many collection items to name before summarising the rest as a count.
_MAX_LISTED = 12

_tracer: LangChainTracer | None = None
_active = False


def configure_tracing(settings: Settings) -> bool:
    """Enable LangSmith tracing if it is both requested and usable.

    Returns False when tracing was requested but could not be started.
    """
    global _tracer, _active

    _tracer = None
    _active = False

    if not settings.langsmith_tracing:
        # Set explicitly so a stray LANGSMITH_TRACING in the environment cannot win.
        os.environ[_TRACING_ENABLED_VAR] = "false"
        return False

    if not settings.langsmith_api_key:
        logger.warning(
            "LANGSMITH_TRACING is on but LANGSMITH_API_KEY is not set; "
            "tracing stays off. Set the key or turn tracing off to silence this."
        )
        os.environ[_TRACING_ENABLED_VAR] = "false"
        return False

    os.environ[_API_KEY_VAR] = settings.langsmith_api_key
    os.environ[_PROJECT_VAR] = settings.langsmith_project
    os.environ[_ENDPOINT_VAR] = settings.langsmith_endpoint

    # Kept off: LangChain would otherwise attach its own tracer with no summarising
    # hooks, uploading a second unsummarised copy of every run.
    os.environ[_TRACING_ENABLED_VAR] = "false"

    try:
        _tracer = _build_tracer(settings)
    except Exception as exc:  # noqa: BLE001 - never fail start-up over telemetry
        logger.warning("LangSmith tracing could not be started: %s", exc)
        return False

    _active = True
    atexit.register(flush)
    logger.info(
        "LangSmith tracing enabled; runs are recorded under project %r",
        settings.langsmith_project,
    )
    return True


def _build_tracer(settings: Settings) -> LangChainTracer:
    """Build the tracer, with a client that summarises what it uploads."""
    from langchain_core.tracers import LangChainTracer
    from langsmith import Client

    client = Client(
        api_key=settings.langsmith_api_key,
        api_url=settings.langsmith_endpoint,
        hide_inputs=summarise_payload,
        hide_outputs=summarise_payload,
        # Without this a rejected batch is discarded in silence.
        tracing_error_callback=_report_upload_error,
    )
    return LangChainTracer(project_name=settings.langsmith_project, client=client)


def _report_upload_error(error: BaseException, /, **_: Any) -> None:
    """Log an upload failure instead of letting the run vanish."""
    logger.warning("LangSmith rejected a trace: %s", error)


def is_tracing_active() -> bool:
    """True when tracing was successfully enabled for this process."""
    return _active


def flush() -> None:
    """Upload anything still buffered, so a short-lived command does not lose its run."""
    if not _active:
        return

    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
        if _tracer is not None:
            _tracer.client.flush()
    except Exception as exc:  # noqa: BLE001 - shutdown must not raise
        logger.debug("Flushing LangSmith traces failed: %s", exc)


def summarise_payload(payload: Any) -> Any:
    """Reduce a traced workflow state to the parts that explain the answer.

    Result rows are dropped entirely: business data does not belong in a trace.
    """
    if not isinstance(payload, dict):
        return payload

    return {key: _summarise_value(key, value) for key, value in payload.items()}


def _summarise_value(key: str, value: Any) -> Any:
    """Summarise one state field, leaving anything already small untouched."""
    if value is None:
        return None

    summarisers = {
        "context": _summarise_context,
        "analysis": _summarise_analysis,
        "validation": _summarise_validation,
        "execution": _summarise_execution,
    }
    summariser = summarisers.get(key)
    if summariser is None:
        return value

    try:
        return summariser(value)
    except Exception:  # noqa: BLE001 - a trace is never worth an exception
        return f"<{type(value).__name__}>"


def _listed(names: list[str]) -> list[str] | str:
    """Return the names, or a count once the list stops being worth reading."""
    if len(names) <= _MAX_LISTED:
        return names
    return f"{len(names)} items"


def _summarise_context(context: Any) -> dict[str, Any]:
    """Describe what retrieval found, without reproducing it."""
    documents = getattr(context, "documents", []) or []

    return {
        "base_table": getattr(context, "base_table", None),
        "tables": _listed([table.name for table in getattr(context, "tables", []) or []]),
        "document_count": len(documents),
        "top_documents": [
            {"id": scored.document.id, "score": round(scored.score, 4)}
            for scored in documents[:5]
        ],
        "join_count": len(getattr(context, "join_steps", []) or []),
        "rule_count": len(getattr(context, "rules", []) or []),
        "example_count": len(getattr(context, "examples", []) or []),
        "unreachable_tables": getattr(context, "unreachable_tables", []) or [],
    }


def _summarise_analysis(analysis: Any) -> dict[str, Any]:
    """Describe what the question was understood to ask for."""
    window = getattr(analysis, "time_window", None)
    metric = getattr(analysis, "metric", None)

    return {
        "intent": getattr(getattr(analysis, "intent", None), "value", None),
        "named_tables": getattr(analysis, "named_tables", []),
        "filters": [
            f"{item.table}.{item.column} = {item.value!r}"
            for item in getattr(analysis, "value_filters", []) or []
        ],
        "groupings": [
            f"{item.table}.{item.column}"
            for item in getattr(analysis, "groupings", []) or []
        ],
        "metric": getattr(metric, "name", None),
        "time_window": window.describe() if window is not None else None,
        "row_limit": getattr(analysis, "row_limit", None),
        "rejection_reason": getattr(analysis, "rejection_reason", None),
    }


def _summarise_validation(report: Any) -> dict[str, Any]:
    """Describe the verdict, not the whole report."""
    return {
        "is_valid": getattr(report, "is_valid", None),
        "errors": report.error_messages() if hasattr(report, "error_messages") else [],
        "warnings": [issue.format() for issue in getattr(report, "warnings", []) or []],
        "referenced_tables": getattr(report, "referenced_tables", []),
    }


def _summarise_execution(result: Any) -> dict[str, Any]:
    """Describe the shape of the result set, never its contents."""
    return {
        "row_count": getattr(result, "row_count", None),
        "columns": getattr(result, "columns", []),
        "truncated": getattr(result, "truncated", None),
        "duration_ms": getattr(result, "duration_ms", None),
    }


def run_config(
    settings: Settings,
    *,
    generator_name: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-run config that labels one question in LangSmith.

    Args:
        settings: Configuration the run executes under.
        generator_name: The generator actually in use, which is not always the
            configured provider — a failed provider degrades to the planner.
        tags: Extra labels for grouping runs.
        metadata: Extra fields to record alongside the standard ones.
    """
    provider = settings.resolve_provider()
    recorded: dict[str, Any] = {
        "generator": generator_name,
        "provider": provider.value,
        "sql_dialect": settings.sql_dialect,
        "retrieval_top_k": settings.retrieval_top_k,
        "lexical_weight": settings.lexical_weight,
        "retrieval_min_score": settings.retrieval_min_score,
        "retrieval_expand_joins": settings.retrieval_expand_joins,
        "max_repair_attempts": settings.max_repair_attempts,
        "execute_queries": settings.execute_queries,
    }

    # Only meaningful for a model-backed run; a null would break metadata filtering.
    if provider not in {LLMProvider.DETERMINISTIC, LLMProvider.AUTO}:
        recorded["model"] = settings.resolve_model()

    if metadata:
        recorded.update(metadata)

    config: dict[str, Any] = {
        "run_name": "nl2sql",
        "tags": [
            f"generator:{generator_name}",
            f"provider:{provider.value}",
            *(tags or []),
        ],
        "metadata": recorded,
    }

    if _tracer is not None:
        config["callbacks"] = [_tracer]

    return config
