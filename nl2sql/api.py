"""HTTP API exposing the pipeline over REST.

Run with ``uvicorn nl2sql.api:app --reload``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from nl2sql.config import get_settings
from nl2sql.exceptions import NL2SQLError
from nl2sql.logging_config import configure_logging, get_logger
from nl2sql.pipeline import NL2SQLPipeline
from nl2sql.tracing import configure_tracing, is_tracing_active

logger = get_logger(__name__)


class AskRequest(BaseModel):
    """Body of a question request."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The natural-language question to convert into SQL.",
        examples=["Show all failed observations in the last 24 hours."],
    )
    include_trace: bool = Field(
        default=False, description="Include the per-step workflow trace."
    )


class AskResponse(BaseModel):
    """Result of converting a question into SQL."""

    question: str
    succeeded: bool
    sql: str | None
    answer: str
    explanation: str
    generator: str
    tables_used: list[str]
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    validation_errors: list[str]
    validation_warnings: list[str]
    errors: list[str]
    """Blocking failures raised after validation, such as a rejected execution."""

    repair_attempts: int
    tokens_used: int
    retrieved_documents: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "What the RAG step retrieved, ranked, with each document's score. This is "
            "the evidence for the SQL: it shows which schema context the query was "
            "written from."
        ),
    )
    trace: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    """Service health and Knowledge Base coverage."""

    status: str
    version: str
    knowledge_base: dict[str, int]
    provider: str = Field(description="The provider the configuration resolves to.")
    generator: str = Field(
        description=(
            "The generator actually in use. Differs from `provider` when a provider "
            "could not be constructed and the deterministic planner took over."
        )
    )
    execution_enabled: bool
    tracing_enabled: bool = Field(
        description=(
            "Whether workflow traces are being sent to LangSmith. False when tracing "
            "is switched off, and also when it was requested without an API key."
        )
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Build the pipeline once at start-up and share it across requests.

    A pipeline already on ``app.state`` is reused, so a host can inject its own.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(settings)

    if getattr(application.state, "pipeline", None) is None:
        logger.info("Building the NL2SQL pipeline")
        application.state.pipeline = NL2SQLPipeline.create(settings)
    else:
        logger.info("Reusing the pipeline already provided on app.state")

    logger.info("API ready")

    yield

    logger.info("API shutting down")


app = FastAPI(
    title="Intelligent NL2SQL API",
    description=(
        "Converts natural-language questions into validated SQL using a LangGraph "
        "workflow grounded on a structured Knowledge Base."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _get_pipeline(request: Request) -> NL2SQLPipeline:
    """Return the shared pipeline, or fail clearly if start-up did not complete."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:  # pragma: no cover - only reachable on a failed start-up
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The pipeline is not ready.",
        )
    return pipeline


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health(request: Request) -> HealthResponse:
    """Report service health and Knowledge Base coverage."""
    pipeline = _get_pipeline(request)
    settings = pipeline.settings

    return HealthResponse(
        status="ok",
        version=app.version,
        knowledge_base=pipeline.registry.knowledge_base.summary(),
        provider=settings.resolve_provider().value,
        generator=pipeline.generator_name,
        execution_enabled=settings.execute_queries,
        tracing_enabled=is_tracing_active(),
    )


@app.post("/ask", response_model=AskResponse, tags=["query"])
def ask(request: Request, payload: AskRequest) -> AskResponse:
    """Convert a natural-language question into validated SQL.

    An unanswerable question still returns HTTP 200 with ``succeeded: false``.
    """
    pipeline = _get_pipeline(request)

    try:
        answer = pipeline.answer(payload.question)
    except NL2SQLError as exc:
        logger.exception("Failed to answer a question")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    data = answer.to_dict()
    if not payload.include_trace:
        data.pop("trace", None)

    return AskResponse(**data)


@app.get("/schema/tables", tags=["knowledge-base"])
def list_tables(request: Request) -> dict[str, Any]:
    """List every table declared in the Knowledge Base."""
    registry = _get_pipeline(request).registry

    return {
        "count": len(registry.tables),
        "tables": [
            {
                "name": table.name,
                "domain": table.domain,
                "grain": table.grain,
                "description": table.description,
                "column_count": len(table.columns),
                "primary_key": table.primary_key,
            }
            for table in registry.tables
        ],
    }


@app.get("/schema/tables/{table_name}", tags=["knowledge-base"])
def get_table(request: Request, table_name: str) -> dict[str, Any]:
    """Return the full Knowledge Base entry for one table."""
    registry = _get_pipeline(request).registry
    table = registry.get_table(table_name)

    if table is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table {table_name!r} is not declared in the Knowledge Base.",
        )

    return table.model_dump(mode="json")


@app.get("/schema/relationships", tags=["knowledge-base"])
def list_relationships(request: Request) -> dict[str, Any]:
    """List every relationship declared in the Knowledge Base."""
    registry = _get_pipeline(request).registry

    return {
        "count": len(registry.relationships),
        "relationships": [
            relationship.model_dump(mode="json")
            for relationship in registry.relationships
        ],
    }
