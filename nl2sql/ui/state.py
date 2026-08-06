"""Shared state behind the Streamlit pages."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import streamlit as st

from nl2sql.config import (
    LLMProvider,
    Settings,
    get_settings,
)
from nl2sql.database.engine import check_connection, describe_database_url
from nl2sql.dialects import is_supported_dialect
from nl2sql.exceptions import ConfigurationError, NL2SQLError
from nl2sql.generation.deterministic.generator import DeterministicSQLGenerator
from nl2sql.generation.llm_generator import LLMSQLGenerator
from nl2sql.knowledge_base.authoring import KnowledgeBaseEditor
from nl2sql.knowledge_base.loader import load_knowledge_base
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.llm.factory import build_llm_client
from nl2sql.logging_config import get_logger
from nl2sql.pipeline import NL2SQLAnswer, NL2SQLPipeline
from nl2sql.retrieval.hybrid_retriever import HybridRetriever

logger = get_logger(__name__)

PLANNER_LABEL = "Knowledge Base planner (no model)"

# Providers without credentials still list their models, labelled with the missing
# variable.
SELECTABLE_MODELS: dict[LLMProvider, tuple[str, ...]] = {
    LLMProvider.ANTHROPIC: ("claude-opus-5", "claude-sonnet-5"),
    LLMProvider.OPENAI: ("gpt-4o", "gpt-4o-mini"),
}

_MISSING_KEY_VARIABLE: dict[LLMProvider, str] = {
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.OPENAI: "OPENAI_API_KEY",
}


def unavailable_label(model: str, provider: LLMProvider) -> str:
    """Label a model the configuration cannot currently use."""
    return f"{model} — set {_MISSING_KEY_VARIABLE[provider]}"


@dataclass(slots=True)
class RunRecord:
    """One answered question, kept so other pages can inspect it."""

    answer: NL2SQLAnswer
    engine: str
    elapsed_seconds: float


@dataclass(slots=True)
class Workspace:
    """Engines, the Knowledge Base editor, and the last run."""

    settings: Settings
    engines: dict[str, NL2SQLPipeline | None] = field(default_factory=dict)
    planner: NL2SQLPipeline | None = None
    retriever: HybridRetriever | None = None
    editor: KnowledgeBaseEditor | None = None
    last_run: RunRecord | None = None

    # -- Engines --------------------------------------------------------------

    @property
    def available(self) -> dict[str, NL2SQLPipeline]:
        """Only the engines that can actually answer."""
        return {
            label: engine
            for label, engine in self.engines.items()
            if engine is not None
        }

    @property
    def ready_model_count(self) -> int:
        """How many model-backed engines are usable, excluding the planner."""
        return sum(1 for label in self.available if label != PLANNER_LABEL)

    @property
    def registry(self) -> KnowledgeBaseRegistry:
        """Knowledge Base indexes the engines share."""
        assert self.planner is not None  # noqa: S101 - set during build
        return self.planner.registry

    def default_engine(self) -> str:
        """The engine a visitor gets before choosing one."""
        models = [label for label in self.available if label != PLANNER_LABEL]
        return models[0] if models else PLANNER_LABEL

    def ask(self, question: str, engine_label: str) -> RunRecord:
        """Answer ``question`` with the named engine and remember the run."""
        engine = self.available[engine_label]

        started = time.perf_counter()
        answer = engine.answer(question, tags=[f"engine:{engine_label}"])
        record = RunRecord(
            answer=answer,
            engine=engine_label,
            elapsed_seconds=time.perf_counter() - started,
        )

        self.last_run = record
        return record

    # -- Mutations ------------------------------------------------------------

    def reload(self) -> None:
        """Rebuild every engine from the Knowledge Base as it now stands on disk."""
        previous = dict(self.engines), self.planner
        try:
            self._build_engines()
        except Exception:
            self.engines, self.planner = previous
            logger.exception("Knowledge Base reload failed; keeping the old engines")
            raise

    def switch_database(self, database_url: str, sql_dialect: str) -> None:
        """Point every engine at a different database."""
        if not is_supported_dialect(sql_dialect):
            raise ConfigurationError(f"Unknown SQL dialect {sql_dialect!r}.")

        # Pipelines tolerate an unreachable database, so a bad URL would otherwise
        # only surface as every later question returning no rows.
        check_connection(database_url)

        previous_settings = self.settings
        previous_engines = dict(self.engines), self.planner
        try:
            self.settings = self.settings.model_copy(
                update={"database_url": database_url, "sql_dialect": sql_dialect}
            )
            self._build_engines()
        except Exception:
            self.settings = previous_settings
            self.engines, self.planner = previous_engines
            logger.exception("Switching database failed; keeping the previous one")
            raise

        logger.info("Now querying %s", describe_database_url(database_url))

    def describe_retrieval(self) -> str:
        """The dense backend actually in use, which a fallback may have changed."""
        if self.retriever is None:
            return "unknown"
        embedder, store = self.retriever.backends
        return f"{embedder} · {store}"

    def describe_database(self) -> str:
        """The current connection, with the password masked."""
        return describe_database_url(self.settings.database_url)

    # -- Construction ---------------------------------------------------------

    def _build_engines(self) -> None:
        """Build one pipeline per selectable model, plus the planner.

        The Knowledge Base, its indexes and the database connection are built once and
        shared across engines; only the generator differs.
        """
        settings = self.settings
        knowledge_base = load_knowledge_base(settings.knowledge_base_path)
        registry = KnowledgeBaseRegistry(knowledge_base)
        retriever = HybridRetriever(
            registry, settings=settings, lexical_weight=settings.lexical_weight
        )
        self.retriever = retriever
        executor = NL2SQLPipeline._build_executor(settings)  # noqa: SLF001

        deterministic = DeterministicSQLGenerator(
            registry,
            dialect=settings.sql_dialect,
        )

        def pipeline_for(generator) -> NL2SQLPipeline:  # noqa: ANN001 - SQLGenerator
            return NL2SQLPipeline(
                registry,
                settings,
                generator=generator,
                executor=executor,
                retriever=retriever,
            )

        self.planner = pipeline_for(deterministic)
        engines: dict[str, NL2SQLPipeline | None] = {PLANNER_LABEL: self.planner}

        credentials = {
            LLMProvider.ANTHROPIC: settings.anthropic_api_key,
            LLMProvider.OPENAI: settings.openai_api_key,
        }

        for provider, models in SELECTABLE_MODELS.items():
            for model in models:
                if not credentials.get(provider):
                    engines[unavailable_label(model, provider)] = None
                    continue

                configured = settings.model_copy(
                    update={"llm_provider": provider, "llm_model": model}
                )
                try:
                    client = build_llm_client(configured)
                except (ConfigurationError, NL2SQLError) as exc:
                    logger.warning("Skipping %s: %s", model, exc)
                    engines[unavailable_label(model, provider)] = None
                    continue

                engines[model] = (
                    pipeline_for(
                        LLMSQLGenerator(
                            client,
                            dialect=settings.sql_dialect,
                            fallback=deterministic,
                        )
                    )
                    if client is not None
                    else None
                )

        self.engines = engines
        self.editor = KnowledgeBaseEditor(settings.knowledge_base_path)
        logger.info("Engines ready: %s", ", ".join(self.available))


@st.cache_resource(show_spinner="Loading the Knowledge Base and building the indexes…")
def get_workspace() -> Workspace:
    """Build the workspace once and share it across reruns and sessions."""
    workspace = Workspace(settings=get_settings())
    workspace._build_engines()  # noqa: SLF001 - construction, kept off the public API
    return workspace
