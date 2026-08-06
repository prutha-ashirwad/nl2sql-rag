"""Shared pytest fixtures.

The Knowledge Base and the retrieval indexes are expensive to build and read-only
once built, so they are session-scoped and shared across every test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine

from nl2sql.analysis.question_analyzer import QuestionAnalyzer
from nl2sql.config import DEFAULT_KB_PATH, Settings, get_settings
from nl2sql.database.engine import build_engine
from nl2sql.database.executor import QueryExecutor
from nl2sql.database.seed import seed_database
from nl2sql.generation.deterministic.generator import DeterministicSQLGenerator
from nl2sql.knowledge_base.loader import load_knowledge_base
from nl2sql.knowledge_base.models import KnowledgeBase
from nl2sql.knowledge_base.registry import KnowledgeBaseRegistry
from nl2sql.pipeline import NL2SQLPipeline
from nl2sql.retrieval.context_builder import SchemaContextBuilder
from nl2sql.retrieval.hybrid_retriever import HybridRetriever
from nl2sql.validation.validator import SQLValidator


@pytest.fixture(scope="session", autouse=True)
def isolate_environment() -> Iterator[None]:
    """Hide the developer's own environment and ``.env`` from the suite.

    LANGSMITH_TRACING is forced off rather than merely unset: the FastAPI lifespan
    calls ``get_settings`` directly, which reads ``.env``, so booting the app in one
    test would otherwise upload every later test question to a real project.
    """
    managed = {name.upper() for name in Settings.model_fields}

    with pytest.MonkeyPatch.context() as patch:
        for name in [key for key in os.environ if key.upper() in managed]:
            patch.delenv(name, raising=False)
        patch.setenv("LANGSMITH_TRACING", "false")
        get_settings.cache_clear()
        yield
    get_settings.cache_clear()


def build_test_settings(**overrides: object) -> Settings:
    """Build settings for a test, ignoring any ``.env`` on the developer's machine.

    ``_env_file=None`` disables dotenv loading entirely, which — together with
    :func:`isolate_environment` — makes settings construction fully hermetic.
    """
    overrides.setdefault("knowledge_base_path", DEFAULT_KB_PATH)
    return Settings(_env_file=None, **overrides)


@pytest.fixture(scope="session")
def knowledge_base() -> KnowledgeBase:
    """The bundled Knowledge Base, loaded and validated once."""
    return load_knowledge_base(DEFAULT_KB_PATH)


@pytest.fixture(scope="session")
def registry(knowledge_base: KnowledgeBase) -> KnowledgeBaseRegistry:
    """Indexes over the bundled Knowledge Base."""
    return KnowledgeBaseRegistry(knowledge_base)


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Settings pointing at a throwaway database, isolated from the demo one."""
    database_path: Path = tmp_path_factory.mktemp("db") / "test.db"
    return build_test_settings(
        database_url=f"sqlite:///{database_path}",
        llm_provider="deterministic",
        log_level="WARNING",
    )


@pytest.fixture(scope="session")
def retriever(registry: KnowledgeBaseRegistry) -> HybridRetriever:
    """Hybrid retriever over the bundled Knowledge Base."""
    return HybridRetriever(registry)


@pytest.fixture(scope="session")
def context_builder(
    registry: KnowledgeBaseRegistry, retriever: HybridRetriever, settings: Settings
) -> SchemaContextBuilder:
    """Context builder wired to the shared retriever."""
    return SchemaContextBuilder(registry, retriever, settings)


@pytest.fixture(scope="session")
def analyzer(registry: KnowledgeBaseRegistry) -> QuestionAnalyzer:
    """Question analyzer over the bundled Knowledge Base."""
    return QuestionAnalyzer(registry)


@pytest.fixture(scope="session")
def validator(registry: KnowledgeBaseRegistry) -> SQLValidator:
    """SQL validator over the bundled Knowledge Base."""
    return SQLValidator(registry, dialect="sqlite")


@pytest.fixture(scope="session")
def generator(registry: KnowledgeBaseRegistry) -> DeterministicSQLGenerator:
    """Deterministic generator, which needs no model credentials."""
    return DeterministicSQLGenerator(registry, dialect="sqlite")


@pytest.fixture(scope="session")
def seeded_engine(
    registry: KnowledgeBaseRegistry, settings: Settings
) -> Iterator[Engine]:
    """A database built from the Knowledge Base and populated with demo data."""
    engine = build_engine(settings.database_url)
    seed_database(engine, registry, recreate=True)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def executor(seeded_engine: Engine) -> QueryExecutor:
    """Query executor bound to the seeded test database."""
    return QueryExecutor(seeded_engine)


@pytest.fixture(scope="session")
def pipeline(
    registry: KnowledgeBaseRegistry,
    settings: Settings,
    generator: DeterministicSQLGenerator,
    executor: QueryExecutor,
) -> NL2SQLPipeline:
    """A fully wired pipeline running against the seeded test database."""
    return NL2SQLPipeline(
        registry, settings, generator=generator, executor=executor
    )
