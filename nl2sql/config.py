"""Runtime configuration read from the environment or a ``.env`` file."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from nl2sql.exceptions import ConfigurationError

PACKAGE_ROOT = Path(__file__).resolve().parent
# Flat layout: the package sits directly under the repository root.
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_KB_PATH = PACKAGE_ROOT / "knowledge_base" / "data"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "observability.db"


class EmbeddingProviderName(str, Enum):
    """Which backend produces the dense half of the retrieval signal."""

    TFIDF = "tfidf"
    OPENAI = "openai"


class VectorStoreName(str, Enum):
    """Which index backs dense search. Both are exact and rank identically."""

    MEMORY = "memory"
    FAISS = "faiss"


class LLMProvider(str, Enum):
    """Which language model backend to use for SQL generation."""

    AUTO = "auto"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DETERMINISTIC = "deterministic"


DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.ANTHROPIC: "claude-opus-5",
    LLMProvider.OPENAI: "gpt-4o",
}

# Substrings that identify which provider a model identifier belongs to.
_MODEL_OWNERSHIP: dict[LLMProvider, tuple[str, ...]] = {
    LLMProvider.ANTHROPIC: ("claude",),
    LLMProvider.OPENAI: ("gpt-", "o1-", "o3-", "o4-"),
}


class Settings(BaseSettings):
    """Application settings, populated from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Knowledge base -------------------------------------------------------
    knowledge_base_path: Path = Field(
        default=DEFAULT_KB_PATH,
        description="Directory containing the Knowledge Base YAML files.",
    )

    # --- Retrieval ------------------------------------------------------------
    retrieval_top_k: int = Field(
        default=8, ge=1, le=50, description="Documents returned by the retriever."
    )
    retrieval_min_score: float = Field(
        default=0.01,
        ge=0.0,
        description="Documents scoring below this threshold are discarded.",
    )
    retrieval_expand_joins: bool = Field(
        default=True,
        description="Pull in tables needed to join the retrieved tables together.",
    )
    lexical_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight of the keyword retriever when fusing with the vector one.",
    )
    # Hosted embeddings are the accuracy default: they reach a table whose wording the
    # question never uses. Without a key this degrades to tfidf, so a clone with no
    # credentials still runs.
    embedding_provider: EmbeddingProviderName = Field(
        default=EmbeddingProviderName.OPENAI,
        description="Dense embedding backend. 'openai' degrades to tfidf without a key.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model used when the embedding provider is openai.",
    )
    embedding_dimensions: int = Field(
        default=1536,
        ge=64,
        le=3072,
        description="Vector width requested from the embedding model.",
    )
    embedding_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Per-request budget for hosted embeddings; short, because a slow "
        "one degrades a single signal rather than stalling the answer.",
    )
    # memory, not faiss: both are exact and rank identically, and at this corpus size
    # the numpy scan is the faster of the two. faiss is for when the corpus grows.
    vector_store: VectorStoreName = Field(
        default=VectorStoreName.MEMORY,
        description="Index backing dense search. 'faiss' degrades to memory if absent.",
    )

    # --- Language model -------------------------------------------------------
    llm_provider: LLMProvider = Field(default=LLMProvider.AUTO)
    llm_model: str | None = Field(
        default=None,
        description=(
            "Model identifier passed to the configured provider. Leave unset to use "
            "that provider's default from DEFAULT_MODELS."
        ),
    )
    llm_max_tokens: int = Field(default=4096, ge=256, le=64000)
    llm_timeout_seconds: float = Field(default=90.0, gt=0)
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)

    # --- Tracing --------------------------------------------------------------
    langsmith_tracing: bool = Field(
        default=False,
        description="Send workflow traces to LangSmith. Requires an API key.",
    )
    langsmith_api_key: str | None = Field(default=None)
    langsmith_project: str = Field(
        default="nl2sql-rag",
        description="LangSmith project that runs are grouped under.",
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint; change it for the EU or a self-hosted one.",
    )

    # --- Workflow -------------------------------------------------------------
    max_repair_attempts: int = Field(
        default=2,
        ge=0,
        le=5,
        description="How many times the agent may rewrite SQL that failed validation.",
    )
    execute_queries: bool = Field(
        default=True, description="Run validated SQL against the configured database."
    )

    # --- Database -------------------------------------------------------------
    database_url: str = Field(
        default=f"sqlite:///{DEFAULT_DATABASE_PATH}",
        description="SQLAlchemy URL of the read-only analytics database.",
    )
    sql_dialect: str = Field(
        default="sqlite",
        description="Target SQL dialect used for parsing and validation.",
    )
    max_result_rows: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional cap on rows returned. Unset means every matching row is "
            "returned; a query is bounded only when the question asks for a number."
        ),
    )
    query_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Web frontend ---------------------------------------------------------
    frontend_host: str = Field(
        default="127.0.0.1",
        description="Network interface the Streamlit app binds to.",
    )
    frontend_port: int = Field(
        default=8501, ge=1024, le=65535, description="Port the Streamlit app serves on."
    )

    # --- Observability --------------------------------------------------------
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalised = value.strip().upper()
        if normalised not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalised

    @field_validator("knowledge_base_path")
    @classmethod
    def _validate_kb_path(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Knowledge Base path does not exist: {resolved}")
        return resolved

    def resolve_provider(self) -> LLMProvider:
        """Resolve ``AUTO`` into the provider that can actually be used."""
        if self.llm_provider is not LLMProvider.AUTO:
            return self.llm_provider
        if self.anthropic_api_key:
            return LLMProvider.ANTHROPIC
        if self.openai_api_key:
            return LLMProvider.OPENAI
        return LLMProvider.DETERMINISTIC

    def resolve_model(self) -> str:
        """Return the model identifier to send to the resolved provider.

        Raises:
            ConfigurationError: if the configured model belongs to another provider.
        """
        provider = self.resolve_provider()
        default = DEFAULT_MODELS.get(provider, "")

        if not self.llm_model:
            return default

        for owner, markers in _MODEL_OWNERSHIP.items():
            if owner is provider:
                continue
            if any(marker in self.llm_model.lower() for marker in markers):
                raise ConfigurationError(
                    f"Model {self.llm_model!r} looks like a {owner.value} model but "
                    f"the resolved provider is {provider.value}. Set "
                    f"LLM_MODEL to a {provider.value} model "
                    f"(for example {default!r}), unset it to use the default, or "
                    f"change LLM_PROVIDER."
                )

        return self.llm_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
