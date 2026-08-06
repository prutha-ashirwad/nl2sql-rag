"""NL2SQL system built on LangGraph and retrieval-augmented generation."""

from __future__ import annotations

from nl2sql.config import LLMProvider, Settings, get_settings
from nl2sql.exceptions import (
    ConfigurationError,
    ExecutionError,
    GenerationError,
    KnowledgeBaseError,
    LLMError,
    NL2SQLError,
    RetrievalError,
    ValidationError,
)
from nl2sql.pipeline import NL2SQLAnswer, NL2SQLPipeline

__version__ = "1.0.0"

__all__ = [
    "ConfigurationError",
    "ExecutionError",
    "GenerationError",
    "KnowledgeBaseError",
    "LLMError",
    "LLMProvider",
    "NL2SQLAnswer",
    "NL2SQLError",
    "NL2SQLPipeline",
    "RetrievalError",
    "Settings",
    "ValidationError",
    "__version__",
    "get_settings",
]
