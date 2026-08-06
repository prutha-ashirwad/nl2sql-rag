"""Exception hierarchy for the NL2SQL system."""

from __future__ import annotations


class NL2SQLError(Exception):
    """Base class for every error raised by this package."""


class KnowledgeBaseError(NL2SQLError):
    """Raised when the Knowledge Base cannot be loaded or fails validation."""


class RetrievalError(NL2SQLError):
    """Raised when the retrieval pipeline cannot serve a request."""


class GenerationError(NL2SQLError):
    """Raised when SQL could not be produced for a question."""


class LLMError(NL2SQLError):
    """Raised when the configured language model provider fails."""


class ValidationError(NL2SQLError):
    """Raised when generated SQL fails a hard validation gate."""


class ExecutionError(NL2SQLError):
    """Raised when a validated query cannot be executed."""


class ConfigurationError(NL2SQLError):
    """Raised when the runtime configuration is incomplete or inconsistent."""
