"""Construction of the configured language model client."""

from __future__ import annotations

from nl2sql.config import LLMProvider, Settings
from nl2sql.exceptions import ConfigurationError
from nl2sql.llm.base import LLMClient
from nl2sql.logging_config import get_logger

logger = get_logger(__name__)


def build_llm_client(settings: Settings) -> LLMClient | None:
    """Build the language model client described by ``settings``.

    Returns:
        A configured client, or ``None`` when the deterministic planner is selected.
    """
    provider = settings.resolve_provider()

    if provider is LLMProvider.DETERMINISTIC:
        logger.info("No model provider configured; using the deterministic planner")
        return None

    if provider is LLMProvider.ANTHROPIC:
        from nl2sql.llm.anthropic_client import AnthropicLLMClient

        if not settings.anthropic_api_key:
            raise ConfigurationError(
                "The Anthropic provider requires ANTHROPIC_API_KEY to be set"
            )
        model = settings.resolve_model()
        logger.info("Using the Anthropic provider with model %s", model)
        return AnthropicLLMClient(
            api_key=settings.anthropic_api_key,
            model=model,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    if provider is LLMProvider.OPENAI:
        from nl2sql.llm.openai_client import OpenAILLMClient

        if not settings.openai_api_key:
            raise ConfigurationError(
                "The OpenAI provider requires OPENAI_API_KEY to be set"
            )
        model = settings.resolve_model()
        logger.info("Using the OpenAI provider with model %s", model)
        return OpenAILLMClient(
            api_key=settings.openai_api_key,
            model=model,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    raise ConfigurationError(f"Unsupported language model provider: {provider}")
