"""Anthropic Claude backend."""

from __future__ import annotations

from typing import Any

from nl2sql.exceptions import ConfigurationError, LLMError
from nl2sql.llm.base import LLMResponse
from nl2sql.logging_config import get_logger

logger = get_logger(__name__)


class AnthropicLLMClient:
    """Generates completions using the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_tokens: int = 4096,
        timeout_seconds: float = 90.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError("An API key is required for the Anthropic provider")

        self._model = model
        self._max_tokens = max_tokens
        self._client = self._build_client(api_key, timeout_seconds)

    @staticmethod
    def _build_client(api_key: str, timeout_seconds: float) -> Any:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ConfigurationError(
                "The 'anthropic' package is required for this provider. "
                "Install it with: pip install anthropic"
            ) from exc

        return anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)

    @property
    def model_name(self) -> str:
        """Identifier of the configured model."""
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a completion for the given prompts."""
        try:
            # Streamed: adaptive thinking can exceed the non-streaming timeout.
            with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        # Stable across repair attempts, so retries hit the cache.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                message = stream.get_final_message()
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        if message.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request")

        text = "".join(
            block.text for block in message.content if block.type == "text"
        ).strip()

        if not text:
            raise LLMError("The model returned an empty response")

        return LLMResponse(
            text=text,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            metadata={"stop_reason": str(message.stop_reason)},
        )
