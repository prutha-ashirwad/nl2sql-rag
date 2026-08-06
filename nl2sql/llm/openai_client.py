"""OpenAI backend."""

from __future__ import annotations

from typing import Any

from nl2sql.exceptions import ConfigurationError, LLMError
from nl2sql.llm.base import LLMResponse
from nl2sql.logging_config import get_logger

logger = get_logger(__name__)


class OpenAILLMClient:
    """Generates completions using the OpenAI chat completions API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_tokens: int = 4096,
        timeout_seconds: float = 90.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError("An API key is required for the OpenAI provider")

        self._model = model
        self._max_tokens = max_tokens
        self._client = self._build_client(api_key, timeout_seconds)

    @staticmethod
    def _build_client(api_key: str, timeout_seconds: float) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ConfigurationError(
                "The 'openai' package is required for this provider. "
                "Install it with: pip install openai"
            ) from exc

        return OpenAI(api_key=api_key, timeout=timeout_seconds)

    @property
    def model_name(self) -> str:
        """Identifier of the configured model."""
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a completion for the given prompts."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise LLMError("The model returned an empty response")

        usage = response.usage
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            metadata={"finish_reason": str(response.choices[0].finish_reason)},
        )
