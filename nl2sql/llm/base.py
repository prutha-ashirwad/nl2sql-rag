"""Provider-agnostic language model interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A completion returned by a language model provider."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Combined prompt and completion token usage."""
        return self.input_tokens + self.output_tokens


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract every language model backend must satisfy."""

    @property
    def model_name(self) -> str:
        """Identifier of the model this client talks to."""
        ...

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a single completion for the given prompts.

        Raises:
            LLMError: if the provider is unreachable or returns an error.
        """
        ...
