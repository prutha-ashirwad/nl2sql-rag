"""Model-backed SQL generator."""

from __future__ import annotations

import re

from nl2sql.analysis.question_analyzer import QuestionAnalysis
from nl2sql.exceptions import LLMError
from nl2sql.generation.base import GenerationResult, SQLGenerator
from nl2sql.llm.base import LLMClient
from nl2sql.logging_config import get_logger
from nl2sql.prompts.templates import (
    INSUFFICIENT_CONTEXT_MARKER,
    build_generation_prompts,
    build_repair_prompts,
)
from nl2sql.retrieval.context_builder import RetrievedContext

logger = get_logger(__name__)

GENERATOR_NAME = "llm"

# Models occasionally wrap output in markdown despite instructions to the contrary.
_CODE_FENCE_PATTERN = re.compile(
    r"^\s*```(?:sql)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE
)


def extract_sql(raw_response: str) -> str:
    """Strip markdown fences and trailing punctuation from a model response."""
    text = raw_response.strip()

    fenced = _CODE_FENCE_PATTERN.match(text)
    if fenced:
        text = fenced.group("body").strip()

    return text.rstrip().rstrip(";").rstrip()


class LLMSQLGenerator:
    """Generates SQL with a language model, grounded on retrieved context."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        dialect: str = "sqlite",
        fallback: SQLGenerator | None = None,
    ) -> None:
        self._client = llm_client
        self._dialect = dialect
        self._fallback = fallback

    @property
    def name(self) -> str:
        """Short identifier used in traces and result metadata."""
        return GENERATOR_NAME

    def generate(
        self, context: RetrievedContext, analysis: QuestionAnalysis
    ) -> GenerationResult:
        """Produce a first-pass query for the question in ``context``."""
        if context.is_empty:
            return GenerationResult(
                sql=None,
                generator=self.name,
                insufficient_context=True,
                explanation="No tables in the Knowledge Base matched this question.",
            )

        system_prompt, user_prompt = build_generation_prompts(
            context, analysis, self._dialect
        )
        return self._complete(system_prompt, user_prompt, context, analysis)

    def repair(
        self,
        context: RetrievedContext,
        analysis: QuestionAnalysis,
        sql: str,
        errors: list[str],
    ) -> GenerationResult:
        """Ask the model to correct a query that failed validation."""
        system_prompt, user_prompt = build_repair_prompts(
            context, sql, errors, self._dialect
        )
        return self._complete(system_prompt, user_prompt, context, analysis)

    def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        context: RetrievedContext,
        analysis: QuestionAnalysis,
    ) -> GenerationResult:
        """Call the provider and normalise the response into a result."""
        try:
            response = self._client.complete(system_prompt, user_prompt)
        except LLMError as exc:
            logger.warning("Model provider failed: %s", exc)
            if self._fallback is None:
                return GenerationResult(
                    sql=None,
                    generator=self.name,
                    explanation=f"The model provider was unavailable: {exc}",
                )
            logger.info("Falling back to the %s generator", self._fallback.name)
            return self._fallback.generate(context, analysis)

        if response.text.strip().upper().startswith(INSUFFICIENT_CONTEXT_MARKER):
            explanation = response.text.strip()[len(INSUFFICIENT_CONTEXT_MARKER) :]
            return GenerationResult(
                sql=None,
                generator=self.name,
                insufficient_context=True,
                explanation=explanation.strip(" :.-") or "Insufficient context.",
                tokens_used=response.total_tokens,
            )

        return GenerationResult(
            sql=extract_sql(response.text),
            generator=self.name,
            tokens_used=response.total_tokens,
            metadata={"model": response.model},
        )
