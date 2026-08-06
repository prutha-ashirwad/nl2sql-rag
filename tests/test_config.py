"""Tests for configuration and provider selection."""

from __future__ import annotations

import pytest

from nl2sql.config import LLMProvider, Settings
from nl2sql.exceptions import ConfigurationError
from nl2sql.llm.factory import build_llm_client
from tests.conftest import build_test_settings as make_settings


class TestProviderResolution:
    def test_auto_falls_back_to_the_deterministic_planner(self) -> None:
        settings = make_settings(llm_provider=LLMProvider.AUTO)
        assert settings.resolve_provider() is LLMProvider.DETERMINISTIC

    def test_auto_selects_anthropic_when_its_key_is_present(self) -> None:
        settings = make_settings(
            llm_provider=LLMProvider.AUTO, anthropic_api_key="test-key"
        )
        assert settings.resolve_provider() is LLMProvider.ANTHROPIC

    def test_auto_selects_openai_when_only_its_key_is_present(self) -> None:
        settings = make_settings(
            llm_provider=LLMProvider.AUTO, openai_api_key="test-key"
        )
        assert settings.resolve_provider() is LLMProvider.OPENAI

    def test_an_explicit_provider_is_never_overridden(self) -> None:
        settings = make_settings(
            llm_provider=LLMProvider.DETERMINISTIC, anthropic_api_key="test-key"
        )
        assert settings.resolve_provider() is LLMProvider.DETERMINISTIC


class TestModelResolution:
    @pytest.mark.parametrize(
        ("provider", "expected"),
        [
            (LLMProvider.ANTHROPIC, "claude-opus-5"),
            (LLMProvider.OPENAI, "gpt-4o"),
        ],
    )
    def test_each_provider_supplies_its_own_default_model(
        self, provider: LLMProvider, expected: str
    ) -> None:
        settings = make_settings(llm_provider=provider)
        assert settings.resolve_model() == expected

    def test_an_explicit_model_is_used_as_given(self) -> None:
        settings = make_settings(
            llm_provider=LLMProvider.OPENAI, llm_model="gpt-4o-mini"
        )
        assert settings.resolve_model() == "gpt-4o-mini"

    def test_rejects_a_model_belonging_to_another_provider(self) -> None:
        # Configuring an OpenAI key while leaving an Anthropic model behind would
        # otherwise fail deep inside the vendor SDK with an opaque error.
        settings = make_settings(
            llm_provider=LLMProvider.OPENAI, llm_model="claude-opus-5"
        )
        with pytest.raises(ConfigurationError, match="looks like a anthropic model"):
            settings.resolve_model()

    def test_rejects_an_openai_model_on_the_anthropic_provider(self) -> None:
        settings = make_settings(
            llm_provider=LLMProvider.ANTHROPIC, llm_model="gpt-4o"
        )
        with pytest.raises(ConfigurationError, match="looks like a openai model"):
            settings.resolve_model()

    def test_unknown_model_names_are_left_alone(self) -> None:
        # A fine-tune or a self-hosted alias should not be second-guessed.
        settings = make_settings(
            llm_provider=LLMProvider.OPENAI, llm_model="my-org-finetune-v3"
        )
        assert settings.resolve_model() == "my-org-finetune-v3"


class TestClientFactory:
    def test_deterministic_provider_needs_no_client(self) -> None:
        settings = make_settings(llm_provider=LLMProvider.DETERMINISTIC)
        assert build_llm_client(settings) is None

    @pytest.mark.parametrize(
        "provider", [LLMProvider.ANTHROPIC, LLMProvider.OPENAI]
    )
    def test_a_provider_without_credentials_fails_clearly(
        self, provider: LLMProvider
    ) -> None:
        settings = make_settings(llm_provider=provider)
        with pytest.raises(ConfigurationError, match="API_KEY"):
            build_llm_client(settings)


class TestValidation:
    def test_rejects_an_unknown_log_level(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            make_settings(log_level="CHATTY")

    def test_rejects_a_missing_knowledge_base_path(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            Settings(_env_file=None, knowledge_base_path=tmp_path / "nope")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("retrieval_top_k", 0),
            ("lexical_weight", 1.5),
            ("max_repair_attempts", 99),
            ("query_timeout_seconds", 0),
        ],
    )
    def test_rejects_out_of_range_values(self, field: str, value: object) -> None:
        with pytest.raises(ValueError):
            make_settings(**{field: value})

    def test_defaults_are_usable_without_any_environment(self) -> None:
        settings = make_settings()
        assert settings.retrieval_top_k > 0
        assert settings.sql_dialect == "sqlite"
        assert settings.execute_queries is True
