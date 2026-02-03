"""Tests for ProviderFactory."""

import pytest
from unittest.mock import patch, MagicMock

from taxonomy_framework.providers.factory import ProviderFactory
from taxonomy_framework.providers.base import BaseLLMProvider


class TestProviderFactoryListProviders:

    def test_list_providers_returns_list(self):
        providers = ProviderFactory.list_providers()
        assert isinstance(providers, list)

    def test_list_providers_contains_expected_providers(self):
        providers = ProviderFactory.list_providers()
        expected_core = {"openai", "anthropic", "google", "cohere", "ollama"}
        available = set(providers)
        assert available.intersection(expected_core), (
            f"Expected at least one of {expected_core} but got {providers}"
        )

    def test_list_providers_is_sorted(self):
        providers = ProviderFactory.list_providers()
        assert providers == sorted(providers)

    def test_list_providers_returns_lowercase_names(self):
        providers = ProviderFactory.list_providers()
        for name in providers:
            assert name == name.lower()


class TestProviderFactoryGetProviderClass:

    def test_get_provider_class_for_openai(self):
        provider_class = ProviderFactory.get_provider_class("openai")
        assert issubclass(provider_class, BaseLLMProvider)
        assert provider_class.__name__ == "OpenAIProvider"

    def test_get_provider_class_for_anthropic(self):
        provider_class = ProviderFactory.get_provider_class("anthropic")
        assert issubclass(provider_class, BaseLLMProvider)
        assert provider_class.__name__ == "AnthropicProvider"

    def test_get_provider_class_is_case_insensitive(self):
        provider_lower = ProviderFactory.get_provider_class("openai")
        provider_upper = ProviderFactory.get_provider_class("OPENAI")
        provider_mixed = ProviderFactory.get_provider_class("OpenAI")
        assert provider_lower == provider_upper == provider_mixed

    def test_get_provider_class_unknown_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            ProviderFactory.get_provider_class("nonexistent_provider")
        
        assert "Unknown provider" in str(exc_info.value)
        assert "nonexistent_provider" in str(exc_info.value)


class TestProviderFactoryCreate:

    @patch("taxonomy_framework.providers.openai_provider.OpenAI")
    def test_create_openai_provider_with_mocked_client(self, mock_openai_class):
        mock_openai_class.return_value = MagicMock()
        
        provider = ProviderFactory.create("openai", api_key="test-key", model="gpt-4")
        
        assert isinstance(provider, BaseLLMProvider)
        assert provider.model_name == "gpt-4"
        mock_openai_class.assert_called_once()

    @patch("taxonomy_framework.providers.openai_provider.OpenAI")
    def test_create_with_alias_gpt(self, mock_openai_class):
        mock_openai_class.return_value = MagicMock()
        
        provider = ProviderFactory.create("gpt", api_key="test-key")
        
        assert isinstance(provider, BaseLLMProvider)
        assert provider.__class__.__name__ == "OpenAIProvider"

    def test_create_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError) as exc_info:
            ProviderFactory.create("unknown_provider_xyz")
        
        assert "Unknown provider" in str(exc_info.value)
        assert "Available providers" in str(exc_info.value)


class TestProviderFactoryAliases:

    def test_list_aliases_returns_dict(self):
        aliases = ProviderFactory.list_aliases()
        assert isinstance(aliases, dict)

    def test_gpt_alias_maps_to_openai(self):
        aliases = ProviderFactory.list_aliases()
        assert aliases.get("gpt") == "openai"

    def test_claude_alias_maps_to_anthropic(self):
        aliases = ProviderFactory.list_aliases()
        assert aliases.get("claude") == "anthropic"

    def test_gemini_alias_maps_to_google(self):
        aliases = ProviderFactory.list_aliases()
        assert aliases.get("gemini") == "google"

    @patch("taxonomy_framework.providers.openai_provider.OpenAI")
    def test_resolve_alias_in_create(self, mock_openai_class):
        mock_openai_class.return_value = MagicMock()
        
        provider_direct = ProviderFactory.create("openai", api_key="test-key")
        provider_alias = ProviderFactory.create("gpt", api_key="test-key")
        
        assert type(provider_direct) == type(provider_alias)


class TestProviderFactoryRegistration:

    def test_register_custom_provider(self):
        class CustomTestProvider(BaseLLMProvider):
            @property
            def model_name(self) -> str:
                return "custom-model"
            
            @property
            def supports_tool_calling(self) -> bool:
                return False
            
            def generate_json(self, system_prompt, user_prompt, max_retries=3):
                return {}
            
            def call_with_tools(self, system_prompt, user_prompt, tools, tool_choice="auto"):
                raise NotImplementedError("Tool calling not supported")
        
        original_registry = dict(ProviderFactory._registry)
        
        try:
            ProviderFactory.register("custom_test", CustomTestProvider)
            assert "custom_test" in ProviderFactory.list_providers()
            
            provider_class = ProviderFactory.get_provider_class("custom_test")
            assert provider_class == CustomTestProvider
        finally:
            ProviderFactory._registry = original_registry

    def test_register_alias_for_existing_provider(self):
        original_aliases = dict(ProviderFactory._aliases)
        
        try:
            ProviderFactory.register_alias("test_alias_openai", "openai")
            aliases = ProviderFactory.list_aliases()
            assert aliases.get("test_alias_openai") == "openai"
        finally:
            ProviderFactory._aliases = original_aliases
