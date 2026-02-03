"""Tests for provider base classes and capabilities."""

import pytest
from typing import Any, Dict, List

from taxonomy_framework.providers.base import (
    BaseLLMProvider,
    ProviderCapabilities,
    ToolCallResult,
)
from taxonomy_framework.providers.factory import ProviderFactory


class TestProviderCapabilities:

    def test_capabilities_dataclass_creation(self):
        caps = ProviderCapabilities(
            supports_json_mode=True,
            supports_tools=True,
            max_tokens=4096,
            supports_streaming=False,
        )
        
        assert caps.supports_json_mode is True
        assert caps.supports_tools is True
        assert caps.max_tokens == 4096
        assert caps.supports_streaming is False

    def test_capabilities_streaming_default(self):
        caps = ProviderCapabilities(
            supports_json_mode=True,
            supports_tools=False,
            max_tokens=2048,
        )
        
        assert caps.supports_streaming is False

    def test_capabilities_with_streaming_enabled(self):
        caps = ProviderCapabilities(
            supports_json_mode=True,
            supports_tools=True,
            max_tokens=8192,
            supports_streaming=True,
        )
        
        assert caps.supports_streaming is True


class TestToolCallResult:

    def test_tool_call_result_creation(self):
        result = ToolCallResult(
            name="classify",
            arguments={"category": "Tech", "confidence": 0.95},
        )
        
        assert result.name == "classify"
        assert result.arguments == {"category": "Tech", "confidence": 0.95}

    def test_tool_call_result_with_empty_arguments(self):
        result = ToolCallResult(name="no_args_tool", arguments={})
        
        assert result.name == "no_args_tool"
        assert result.arguments == {}


class MockProvider(BaseLLMProvider):
    """Test implementation of BaseLLMProvider for interface compliance testing."""

    def __init__(self, model: str = "mock-model"):
        self._model = model
        self._supports_tools = True

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_tool_calling(self) -> bool:
        return self._supports_tools

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        return {"mock": "response", "system": system_prompt[:20], "user": user_prompt[:20]}

    def call_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> ToolCallResult:
        if not self._supports_tools:
            raise NotImplementedError("Tool calling not supported")
        
        tool_name = tools[0]["function"]["name"] if tools else "unknown"
        return ToolCallResult(name=tool_name, arguments={"mock": True})


class TestBaseLLMProviderInterface:

    def test_mock_provider_inherits_from_base(self):
        provider = MockProvider()
        assert isinstance(provider, BaseLLMProvider)

    def test_model_name_property(self):
        provider = MockProvider(model="test-model-v1")
        assert provider.model_name == "test-model-v1"

    def test_supports_tool_calling_property(self):
        provider = MockProvider()
        assert provider.supports_tool_calling is True

    def test_generate_json_returns_dict(self):
        provider = MockProvider()
        result = provider.generate_json(
            system_prompt="You are a classifier",
            user_prompt="Classify this text",
        )
        
        assert isinstance(result, dict)
        assert "mock" in result

    def test_call_with_tools_returns_tool_call_result(self):
        provider = MockProvider()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "classify",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        
        result = provider.call_with_tools(
            system_prompt="System",
            user_prompt="User",
            tools=tools,
        )
        
        assert isinstance(result, ToolCallResult)
        assert result.name == "classify"


class TestConcreteProvidersImplementInterface:

    @pytest.fixture
    def registered_providers(self):
        return ProviderFactory.list_providers()

    def test_all_registered_providers_subclass_base(self, registered_providers):
        for provider_name in registered_providers:
            provider_class = ProviderFactory.get_provider_class(provider_name)
            assert issubclass(provider_class, BaseLLMProvider), (
                f"{provider_name} provider does not inherit from BaseLLMProvider"
            )

    def test_all_providers_have_model_name_property(self, registered_providers):
        for provider_name in registered_providers:
            provider_class = ProviderFactory.get_provider_class(provider_name)
            assert hasattr(provider_class, "model_name"), (
                f"{provider_name} is missing model_name property"
            )

    def test_all_providers_have_supports_tool_calling_property(self, registered_providers):
        for provider_name in registered_providers:
            provider_class = ProviderFactory.get_provider_class(provider_name)
            assert hasattr(provider_class, "supports_tool_calling"), (
                f"{provider_name} is missing supports_tool_calling property"
            )

    def test_all_providers_have_generate_json_method(self, registered_providers):
        for provider_name in registered_providers:
            provider_class = ProviderFactory.get_provider_class(provider_name)
            assert hasattr(provider_class, "generate_json"), (
                f"{provider_name} is missing generate_json method"
            )
            assert callable(getattr(provider_class, "generate_json")), (
                f"{provider_name}.generate_json is not callable"
            )

    def test_all_providers_have_call_with_tools_method(self, registered_providers):
        for provider_name in registered_providers:
            provider_class = ProviderFactory.get_provider_class(provider_name)
            assert hasattr(provider_class, "call_with_tools"), (
                f"{provider_name} is missing call_with_tools method"
            )
            assert callable(getattr(provider_class, "call_with_tools")), (
                f"{provider_name}.call_with_tools is not callable"
            )


class TestAbstractBaseClassEnforcement:

    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_incomplete_implementation_raises_error(self):
        class IncompleteProvider(BaseLLMProvider):
            @property
            def model_name(self) -> str:
                return "incomplete"
        
        with pytest.raises(TypeError) as exc_info:
            IncompleteProvider()
        
        error_msg = str(exc_info.value)
        assert "supports_tool_calling" in error_msg or "generate_json" in error_msg or "call_with_tools" in error_msg
