"""LLM Provider abstractions for the taxonomy framework.

Providers are imported lazily to avoid requiring all SDKs to be installed.
Import specific providers directly when needed:
    from taxonomy_framework.providers.openai_provider import OpenAIProvider
"""

from .base import BaseLLMProvider, ProviderCapabilities, ToolCallResult, build_traversal_tools
from .factory import ProviderFactory

__all__ = [
    "BaseLLMProvider",
    "ProviderCapabilities",
    "ToolCallResult",
    "build_traversal_tools",
    "ProviderFactory",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "CohereProvider",
    "OllamaProvider",
    "VLLMProvider",
    "HuggingFaceTGIProvider",
]


def __getattr__(name: str):
    """Lazy import providers to avoid requiring all SDKs."""
    if name == "OpenAIProvider":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider
    elif name == "AnthropicProvider":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider
    elif name == "GoogleProvider":
        from .google_provider import GoogleProvider
        return GoogleProvider
    elif name == "CohereProvider":
        from .cohere_provider import CohereProvider
        return CohereProvider
    elif name == "OllamaProvider":
        from .ollama_provider import OllamaProvider
        return OllamaProvider
    elif name == "VLLMProvider":
        from .vllm_provider import VLLMProvider
        return VLLMProvider
    elif name == "HuggingFaceTGIProvider":
        from .huggingface_tgi import HuggingFaceTGIProvider
        return HuggingFaceTGIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
