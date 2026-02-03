"""Factory for creating LLM provider instances."""

import logging
from typing import Any, Dict, List, Optional, Type

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating LLM provider instances.
    
    Supports auto-registration and aliases for provider names.
    
    Example:
        >>> from taxonomy_framework.providers import ProviderFactory
        >>> provider = ProviderFactory.create('openai', model='gpt-4o-mini')
        >>> print(type(provider).__name__)
        'OpenAIProvider'
    """
    
    _registry: Dict[str, Type[BaseLLMProvider]] = {}
    _aliases: Dict[str, str] = {}
    _initialized: bool = False
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """Register a provider class.
        
        Args:
            name: The canonical name for the provider.
            provider_class: The provider class to register.
        """
        cls._registry[name.lower()] = provider_class
        logger.debug(f"Registered provider: {name}")
    
    @classmethod
    def register_alias(cls, alias: str, canonical_name: str) -> None:
        """Register an alias for a provider.
        
        Args:
            alias: The alias name.
            canonical_name: The canonical provider name it maps to.
        """
        cls._aliases[alias.lower()] = canonical_name.lower()
        logger.debug(f"Registered alias: {alias} -> {canonical_name}")
    
    @classmethod
    def _resolve_name(cls, name: str) -> str:
        """Resolve an alias to the canonical provider name.
        
        Args:
            name: Provider name or alias.
            
        Returns:
            The canonical provider name.
        """
        name_lower = name.lower()
        return cls._aliases.get(name_lower, name_lower)
    
    @classmethod
    def create(cls, provider_type: str, **config: Any) -> BaseLLMProvider:
        """Create a provider instance by type name.
        
        Args:
            provider_type: Provider name or alias (e.g., 'openai', 'gpt', 'anthropic').
            **config: Configuration to pass to the provider constructor.
            
        Returns:
            An instance of the requested provider.
            
        Raises:
            ValueError: If the provider is not found in the registry.
            ImportError: If the provider's dependencies are not installed.
        """
        cls._ensure_initialized()
        
        resolved_name = cls._resolve_name(provider_type)
        
        if resolved_name not in cls._registry:
            available = cls.list_providers()
            raise ValueError(
                f"Unknown provider: '{provider_type}'. "
                f"Available providers: {available}"
            )
        
        provider_class = cls._registry[resolved_name]
        return provider_class(**config)
    
    @classmethod
    def get_provider_class(cls, provider_type: str) -> Type[BaseLLMProvider]:
        """Get the provider class without instantiating.
        
        Args:
            provider_type: Provider name or alias.
            
        Returns:
            The provider class.
            
        Raises:
            ValueError: If the provider is not found in the registry.
        """
        cls._ensure_initialized()
        
        resolved_name = cls._resolve_name(provider_type)
        
        if resolved_name not in cls._registry:
            available = cls.list_providers()
            raise ValueError(
                f"Unknown provider: '{provider_type}'. "
                f"Available providers: {available}"
            )
        
        return cls._registry[resolved_name]
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names.
        
        Returns:
            List of canonical provider names (not aliases).
        """
        cls._ensure_initialized()
        return sorted(cls._registry.keys())
    
    @classmethod
    def list_aliases(cls) -> Dict[str, str]:
        """List all registered aliases and their canonical names.
        
        Returns:
            Dictionary mapping aliases to canonical provider names.
        """
        cls._ensure_initialized()
        return dict(cls._aliases)
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure providers are registered on first use."""
        if not cls._initialized:
            cls._auto_register_providers()
            cls._initialized = True
    
    @classmethod
    def _auto_register_providers(cls) -> None:
        """Auto-register all known providers with lazy imports for missing dependencies."""
        provider_configs = [
            ("openai", "openai_provider", "OpenAIProvider", ["gpt", "chatgpt"]),
            ("anthropic", "anthropic_provider", "AnthropicProvider", ["claude"]),
            ("google", "google_provider", "GoogleProvider", ["gemini", "palm"]),
            ("cohere", "cohere_provider", "CohereProvider", ["command"]),
            ("ollama", "ollama_provider", "OllamaProvider", ["local"]),
            ("vllm", "vllm_provider", "VLLMProvider", []),
            ("huggingface_tgi", "huggingface_tgi", "HuggingFaceTGIProvider", ["hf_tgi", "tgi", "huggingface"]),
        ]
        
        for canonical_name, module_name, class_name, aliases in provider_configs:
            cls._try_register_provider(canonical_name, module_name, class_name, aliases)
    
    @classmethod
    def _try_register_provider(
        cls,
        canonical_name: str,
        module_name: str,
        class_name: str,
        aliases: List[str],
    ) -> None:
        """Try to register a provider, handling ImportError gracefully."""
        try:
            import importlib
            module = importlib.import_module(f".{module_name}", package="taxonomy_framework.providers")
            provider_class = getattr(module, class_name)
            
            cls.register(canonical_name, provider_class)
            
            for alias in aliases:
                cls.register_alias(alias, canonical_name)
                
        except ImportError as e:
            logger.debug(
                f"Provider '{canonical_name}' not available: {e}. "
                "Install the required SDK to enable this provider."
            )
        except Exception as e:
            logger.warning(f"Failed to register provider '{canonical_name}': {e}")
