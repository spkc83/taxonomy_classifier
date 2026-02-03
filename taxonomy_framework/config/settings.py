"""Configuration settings using Pydantic Settings."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support.
    
    Settings can be configured via:
    - Environment variables (e.g., LLM_PROVIDER=openai)
    - .env file in the project root
    
    Example:
        >>> from taxonomy_framework.config import Settings
        >>> settings = Settings()
        >>> print(settings.llm_provider)
        'openai'
    """
    
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    api_base_url: Optional[str] = None
    
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
