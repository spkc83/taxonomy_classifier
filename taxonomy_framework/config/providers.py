"""Provider configuration dataclass."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    
    provider_type: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
