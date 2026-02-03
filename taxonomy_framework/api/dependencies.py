from functools import lru_cache
from typing import Any, Optional

from taxonomy_framework.config.settings import Settings


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_llm_client() -> Optional[Any]:
    return None
