"""API Key authentication for the Taxonomy Framework API."""

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


def get_valid_api_keys() -> set:
    """Get valid API keys from environment.

    Reads API keys from the API_KEYS environment variable.
    Keys should be comma-separated.
    Falls back to a default development key if not set.

    Returns:
        Set of valid API keys.
    """
    keys = os.getenv("API_KEYS", "dev-api-key-12345")
    return set(k.strip() for k in keys.split(",") if k.strip())


async def get_api_key(
    header_key: str = Security(api_key_header),
    query_key: str = Security(api_key_query),
) -> str:
    """Validate API key from header or query parameter.

    Checks for API key in the following order:
    1. X-API-Key header
    2. api_key query parameter

    Args:
        header_key: API key from X-API-Key header.
        query_key: API key from api_key query parameter.

    Returns:
        The validated API key.

    Raises:
        HTTPException: If no valid API key is provided.
    """
    valid_keys = get_valid_api_keys()
    api_key = header_key or query_key

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Provide via X-API-Key header or api_key query parameter.",
        )

    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
