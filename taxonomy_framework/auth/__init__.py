"""Authentication module for the Taxonomy Framework API."""

from taxonomy_framework.auth.api_key import get_api_key
from taxonomy_framework.auth.models import Token, TokenData, User
from taxonomy_framework.auth.oauth2 import (
    create_access_token,
    get_current_active_user,
    get_current_user,
    oauth2_scheme,
    verify_password,
)
from taxonomy_framework.auth.sso import (
    SSOConfig,
    SSOError,
    SSONotConfiguredError,
    SSOTokenValidationError,
    SSOUser,
    get_oidc_config,
    get_sso_config,
    is_sso_enabled,
    validate_sso_token,
)

__all__ = [
    "SSOConfig",
    "SSOError",
    "SSONotConfiguredError",
    "SSOTokenValidationError",
    "SSOUser",
    "Token",
    "TokenData",
    "User",
    "create_access_token",
    "get_api_key",
    "get_current_active_user",
    "get_current_user",
    "get_oidc_config",
    "get_sso_config",
    "is_sso_enabled",
    "oauth2_scheme",
    "validate_sso_token",
    "verify_password",
]
