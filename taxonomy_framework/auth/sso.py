"""SSO/OIDC authentication module for the Taxonomy Framework API.

Provides generic OpenID Connect (OIDC) integration supporting common identity
providers like Auth0, Okta, Keycloak, and Google.

Configuration is done via environment variables:
    - SSO_ENABLED: Enable/disable SSO (default: false)
    - SSO_PROVIDER_URL: OIDC discovery URL (e.g., https://auth.example.com/.well-known/openid-configuration)
    - SSO_CLIENT_ID: OAuth2 client ID
    - SSO_CLIENT_SECRET: OAuth2 client secret
    - SSO_REDIRECT_URI: Callback URL after authentication
    - SSO_SCOPES: Space-separated scopes (default: "openid profile email")
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError


@dataclass
class SSOConfig:
    """SSO configuration loaded from environment variables."""

    enabled: bool
    provider_url: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: str

    @classmethod
    def from_env(cls) -> "SSOConfig":
        """Create SSOConfig from environment variables."""
        return cls(
            enabled=os.getenv("SSO_ENABLED", "false").lower() in ("true", "1", "yes"),
            provider_url=os.getenv("SSO_PROVIDER_URL", ""),
            client_id=os.getenv("SSO_CLIENT_ID", ""),
            client_secret=os.getenv("SSO_CLIENT_SECRET", ""),
            redirect_uri=os.getenv("SSO_REDIRECT_URI", ""),
            scopes=os.getenv("SSO_SCOPES", "openid profile email"),
        )


@dataclass
class OIDCConfig:
    """OpenID Connect discovery document configuration."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    jwks_uri: str
    end_session_endpoint: Optional[str] = None


@dataclass
class SSOUser:
    """User information extracted from SSO token claims."""

    sub: str
    email: Optional[str] = None
    name: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    picture: Optional[str] = None
    email_verified: Optional[bool] = None
    raw_claims: Optional[dict[str, Any]] = None


class SSOError(Exception):
    """Base exception for SSO-related errors."""

    pass


class SSONotConfiguredError(SSOError):
    """Raised when SSO is accessed but not properly configured."""

    pass


class SSOTokenValidationError(SSOError):
    """Raised when SSO token validation fails."""

    pass


def is_sso_enabled() -> bool:
    """Check if SSO is enabled and properly configured.

    SSO is considered enabled if:
    - SSO_ENABLED is set to true/1/yes
    - Required configuration (provider_url, client_id, client_secret) is present

    Returns:
        True if SSO is enabled and configured, False otherwise.
    """
    config = get_sso_config()
    if not config.enabled:
        return False

    required = [config.provider_url, config.client_id, config.client_secret]
    return all(required)


@lru_cache(maxsize=1)
def get_sso_config() -> SSOConfig:
    """Get SSO configuration from environment.

    Configuration is cached for performance.

    Returns:
        SSOConfig instance with values from environment variables.
    """
    return SSOConfig.from_env()


def clear_sso_config_cache() -> None:
    """Clear the SSO configuration cache.

    Useful for testing or when configuration changes.
    """
    get_sso_config.cache_clear()
    get_oidc_config.cache_clear()
    _get_jwks.cache_clear()


async def get_oidc_config() -> OIDCConfig:
    """Fetch and parse the OIDC discovery document.

    The discovery document is fetched from the SSO_PROVIDER_URL.

    Returns:
        OIDCConfig with endpoints from the discovery document.

    Raises:
        SSONotConfiguredError: If SSO is not enabled or configured.
        SSOError: If the discovery document cannot be fetched or parsed.
    """
    sso_config = get_sso_config()

    if not is_sso_enabled():
        raise SSONotConfiguredError(
            "SSO is not enabled. Set SSO_ENABLED=true and provide required configuration."
        )

    provider_url = sso_config.provider_url
    if not provider_url.endswith("/.well-known/openid-configuration"):
        provider_url = provider_url.rstrip("/") + "/.well-known/openid-configuration"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(provider_url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        raise SSOError(f"Failed to fetch OIDC discovery document: {e}") from e
    except Exception as e:
        raise SSOError(f"Failed to parse OIDC discovery document: {e}") from e

    try:
        return OIDCConfig(
            issuer=data["issuer"],
            authorization_endpoint=data["authorization_endpoint"],
            token_endpoint=data["token_endpoint"],
            userinfo_endpoint=data["userinfo_endpoint"],
            jwks_uri=data["jwks_uri"],
            end_session_endpoint=data.get("end_session_endpoint"),
        )
    except KeyError as e:
        raise SSOError(f"Missing required field in OIDC discovery document: {e}") from e


async def _get_jwks(jwks_uri: str) -> dict[str, Any]:
    """Fetch JWKS (JSON Web Key Set) from the SSO provider.

    Args:
        jwks_uri: URL to fetch JWKS from.

    Returns:
        JWKS data as a dictionary.

    Raises:
        SSOError: If JWKS cannot be fetched.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(jwks_uri, timeout=10.0)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise SSOError(f"Failed to fetch JWKS: {e}") from e


def _get_signing_key(jwks: dict[str, Any], kid: str) -> dict[str, Any]:
    """Find the signing key in JWKS matching the key ID.

    Args:
        jwks: JWKS data.
        kid: Key ID to find.

    Returns:
        The matching key from JWKS.

    Raises:
        SSOTokenValidationError: If no matching key is found.
    """
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key

    raise SSOTokenValidationError(f"Unable to find signing key with kid: {kid}")


async def validate_sso_token(token: str, verify_audience: bool = True) -> SSOUser:
    """Validate an SSO token and extract user information.

    Validates the token signature using JWKS from the SSO provider
    and verifies standard JWT claims (issuer, audience, expiration).

    Args:
        token: The JWT token to validate.
        verify_audience: Whether to verify the audience claim (default: True).

    Returns:
        SSOUser with user information from the token claims.

    Raises:
        SSONotConfiguredError: If SSO is not enabled.
        SSOTokenValidationError: If token validation fails.
    """
    if not is_sso_enabled():
        raise SSONotConfiguredError("SSO is not enabled.")

    sso_config = get_sso_config()
    oidc_config = await get_oidc_config()

    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        if not kid:
            raise SSOTokenValidationError("Token header missing 'kid' claim")
    except JWTError as e:
        raise SSOTokenValidationError(f"Invalid token header: {e}") from e

    jwks = await _get_jwks(oidc_config.jwks_uri)
    signing_key = _get_signing_key(jwks, kid)

    try:
        options = {
            "verify_aud": verify_audience,
            "verify_iss": True,
            "verify_exp": True,
        }

        claims = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
            audience=sso_config.client_id if verify_audience else None,
            issuer=oidc_config.issuer,
            options=options,
        )
    except ExpiredSignatureError:
        raise SSOTokenValidationError("Token has expired")
    except JWTError as e:
        raise SSOTokenValidationError(f"Token validation failed: {e}") from e

    return SSOUser(
        sub=claims["sub"],
        email=claims.get("email"),
        name=claims.get("name"),
        given_name=claims.get("given_name"),
        family_name=claims.get("family_name"),
        picture=claims.get("picture"),
        email_verified=claims.get("email_verified"),
        raw_claims=claims,
    )


def build_authorization_url(state: str, nonce: Optional[str] = None) -> str:
    """Build the SSO authorization URL for redirecting users to login.

    Args:
        state: Random state parameter for CSRF protection.
        nonce: Optional nonce for replay protection.

    Returns:
        The full authorization URL to redirect users to.

    Raises:
        SSONotConfiguredError: If SSO is not enabled.
    """
    if not is_sso_enabled():
        raise SSONotConfiguredError("SSO is not enabled.")

    sso_config = get_sso_config()

    params = {
        "client_id": sso_config.client_id,
        "redirect_uri": sso_config.redirect_uri,
        "response_type": "code",
        "scope": sso_config.scopes,
        "state": state,
    }

    if nonce:
        params["nonce"] = nonce

    return urlencode(params)


async def exchange_code_for_tokens(
    code: str,
    code_verifier: Optional[str] = None,
) -> dict[str, Any]:
    """Exchange an authorization code for tokens.

    Args:
        code: The authorization code from the callback.
        code_verifier: Optional PKCE code verifier.

    Returns:
        Token response containing access_token, id_token, etc.

    Raises:
        SSONotConfiguredError: If SSO is not enabled.
        SSOError: If token exchange fails.
    """
    if not is_sso_enabled():
        raise SSONotConfiguredError("SSO is not enabled.")

    sso_config = get_sso_config()
    oidc_config = await get_oidc_config()

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": sso_config.redirect_uri,
        "client_id": sso_config.client_id,
        "client_secret": sso_config.client_secret,
    }

    if code_verifier:
        data["code_verifier"] = code_verifier

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                oidc_config.token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        try:
            error_data = e.response.json()
            error_msg = error_data.get("error_description", error_data.get("error", str(e)))
        except Exception:
            error_msg = str(e)
        raise SSOError(f"Token exchange failed: {error_msg}") from e
    except httpx.HTTPError as e:
        raise SSOError(f"Token exchange failed: {e}") from e


async def get_userinfo(access_token: str) -> dict[str, Any]:
    """Fetch user information from the SSO provider's userinfo endpoint.

    Args:
        access_token: The access token to use for authentication.

    Returns:
        User information from the userinfo endpoint.

    Raises:
        SSONotConfiguredError: If SSO is not enabled.
        SSOError: If userinfo request fails.
    """
    if not is_sso_enabled():
        raise SSONotConfiguredError("SSO is not enabled.")

    oidc_config = await get_oidc_config()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                oidc_config.userinfo_endpoint,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        raise SSOError(f"Failed to fetch user info: {e}") from e
