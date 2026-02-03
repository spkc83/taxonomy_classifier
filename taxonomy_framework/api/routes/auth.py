import secrets
from datetime import timedelta
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from taxonomy_framework.auth.models import Token, User
from taxonomy_framework.auth.oauth2 import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    FAKE_USERS_DB,
    authenticate_user,
    create_access_token,
    get_current_active_user,
)
from taxonomy_framework.auth.sso import (
    SSOError,
    SSONotConfiguredError,
    build_authorization_url,
    exchange_code_for_tokens,
    get_oidc_config,
    is_sso_enabled,
    validate_sso_token,
)

router = APIRouter(tags=["auth"])

_sso_state_store: dict[str, str] = {}


class SSOStatusResponse(BaseModel):
    enabled: bool
    message: str


class SSOCallbackResponse(BaseModel):
    access_token: str
    token_type: str
    email: Optional[str] = None
    name: Optional[str] = None
    sub: str


@router.post("/auth/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = authenticate_user(FAKE_USERS_DB, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@router.get("/api/v1/me", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    return current_user


@router.get("/auth/sso/status", response_model=SSOStatusResponse)
async def sso_status() -> SSOStatusResponse:
    enabled = is_sso_enabled()
    message = "SSO is enabled and configured" if enabled else "SSO is not enabled"
    return SSOStatusResponse(enabled=enabled, message=message)


@router.get("/auth/sso/login")
async def sso_login() -> RedirectResponse:
    if not is_sso_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SSO is not enabled. Configure SSO_ENABLED and required environment variables.",
        )

    try:
        oidc_config = await get_oidc_config()
    except SSONotConfiguredError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    except SSOError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to fetch OIDC configuration: {e}",
        ) from e

    state = secrets.token_urlsafe(32)
    _sso_state_store[state] = state

    query_params = build_authorization_url(state=state)
    authorization_url = f"{oidc_config.authorization_endpoint}?{query_params}"

    return RedirectResponse(url=authorization_url, status_code=status.HTTP_302_FOUND)


@router.get("/auth/sso/callback", response_model=SSOCallbackResponse)
async def sso_callback(
    code: str = Query(..., description="Authorization code from SSO provider"),
    state: str = Query(..., description="State parameter for CSRF protection"),
    error: Optional[str] = Query(None, description="Error from SSO provider"),
    error_description: Optional[str] = Query(None, description="Error description"),
) -> SSOCallbackResponse:
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_description or error,
        )

    if not is_sso_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SSO is not enabled.",
        )

    if state not in _sso_state_store:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid state parameter. Possible CSRF attack.",
        )
    del _sso_state_store[state]

    try:
        tokens = await exchange_code_for_tokens(code)
    except SSOError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to exchange authorization code: {e}",
        ) from e

    id_token = tokens.get("id_token")
    if not id_token:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="SSO provider did not return an ID token",
        )

    try:
        sso_user = await validate_sso_token(id_token)
    except SSOError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid SSO token: {e}",
        ) from e

    return SSOCallbackResponse(
        access_token=tokens.get("access_token", id_token),
        token_type="bearer",
        email=sso_user.email,
        name=sso_user.name,
        sub=sso_user.sub,
    )
