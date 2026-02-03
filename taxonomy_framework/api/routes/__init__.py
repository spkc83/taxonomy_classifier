from taxonomy_framework.api.routes.auth import router as auth_router
from taxonomy_framework.api.routes.classify import router as classify_router
from taxonomy_framework.api.routes.classify import taxonomy_router
from taxonomy_framework.api.routes.health import router as health_router

__all__ = ["health_router", "classify_router", "taxonomy_router", "auth_router"]
