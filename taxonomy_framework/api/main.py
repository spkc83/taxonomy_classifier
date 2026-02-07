from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from taxonomy_framework.api.routes import auth_router, classify_router, health_router, taxonomy_router

app = FastAPI(
    title="Taxonomy Classifier API",
    description="API for hierarchical text classification",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_v1 = FastAPI(title="Taxonomy Classifier API v1")
api_v1.include_router(health_router)
api_v1.include_router(classify_router)
api_v1.include_router(taxonomy_router)

app.mount("/api/v1", api_v1)
app.include_router(health_router)
app.include_router(auth_router)
