# FastAPI Service

Production-ready REST API for taxonomy classification.

## Quick Start

### Start the Server

```bash
# Development
uvicorn taxonomy_framework.api.main:app --reload --port 8000

# Production
uvicorn taxonomy_framework.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Endpoints

### Health Check

```bash
# Root health
curl http://localhost:8000/health

# API health
curl http://localhost:8000/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Classification

#### Single Text Classification

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "text": "iPhone 15 Pro with titanium design",
    "options": {
      "top_k": 3,
      "include_alternatives": true
    }
  }'
```

**Response:**
```json
{
  "predicted_category": "Smartphones",
  "predicted_path": "Root > Electronics > Smartphones",
  "confidence": 0.92,
  "path_confidence": [0.98, 0.95, 0.92],
  "alternatives": [
    {"path": "Root > Electronics > Tablets", "confidence": 0.65},
    {"path": "Root > Electronics > Laptops", "confidence": 0.42}
  ]
}
```

#### Batch Classification

```bash
curl -X POST http://localhost:8000/api/v1/classify/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "texts": [
      "iPhone 15 Pro",
      "Nike Air Jordan sneakers",
      "The Great Gatsby book"
    ],
    "options": {
      "top_k": 1
    }
  }'
```

**Response:**
```json
{
  "results": [
    {
      "text": "iPhone 15 Pro",
      "predicted_category": "Smartphones",
      "predicted_path": "Root > Electronics > Smartphones",
      "confidence": 0.92
    },
    {
      "text": "Nike Air Jordan sneakers",
      "predicted_category": "Shoes",
      "predicted_path": "Root > Clothing > Shoes",
      "confidence": 0.88
    },
    {
      "text": "The Great Gatsby book",
      "predicted_category": "Fiction",
      "predicted_path": "Root > Books > Fiction",
      "confidence": 0.95
    }
  ],
  "processing_time_ms": 245
}
```

### Taxonomy

```bash
curl http://localhost:8000/api/v1/taxonomy \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "root": {
    "name": "Root",
    "children": [
      {
        "name": "Electronics",
        "children": [
          {"name": "Smartphones", "children": []},
          {"name": "Laptops", "children": []}
        ]
      }
    ]
  },
  "total_nodes": 25,
  "max_depth": 3
}
```

## Authentication

The API supports multiple authentication methods:

### API Key Authentication

**Header:**
```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/classify
```

**Query Parameter:**
```bash
curl "http://localhost:8000/api/v1/classify?api_key=your-api-key"
```

### OAuth2 (JWT) Authentication

**Get Token:**
```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=user@example.com&password=secret"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Use Token:**
```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  http://localhost:8000/api/v1/classify
```

### SSO/OIDC Authentication

**Initiate Login:**
```bash
curl http://localhost:8000/auth/sso/login?provider=google
# Redirects to SSO provider
```

**Callback:**
```bash
# Handled automatically by SSO provider redirect
GET /auth/sso/callback?code=...&state=...
```

## Configuration

### Environment Variables

```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
DEBUG=false

# Authentication
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
API_KEYS=key1,key2,key3  # Comma-separated valid API keys

# SSO/OIDC (optional)
SSO_ENABLED=true
SSO_PROVIDER_URL=https://accounts.google.com
SSO_CLIENT_ID=your-client-id
SSO_CLIENT_SECRET=your-client-secret
SSO_REDIRECT_URI=http://localhost:8000/auth/sso/callback

# LLM Provider
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# CORS
CORS_ORIGINS=http://localhost:3000,https://myapp.com
```

### CORS Configuration

The API has CORS enabled by default. Configure allowed origins:

```python
# In taxonomy_framework/api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Request/Response Schemas

### ClassifyRequest

```python
class ClassifyRequest(BaseModel):
    text: str                           # Text to classify
    taxonomy_id: Optional[str] = None   # Optional taxonomy ID
    options: Optional[ClassifyOptions] = None

class ClassifyOptions(BaseModel):
    top_k: int = 3                      # Number of alternatives
    confidence_threshold: float = 0.5   # Minimum confidence
    include_alternatives: bool = True   # Include alternative categories
    max_depth: Optional[int] = None     # Max traversal depth
```

### ClassifyResponse

```python
class ClassifyResponse(BaseModel):
    predicted_category: str
    predicted_path: str
    confidence: float
    path_confidence: List[float]
    alternatives: Optional[List[AlternativeCategory]] = None
    abstained: bool = False
    abstain_reason: Optional[str] = None

class AlternativeCategory(BaseModel):
    path: str
    confidence: float
```

### BatchClassifyRequest/Response

```python
class BatchClassifyRequest(BaseModel):
    texts: List[str]
    options: Optional[ClassifyOptions] = None

class BatchClassifyResponse(BaseModel):
    results: List[ClassifyResponse]
    processing_time_ms: int
```

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message",
  "error_code": "CLASSIFICATION_FAILED",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 401 | Unauthorized (invalid/missing auth) |
| 403 | Forbidden (insufficient permissions) |
| 422 | Validation Error |
| 500 | Internal Server Error |

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "taxonomy_framework.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLM_PROVIDER=openai
      - LLM_MODEL=gpt-4o-mini
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - API_KEYS=${API_KEYS}
    restart: unless-stopped
```

### Run with Docker

```bash
# Build
docker build -t taxonomy-classifier .

# Run
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e JWT_SECRET_KEY=secret \
  -e API_KEYS=key1,key2 \
  taxonomy-classifier
```

## Testing

### Using pytest

```bash
pytest tests/test_api/ -v
```

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Classify (with API key)
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key" \
  -d '{"text": "iPhone 15 Pro"}'
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Classify
response = requests.post(
    "http://localhost:8000/api/v1/classify",
    headers={"X-API-Key": "your-api-key"},
    json={"text": "iPhone 15 Pro"}
)
print(response.json())
```

## Rate Limiting

Configure rate limiting in production:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/classify")
@limiter.limit("100/minute")
async def classify(request: ClassifyRequest):
    ...
```

## Monitoring

### Prometheus Metrics

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Access metrics at: http://localhost:8000/metrics

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

## Performance Tips

1. **Use batch endpoints** for multiple classifications
2. **Enable connection pooling** for database/cache
3. **Use async endpoints** for I/O-bound operations
4. **Cache taxonomy** to avoid reloading
5. **Use workers** in production (`--workers 4`)
