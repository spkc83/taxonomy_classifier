# Taxonomy Classifier

A production-ready framework for hierarchical text classification using a hybrid approach that combines semantic embeddings with LLM-powered constrained navigation.

## Overview

This framework addresses a fundamental challenge in taxonomy classification: **general classification and fine-grained classification fail in different ways**. Rather than forcing one mechanism to do both, this system switches techniques mid-stream:

1. **Semantic Recall (Embeddings)**: Routes input to the right high-level region with high recall
2. **Constrained Traversal (LLM)**: Navigates to the exact leaf with guaranteed validity
3. **Sibling Contrast**: Disambiguates confusing siblings when needed
4. **Abstain Path**: Handles ambiguity gracefully instead of forcing incorrect predictions

**Result**: No hallucinated categories. No invalid labels. Scalable across flat, hierarchical, and fast-changing taxonomies.

## Key Features

- **Multi-Provider LLM Support**: OpenAI, Anthropic, Google, Cohere, Ollama, vLLM, HuggingFace TGI
- **FastAPI REST Service**: Production-ready API with Swagger documentation
- **Multiple Auth Methods**: OAuth2, API Key, and SSO/OIDC
- **SetFit Fine-tuning**: Few-shot learning for domain-specific improvements
- **Hierarchical Metrics**: Proper evaluation accounting for taxonomy structure
- **Memory Efficient**: Optimized for resource-constrained environments

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### With All Optional Dependencies

```bash
pip install -r requirements.txt

# Install provider-specific SDKs as needed
pip install openai anthropic google-generativeai cohere
```

## Quick Start

### 1. Configure Environment

```bash
cp config/.env.example .env
# Edit .env with your API keys
```

Example `.env`:
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your-key-here
```

### 2. Run the API Server

```bash
uvicorn taxonomy_framework.api.main:app --reload
```

Access the Swagger UI at http://localhost:8000/docs

### 3. Classify Text

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"text": "iPhone 15 Pro with titanium design"}'
```

### 4. Use Python API

```python
from taxonomy_framework.models import CategoryNode, Taxonomy
from taxonomy_framework.pipeline import HybridClassifier
from taxonomy_framework.embeddings import SentenceTransformerBackend, EnsembleEmbedder
from taxonomy_framework.providers import ProviderFactory

# Build taxonomy
root = CategoryNode(name="Root", description="Product Categories")
electronics = CategoryNode(name="Electronics", parent=root)
root.children.append(electronics)
# ... add more categories

taxonomy = Taxonomy(root)

# Create classifier
embedder = EnsembleEmbedder([SentenceTransformerBackend()])
llm = ProviderFactory.create("openai", model="gpt-4o-mini", api_key="sk-...")

classifier = HybridClassifier(
    taxonomy=taxonomy,
    embedder=embedder,
    llm=llm
)

# Classify
result = classifier.classify("iPhone 15 Pro with titanium design")
print(f"Category: {result.predicted_category.path()}")
print(f"Confidence: {result.confidence_score}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Input Text                                    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Semantic Recall (Embeddings)                      │
│  • SentenceTransformer / SetFit embeddings                          │
│  • Cosine similarity to find top-K candidate regions                │
│  • High recall, efficient (no full taxonomy in context)             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 Constrained Traversal (LLM)                          │
│  • Navigate tree one level at a time                                │
│  • Only valid children presented at each step                       │
│  • Tool/function calling for structured output                      │
│  • IMPOSSIBLE to hallucinate invalid categories                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Sibling Contrast (Optional)                       │
│  • Compare confusing siblings explicitly                            │
│  • Request evidence for each candidate                              │
│  • Precision booster for similar categories                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Output                                       │
│  • ClassificationResult: category + confidence + path               │
│  • AbstainResult: when ambiguous (partial path + reason)            │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
taxonomy_classifier/
├── taxonomy_framework/          # Core framework
│   ├── models.py               # CategoryNode, Taxonomy, Result types
│   ├── pipeline.py             # HybridClassifier orchestration
│   ├── embeddings.py           # Embedding backends (SentenceTransformer, SetFit)
│   ├── traverser.py            # Constrained LLM tree traversal
│   ├── contrast.py             # Sibling contrast comparison
│   ├── evaluation.py           # Hierarchical metrics
│   ├── training.py             # SetFit trainer
│   │
│   ├── providers/              # LLM provider implementations
│   │   ├── base.py            # BaseLLMProvider ABC
│   │   ├── factory.py         # ProviderFactory
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── google_provider.py
│   │   ├── cohere_provider.py
│   │   ├── ollama_provider.py
│   │   ├── vllm_provider.py
│   │   └── huggingface_tgi.py
│   │
│   ├── api/                    # FastAPI service
│   │   ├── main.py            # Application entry point
│   │   ├── routes/            # API endpoints
│   │   ├── schemas.py         # Request/response models
│   │   └── dependencies.py    # Dependency injection
│   │
│   ├── auth/                   # Authentication
│   │   ├── oauth2.py          # JWT authentication
│   │   ├── api_key.py         # API key validation
│   │   └── sso.py             # OIDC/SSO integration
│   │
│   └── config/                 # Configuration
│       ├── settings.py        # Pydantic Settings
│       └── providers.py       # Provider configs
│
├── examples/                   # Usage examples
│   ├── 01_zero_shot_semantic_classification.py
│   ├── 02_setfit_finetuning_comparison.py
│   └── 03_memory_efficient_training.py
│
├── tests/                      # Test suite
│   ├── test_providers/        # Provider tests
│   └── test_api/              # API tests
│
├── config/                     # Configuration templates
│   ├── .env.example
│   └── providers.example.yaml
│
└── requirements.txt
```

## Examples

### Zero-Shot Classification (No Training)

```bash
python examples/01_zero_shot_semantic_classification.py
```

Uses pre-trained SentenceTransformer for immediate classification without any fine-tuning.

### Fine-tuned vs Baseline Comparison

```bash
python examples/02_setfit_finetuning_comparison.py
```

Trains a SetFit model on DBpedia and compares against pre-trained baseline.
Saves the trained model to `./models/dbpedia-setfit/` for reuse.

```bash
# Force retraining (ignores saved model)
python examples/02_setfit_finetuning_comparison.py --retrain
```

### Memory-Efficient Training

```bash
python examples/03_memory_efficient_training.py
```

Optimized for resource-constrained environments (8GB RAM, 4GB GPU).
Saves the trained model to `./models/efficient-setfit/` for reuse.

```bash
# Force retraining (ignores saved model)
python examples/03_memory_efficient_training.py --retrain
```

## Supported LLM Providers

| Provider | Tool Calling | JSON Mode | Local/Cloud |
|----------|--------------|-----------|-------------|
| OpenAI | ✅ | ✅ | Cloud |
| Anthropic | ✅ | ✅ | Cloud |
| Google Gemini | ✅ | ✅ | Cloud |
| Cohere | ✅ | ✅ | Cloud |
| Ollama | ✅* | ✅ | Local |
| vLLM | ✅* | ✅ | Local |
| HuggingFace TGI | ❌ | ✅ | Local |

*Model-dependent

## API Reference

### Classification Endpoints

- `POST /api/v1/classify` - Classify single text
- `POST /api/v1/classify/batch` - Classify multiple texts
- `GET /api/v1/taxonomy` - Get taxonomy structure

### Authentication

- `POST /auth/token` - Get OAuth2 access token
- `GET /auth/sso/login` - Initiate SSO login
- `GET /auth/sso/callback` - SSO callback

### Health

- `GET /health` - Health check
- `GET /api/v1/health` - API health check

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Provider name | `openai` |
| `LLM_MODEL` | Model identifier | `gpt-4o-mini` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `GOOGLE_API_KEY` | Google AI API key | - |
| `COHERE_API_KEY` | Cohere API key | - |
| `API_BASE_URL` | Custom API endpoint | - |
| `JWT_SECRET_KEY` | JWT signing key | (generated) |
| `API_KEYS` | Comma-separated valid API keys | - |

See `config/.env.example` for full list.

## Evaluation Metrics

The framework provides hierarchical-aware metrics:

- **Hierarchical Precision/Recall/F1**: Accounts for partial path matches
- **Per-Level Accuracy**: Accuracy at each taxonomy level (L1, L2, L3)
- **Exact Match Accuracy**: Full path match rate
- **Depth-Weighted Accuracy**: Weighted by prediction depth
- **LCA Distance**: Mean distance to Lowest Common Ancestor

## Development

### Running Tests

```bash
# All tests
pytest

# Provider tests only
pytest tests/test_providers/ -v

# API tests only
pytest tests/test_api/ -v

# With coverage
pytest --cov=taxonomy_framework --cov-report=html
```

### Code Style

```bash
# Format
black taxonomy_framework/ tests/ examples/

# Lint
ruff check taxonomy_framework/ tests/ examples/

# Type check
mypy taxonomy_framework/
```

## Theoretical Background

This framework implements the architecture described in the project's [design document](GOAL.md), which addresses:

1. **Why embeddings for routing**: High recall, efficient, robust to varied phrasing
2. **Why constrained traversal**: Guaranteed validity, no hallucinations
3. **Why abstention**: Honest uncertainty handling for ambiguous cases
4. **Why the combination works**: Each technique handles what it's good at

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (`pytest`)
5. Submit a pull request

## Acknowledgments

- [SetFit](https://github.com/huggingface/setfit) for few-shot fine-tuning
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
