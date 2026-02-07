# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2025-02-07

### Changed
- **Example 03**: Replaced DBpedia dataset (30-40% baseline) with TREC Question Classification
  (`SetFit/TREC-QC`) — 6 coarse / 50 fine classes with 80%+ baseline accuracy on coarse labels
- **Example 03**: Switched embedding model from `paraphrase-MiniLM-L6-v2` (22M params) to
  `BAAI/bge-small-en-v1.5` (33M params) for substantially better retrieval quality
- **Example 03**: Reduced training batch size from 16 to 8 for 4GB GPU compatibility
- **Example 03**: Reordered pipeline stages to load only one model at a time, preventing
  OOM on memory-constrained GPUs

### Added
- `EnsembleEmbedder.build_index()` / `clear_index()`: pre-compute and cache candidate
  embeddings once, eliminating redundant re-embedding on every query
- `SentenceTransformerBackend.query_prefix` parameter: supports models that require
  asymmetric query/passage prefixes (e.g. BGE family)
- `SentenceTransformerBackend.device` parameter: explicit device placement for GPU inference
- `EmbeddingModel.model_id` property: stable identifier for embedding cache keying
- Normalized embeddings (`normalize_embeddings=True`) in both `SentenceTransformerBackend`
  and `SetFitBackend` — cosine similarity reduces to a single dot product

### Fixed
- Suppressed `encoder_attention_mask` FutureWarning from transformers 4.54 (upstream bug
  where `BertEncoder` passes its own deprecated kwarg internally; fixed in 4.55)
- `SetFitBackend.from_finetuned()` now sets `model_id` correctly for cache keying

## [1.0.0] - 2025-02-02

### Added
- Initial project structure with taxonomy classification framework
- Core classification modules: `traverser.py`, `contrast.py`, `embeddings.py`
- Multi-provider LLM support (OpenAI, Anthropic, Google, Cohere, Ollama, vLLM, HuggingFace TGI)
- Provider factory pattern for dynamic LLM backend selection
- SetFit model training and evaluation capabilities
- OAuth2/SSO authentication module
- FastAPI-based classification API routes
- Configuration management with YAML and environment variables
- Example scripts for zero-shot semantic classification
- Example scripts for SetFit fine-tuning comparison
- Memory-efficient training examples
- Comprehensive test suite with pytest

### Infrastructure
- Project documentation (README.md, GOAL.md)
- Example configuration files (`.env.example`, `providers.example.yaml`)
- GitHub Actions workflows

## [0.1.0] - 2025-02-02

### Added
- Initial release of taxonomy classifier framework
- Core taxonomy traversal and classification pipeline
- Support for multiple LLM providers
- SetFit-based efficient text classification
- Document reader utilities
- Evaluation metrics and training utilities
