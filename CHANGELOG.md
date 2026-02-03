# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
