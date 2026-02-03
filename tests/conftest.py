"""
Shared pytest fixtures for taxonomy_framework tests.

Provides mock taxonomy, LLM, and embedding fixtures following patterns
from framework_test.py.
"""
import pytest
import numpy as np
from typing import Any, Dict, List, Optional

from taxonomy_framework import (
    Taxonomy,
    CategoryNode,
    EnsembleEmbedder,
    ToolCallResult,
)
from taxonomy_framework.embeddings import EmbeddingModel
from taxonomy_framework.providers import BaseLLMProvider


# ============================================================================
# Test-only Mock Classes
# These are defined locally in the test suite, not in the main framework.
# ============================================================================

class MockLLMClient(BaseLLMProvider):
    """Mock client for testing without API keys."""
    def __init__(self, mock_response: Optional[Dict[str, Any]] = None):
        self.mock_response = mock_response or {"result": "mock"}
        self.mock_tool_response: Optional[ToolCallResult] = None

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def supports_tool_calling(self) -> bool:
        return True

    def generate_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        return self.mock_response
    
    def call_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto"
    ) -> ToolCallResult:
        if self.mock_tool_response:
            return self.mock_tool_response
        if tools:
            func_name = tools[0]["function"]["name"]
            if func_name == "select_child":
                params = tools[0]["function"]["parameters"]
                child_names = params.get("properties", {}).get("child_name", {}).get("enum", [])
                first_child = child_names[0] if child_names else "unknown"
                return ToolCallResult(
                    name="select_child",
                    arguments={"child_name": first_child, "confidence": 0.5}
                )
            return ToolCallResult(name=func_name, arguments={})
        return ToolCallResult(name="unknown", arguments={})


class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing with random embeddings."""
    def __init__(self, dim: int = 384):
        self.dim = dim
        
    def embed_text(self, text: str) -> np.ndarray:
        return np.random.rand(self.dim)
        
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.random.rand(len(texts), self.dim)


@pytest.fixture
def mock_taxonomy():
    """Create a mock taxonomy tree for testing.
    
    Structure:
        Root
        ├── Tech (Technology related issues)
        │   ├── Hardware (Physical devices)
        │   └── Software (Code and apps)
        └── Finance (Money matters)
            └── Billing (Invoices and payments)
    """
    root = CategoryNode(name="Root")
    tech = CategoryNode(name="Tech", description="Technology related issues", parent=root)
    hardware = CategoryNode(name="Hardware", description="Physical devices", parent=tech)
    software = CategoryNode(name="Software", description="Code and apps", parent=tech)
    tech.children = [hardware, software]
    
    finance = CategoryNode(name="Finance", description="Money matters", parent=root)
    billing = CategoryNode(name="Billing", description="Invoices and payments", parent=finance)
    finance.children = [billing]
    
    root.children = [tech, finance]
    return Taxonomy(root)


class SmartMockLLM(MockLLMClient):
    """Mock LLM that handles both classification and critic responses."""
    
    def generate_json(self, system_prompt: str, user_prompt: str, max_retries: int = 3):
        """Return appropriate mock response based on prompt type."""
        if "QA Auditor" in system_prompt:
            return {"valid": True, "reason": "Looks good."}
        else:
            return self.mock_response


@pytest.fixture
def mock_llm():
    """Create a SmartMockLLM configured for classification tests."""
    mock_response = {
        "category_path": "Tech > Hardware",
        "confidence": 0.95,
        "reasoning": "The input mentions a broken screen."
    }
    return SmartMockLLM(mock_response=mock_response)


@pytest.fixture
def mock_embedder():
    """Create a mock embedding model for testing."""
    return MockEmbeddingModel(dim=10)


@pytest.fixture
def mock_ensemble_embedder(mock_embedder):
    """Create an ensemble embedder with mock backend."""
    return EnsembleEmbedder([mock_embedder])
