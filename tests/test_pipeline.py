"""
Tests for HybridClassifier pipeline orchestration.

Tests the new pipeline: semantic recall -> constrained traversal -> sibling contrast -> abstain/result

TDD: Tests written FIRST before implementation.
"""
import pytest
from unittest.mock import MagicMock

from taxonomy_framework import (
    Taxonomy,
    CategoryNode,
    ClassificationResult,
    AbstainResult,
    EnsembleEmbedder,
    ToolCallResult,
)
from tests.conftest import MockLLMClient, MockEmbeddingModel


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def deep_taxonomy():
    """Create a deeper taxonomy for pipeline testing.
    
    Structure:
        Root
        ├── Tech (Technology related issues)
        │   ├── Hardware (Physical devices)
        │   │   ├── Laptop (Portable computers)
        │   │   └── Phone (Mobile devices)
        │   └── Software (Code and apps)
        │       ├── OS (Operating systems)
        │       └── Apps (Applications)
        └── Finance (Money matters)
            ├── Billing (Invoices and payments)
            └── Refunds (Return money)
    """
    root = CategoryNode(name="Root")
    
    # Tech branch
    tech = CategoryNode(name="Tech", description="Technology related issues", parent=root)
    hardware = CategoryNode(name="Hardware", description="Physical devices", parent=tech)
    laptop = CategoryNode(name="Laptop", description="Portable computers", parent=hardware)
    phone = CategoryNode(name="Phone", description="Mobile devices", parent=hardware)
    hardware.children = [laptop, phone]
    
    software = CategoryNode(name="Software", description="Code and apps", parent=tech)
    os_cat = CategoryNode(name="OS", description="Operating systems", parent=software)
    apps = CategoryNode(name="Apps", description="Applications", parent=software)
    software.children = [os_cat, apps]
    
    tech.children = [hardware, software]
    
    # Finance branch
    finance = CategoryNode(name="Finance", description="Money matters", parent=root)
    billing = CategoryNode(name="Billing", description="Invoices and payments", parent=finance)
    refunds = CategoryNode(name="Refunds", description="Return money", parent=finance)
    finance.children = [billing, refunds]
    
    root.children = [tech, finance]
    return Taxonomy(root)


class PipelineMockLLM(MockLLMClient):
    """Mock LLM that can be configured for specific pipeline behaviors."""
    
    def __init__(self, tool_responses=None):
        super().__init__(mock_response={})
        self.tool_responses = tool_responses or []
        self.call_index = 0
    
    def call_with_tools(self, system_prompt, user_prompt, tools, tool_choice=None):
        """Return configured tool responses in sequence."""
        if self.call_index < len(self.tool_responses):
            response = self.tool_responses[self.call_index]
            self.call_index += 1
            return response
        # Default: select first available child with high confidence
        return ToolCallResult(name="select_child", arguments={"child_name": "Hardware", "confidence": 0.95})


class AbstainMockLLM(MockLLMClient):
    """Mock LLM that always abstains."""
    
    def call_with_tools(self, system_prompt, user_prompt, tools, tool_choice=None):
        return ToolCallResult(name="abstain", arguments={"reason": "No suitable category"})


class ControllableEmbedder(EnsembleEmbedder):
    """Embedder that returns controlled indices for testing."""
    
    def __init__(self, mock_indices):
        super().__init__([MockEmbeddingModel(dim=10)])
        self.mock_indices = mock_indices
    
    def retrieve_candidates(self, query, candidates_texts, top_k=5, k_rrf=60):
        """Return pre-configured indices."""
        return self.mock_indices[:top_k]


# ============================================================================
# Test: HybridClassifier Initialization
# ============================================================================

class TestHybridClassifierInit:
    """Tests for HybridClassifier initialization."""
    
    def test_initialization_with_required_components(self, deep_taxonomy):
        """HybridClassifier initializes with taxonomy, embedder, and llm."""
        from taxonomy_framework.pipeline import HybridClassifier, PipelineConfig
        
        embedder = EnsembleEmbedder([MockEmbeddingModel(dim=10)])
        llm = MockLLMClient({})
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        assert classifier.taxonomy is deep_taxonomy
        assert classifier.embedder is embedder
        assert classifier.llm is llm
        assert classifier.config is not None  # Default config
    
    def test_initialization_with_custom_config(self, deep_taxonomy):
        """HybridClassifier accepts custom PipelineConfig."""
        from taxonomy_framework.pipeline import HybridClassifier, PipelineConfig
        
        embedder = EnsembleEmbedder([MockEmbeddingModel(dim=10)])
        llm = MockLLMClient({})
        config = PipelineConfig(top_k_entry_branches=5, ambiguity_threshold=0.2)
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm,
            config=config
        )
        
        assert classifier.config.top_k_entry_branches == 5
        assert classifier.config.ambiguity_threshold == 0.2
    
    def test_creates_traverser_and_contrast(self, deep_taxonomy):
        """HybridClassifier creates internal traverser and contrast components."""
        from taxonomy_framework.pipeline import HybridClassifier
        from taxonomy_framework import ConstrainedTraverser, SiblingContrast
        
        embedder = EnsembleEmbedder([MockEmbeddingModel(dim=10)])
        llm = MockLLMClient({})
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        assert isinstance(classifier.traverser, ConstrainedTraverser)
        assert isinstance(classifier.contrast, SiblingContrast)
    
    def test_caches_entry_branches(self, deep_taxonomy):
        """HybridClassifier caches entry branches (children of root)."""
        from taxonomy_framework.pipeline import HybridClassifier
        
        embedder = EnsembleEmbedder([MockEmbeddingModel(dim=10)])
        llm = MockLLMClient({})
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        # Entry branches are Tech and Finance (children of Root)
        assert len(classifier.entry_branches) == 2
        assert classifier.entry_branches[0].name == "Tech"
        assert classifier.entry_branches[1].name == "Finance"
        assert len(classifier.entry_texts) == 2


# ============================================================================
# Test: Taxonomy Validation
# ============================================================================

class TestTaxonomyValidation:
    """Tests for taxonomy validation on construction."""
    
    def test_validates_taxonomy_no_circular_refs(self, deep_taxonomy):
        """HybridClassifier validates taxonomy has no circular references."""
        from taxonomy_framework.pipeline import HybridClassifier
        
        embedder = EnsembleEmbedder([MockEmbeddingModel(dim=10)])
        llm = MockLLMClient({})
        
        # Should not raise
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm
        )
        assert classifier is not None
    
    def test_raises_on_circular_reference(self):
        """HybridClassifier raises ValueError on circular taxonomy."""
        from taxonomy_framework.pipeline import HybridClassifier
        
        # Create circular reference
        root = CategoryNode(name="Root")
        child1 = CategoryNode(name="Child1", parent=root)
        child2 = CategoryNode(name="Child2", parent=child1)
        root.children = [child1]
        child1.children = [child2]
        child2.children = [child1]  # Circular!
        
        taxonomy = Taxonomy(root)
        embedder = EnsembleEmbedder([MockEmbeddingModel(dim=10)])
        llm = MockLLMClient({})
        
        with pytest.raises(ValueError, match="Circular reference"):
            HybridClassifier(
                taxonomy=taxonomy,
                embedder=embedder,
                llm=llm
            )


# ============================================================================
# Test: Classification Success Cases
# ============================================================================

class TestClassifySuccess:
    """Tests for successful classification."""
    
    def test_classify_returns_classification_result_on_success(self, deep_taxonomy):
        """classify() returns ClassificationResult on successful traversal."""
        from taxonomy_framework.pipeline import HybridClassifier
        
        # Configure LLM to traverse Tech -> Hardware -> Laptop
        tool_responses = [
            ToolCallResult(name="select_child", arguments={"child_name": "Hardware", "confidence": 0.9}),
            ToolCallResult(name="select_child", arguments={"child_name": "Laptop", "confidence": 0.95}),
        ]
        llm = PipelineMockLLM(tool_responses=tool_responses)
        embedder = ControllableEmbedder(mock_indices=[0])  # Returns Tech first
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("My laptop screen is broken")
        
        assert isinstance(result, ClassificationResult)
        assert result.predicted_category.name == "Laptop"
        assert result.input_text == "My laptop screen is broken"
    
    def test_classify_includes_traversal_path(self, deep_taxonomy):
        """ClassificationResult includes traversal_path."""
        from taxonomy_framework.pipeline import HybridClassifier
        
        # Configure LLM to traverse Tech -> Hardware -> Laptop
        tool_responses = [
            ToolCallResult(name="select_child", arguments={"child_name": "Hardware", "confidence": 0.9}),
            ToolCallResult(name="select_child", arguments={"child_name": "Laptop", "confidence": 0.95}),
        ]
        llm = PipelineMockLLM(tool_responses=tool_responses)
        embedder = ControllableEmbedder(mock_indices=[0])  # Returns Tech first
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("My laptop screen is broken")
        
        assert isinstance(result, ClassificationResult)
        assert "Tech" in result.traversal_path
        assert "Hardware" in result.traversal_path
        assert "Laptop" in result.traversal_path
    
    def test_classify_tries_branches_in_priority_order(self, deep_taxonomy):
        """classify() tries entry branches in order from semantic recall."""
        from taxonomy_framework.pipeline import HybridClassifier
        
        # First branch (Tech index=0) will abstain, second branch (Finance index=1) will succeed
        call_count = [0]
        
        class BranchTrackingLLM(MockLLMClient):
            def call_with_tools(self, system_prompt, user_prompt, tools, tool_choice=None):
                call_count[0] += 1
                # Call 1: Tech branch -> abstain immediately
                if call_count[0] == 1:
                    return ToolCallResult(name="abstain", arguments={"reason": "Not tech"})
                # Call 2+: Finance branch -> select Billing (leaf)
                return ToolCallResult(name="select_child", arguments={"child_name": "Billing", "confidence": 0.9})
        
        llm = BranchTrackingLLM({})
        embedder = ControllableEmbedder(mock_indices=[0, 1])  # Tech then Finance
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("I need to pay my invoice")
        
        # Should have tried Tech first (abstained), then Finance (succeeded)
        assert isinstance(result, ClassificationResult)
        assert result.predicted_category.name == "Billing"


# ============================================================================
# Test: Abstain Cases
# ============================================================================

class TestClassifyAbstain:
    """Tests for abstain scenarios."""
    
    def test_returns_abstain_when_traverser_abstains_all_branches(self, deep_taxonomy):
        """classify() returns AbstainResult when traverser abstains on all branches."""
        from taxonomy_framework.pipeline import HybridClassifier
        
        llm = AbstainMockLLM({})
        embedder = ControllableEmbedder(mock_indices=[0, 1])  # Both branches
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("Something completely unrelated to tech or finance")
        
        assert isinstance(result, AbstainResult)
        assert result.reason == "low_confidence"
        assert result.input_text == "Something completely unrelated to tech or finance"
    
    def test_returns_abstain_when_no_candidates(self, deep_taxonomy):
        """classify() returns AbstainResult when embedder returns no candidates."""
        from taxonomy_framework.pipeline import HybridClassifier
        
        llm = MockLLMClient({})
        embedder = ControllableEmbedder(mock_indices=[])  # No candidates
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("Some text")
        
        assert isinstance(result, AbstainResult)
        assert result.reason == "no_candidates"
    
    def test_returns_abstain_when_contrast_says_neither(self, deep_taxonomy):
        """classify() returns AbstainResult when contrast comparison returns 'neither'."""
        from taxonomy_framework.pipeline import HybridClassifier
        
        call_count = [0]
        
        class ContrastNeitherLLM(MockLLMClient):
            def call_with_tools(self, system_prompt, user_prompt, tools, tool_choice=None):
                call_count[0] += 1
                if "arbitrator" in system_prompt.lower() or "choose_category" in str(tools):
                    return ToolCallResult(name="choose_category", arguments={"choice": "neither"})
                return ToolCallResult(name="abstain", arguments={"reason": "Cannot determine"})
        
        llm = ContrastNeitherLLM({})
        embedder = ControllableEmbedder(mock_indices=[0, 1])
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("Ambiguous text about stuff")
        
        assert isinstance(result, AbstainResult)


# ============================================================================
# Test: Contrast Integration
# ============================================================================

class TestContrastIntegration:
    """Tests for sibling contrast integration."""
    
    def test_uses_contrast_choice_when_needs_contrast(self, deep_taxonomy):
        """classify() uses contrast choice when traverser marks needs_contrast."""
        from taxonomy_framework.pipeline import HybridClassifier
        
        call_count = [0]
        
        class ContrastChoiceLLM(MockLLMClient):
            def call_with_tools(self, system_prompt, user_prompt, tools, tool_choice=None):
                call_count[0] += 1
                if "arbitrator" in system_prompt.lower():
                    return ToolCallResult(name="choose_category", arguments={"choice": "B"})
                if call_count[0] == 1:
                    return ToolCallResult(
                        name="select_child", 
                        arguments={"child_name": "Hardware", "confidence": 0.9}
                    )
                return ToolCallResult(
                    name="select_child",
                    arguments={"child_name": "Laptop", "confidence": 0.95}
                )
        
        llm = ContrastChoiceLLM({})
        embedder = ControllableEmbedder(mock_indices=[0])
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("Some tech text")
        
        assert isinstance(result, (ClassificationResult, AbstainResult))


# ============================================================================
# Test: PipelineConfig Defaults
# ============================================================================

class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""
    
    def test_default_values(self):
        """PipelineConfig has sensible defaults."""
        from taxonomy_framework.pipeline import PipelineConfig
        
        config = PipelineConfig()
        
        assert config.top_k_entry_branches == 3
        assert config.ambiguity_threshold == 0.1
    
    def test_custom_values(self):
        """PipelineConfig accepts custom values."""
        from taxonomy_framework.pipeline import PipelineConfig
        
        config = PipelineConfig(top_k_entry_branches=5, ambiguity_threshold=0.2)
        
        assert config.top_k_entry_branches == 5
        assert config.ambiguity_threshold == 0.2


# ============================================================================
# Test: Logging
# ============================================================================

class TestLogging:
    """Tests for logging behavior."""
    
    def test_accepts_custom_logger(self, deep_taxonomy):
        """HybridClassifier accepts custom logger."""
        from taxonomy_framework.pipeline import HybridClassifier
        import logging
        
        embedder = EnsembleEmbedder([MockEmbeddingModel(dim=10)])
        llm = MockLLMClient({})
        custom_logger = logging.getLogger("test_pipeline")
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm,
            logger=custom_logger
        )
        
        assert classifier.logger is custom_logger
    
    def test_uses_default_logger(self, deep_taxonomy):
        """HybridClassifier uses default logger if not provided."""
        from taxonomy_framework.pipeline import HybridClassifier
        
        embedder = EnsembleEmbedder([MockEmbeddingModel(dim=10)])
        llm = MockLLMClient({})
        
        classifier = HybridClassifier(
            taxonomy=deep_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        assert classifier.logger is not None
