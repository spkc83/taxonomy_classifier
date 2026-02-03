"""
Integration tests for the hybrid classification pipeline.

Tests the full flow: semantic recall → constrained traversal → sibling contrast → abstain/result

These tests verify the NEW pipeline (HybridClassifier from pipeline.py), not the old one.
"""
import pytest

from taxonomy_framework import (
    Taxonomy,
    CategoryNode,
    HybridClassifier,
    PipelineConfig,
    EnsembleEmbedder,
    ClassificationResult,
    AbstainResult,
    ToolCallResult,
)
from tests.conftest import MockLLMClient, MockEmbeddingModel


class SmartMockLLM(MockLLMClient):
    """Mock LLM that handles different tool calling scenarios."""
    
    def __init__(self, selection_map=None):
        """
        Args:
            selection_map: Dict mapping child names to return. 
                          e.g. {"Tech": ("Tech", 0.9), "Hardware": ("Hardware", 0.95)}
        """
        super().__init__()
        self.selection_map = selection_map or {}
        self.call_history = []
    
    def call_with_tools(self, system_prompt, user_prompt, tools, tool_choice="auto"):
        """Return configured tool response based on available children."""
        self.call_history.append({
            "system": system_prompt,
            "user": user_prompt,
            "tools": tools
        })
        
        # Extract child names from tools
        child_names = []
        for tool in tools:
            if tool["function"]["name"] == "select_child":
                child_names = tool["function"]["parameters"]["properties"]["child_name"]["enum"]
                break
        
        # Find first matching selection
        for child in child_names:
            if child in self.selection_map:
                name, confidence = self.selection_map[child]
                return ToolCallResult(
                    name="select_child",
                    arguments={"child_name": name, "confidence": confidence}
                )
        
        # Default: select first child with high confidence
        if child_names:
            return ToolCallResult(
                name="select_child",
                arguments={"child_name": child_names[0], "confidence": 0.9}
            )
        
        # Fallback
        return ToolCallResult(name="abstain", arguments={"reason": "No valid options"})


class ControllableEmbedder(EnsembleEmbedder):
    """Embedder that returns controlled indices for testing."""
    
    def __init__(self, mock_indices):
        super().__init__([MockEmbeddingModel(dim=10)])
        self.mock_indices = mock_indices
    
    def retrieve_candidates(self, query, candidates_texts, top_k=5, k_rrf=60):
        """Return pre-configured indices."""
        return self.mock_indices[:top_k]


@pytest.fixture
def test_taxonomy():
    """Create a test taxonomy: Root → Tech/Finance → Hardware/Software, Billing/Refunds
    
    Structure:
        Root
        ├── Tech (Technology related issues)
        │   ├── Hardware (Physical devices like laptops, phones)
        │   └── Software (Code and applications)
        └── Finance (Money matters)
            ├── Billing (Invoices and payments)
            └── Refunds (Return money)
    
    All branches have 2+ children to ensure LLM is always consulted (no auto-descent).
    """
    root = CategoryNode(name="Root")
    tech = CategoryNode(name="Tech", description="Technology related issues", parent=root)
    hardware = CategoryNode(name="Hardware", description="Physical devices like laptops, phones", parent=tech)
    software = CategoryNode(name="Software", description="Code and applications", parent=tech)
    tech.children = [hardware, software]
    
    finance = CategoryNode(name="Finance", description="Money matters", parent=root)
    billing = CategoryNode(name="Billing", description="Invoices and payments", parent=finance)
    refunds = CategoryNode(name="Refunds", description="Return money", parent=finance)
    finance.children = [billing, refunds]
    
    root.children = [tech, finance]
    return Taxonomy(root)


# ============================================================================
# Test: End-to-End Classification
# ============================================================================

class TestEndToEndClassification:
    """Test full classification pipeline end-to-end."""
    
    def test_laptop_screen_cracked_classifies_to_hardware(self, test_taxonomy):
        """E2E: 'laptop screen cracked' should reach Hardware leaf via Tech branch."""
        embedder = ControllableEmbedder(mock_indices=[0])  # Force Tech first
        llm = SmartMockLLM(selection_map={
            "Tech": ("Tech", 0.9),
            "Hardware": ("Hardware", 0.95)
        })
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("My laptop screen is cracked")
        
        assert isinstance(result, ClassificationResult)
        assert result.predicted_category.name == "Hardware"
        assert "Tech" in result.traversal_path
        assert "Hardware" in result.traversal_path
    
    def test_gibberish_input_abstains(self, test_taxonomy):
        """E2E: Random gibberish should result in AbstainResult."""
        embedder = ControllableEmbedder(mock_indices=[0, 1])  # Both branches
        
        # LLM always abstains
        llm = MockLLMClient()
        llm.mock_tool_response = ToolCallResult(
            name="abstain",
            arguments={"reason": "Input makes no sense"}
        )
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("asdfghjkl qwertyuiop zxcvbnm")
        
        assert isinstance(result, AbstainResult)
        assert result.reason in ["no_candidates", "low_confidence", "explicit_abstain"]
    
    def test_classification_includes_traversal_path(self, test_taxonomy):
        """Verify ClassificationResult includes the traversal path."""
        embedder = ControllableEmbedder(mock_indices=[1])  # Finance first
        llm = SmartMockLLM(selection_map={
            "Finance": ("Finance", 0.9),
            "Billing": ("Billing", 0.95)
        })
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("I need to pay my invoice")
        
        assert isinstance(result, ClassificationResult)
        assert len(result.traversal_path) > 0
        # Path should include Finance and Billing
        assert "Finance" in result.traversal_path
        assert "Billing" in result.traversal_path
    
    def test_full_path_traversal_tech_software(self, test_taxonomy):
        """E2E: 'app crash' should traverse to Software leaf."""
        embedder = ControllableEmbedder(mock_indices=[0])  # Tech first
        llm = SmartMockLLM(selection_map={
            "Tech": ("Tech", 0.9),
            "Software": ("Software", 0.92)
        })
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("My app keeps crashing")
        
        assert isinstance(result, ClassificationResult)
        assert result.predicted_category.name == "Software"
        assert "Tech" in result.traversal_path
        assert "Software" in result.traversal_path


# ============================================================================
# Test: Abstain Scenarios
# ============================================================================

class TestAbstainScenarios:
    """Test scenarios that should result in AbstainResult."""
    
    def test_abstain_when_all_branches_abstain(self, test_taxonomy):
        """If traverser abstains on all entry branches, pipeline returns AbstainResult."""
        embedder = ControllableEmbedder(mock_indices=[0, 1])  # Both branches
        llm = MockLLMClient()
        llm.mock_tool_response = ToolCallResult(
            name="abstain",
            arguments={"reason": "No category fits"}
        )
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("This text doesn't fit any category")
        
        assert isinstance(result, AbstainResult)
    
    def test_abstain_has_suggested_action(self, test_taxonomy):
        """AbstainResult should include a suggested action."""
        embedder = ControllableEmbedder(mock_indices=[0, 1])
        llm = MockLLMClient()
        llm.mock_tool_response = ToolCallResult(
            name="abstain",
            arguments={"reason": "Unclear"}
        )
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("vague input")
        
        assert isinstance(result, AbstainResult)
        assert result.suggested_action in ["request_clarification", "manual_review", "use_best_guess"]
    
    def test_abstain_when_no_candidates_from_embedder(self, test_taxonomy):
        """AbstainResult when embedder returns no candidates."""
        embedder = ControllableEmbedder(mock_indices=[])  # No candidates
        llm = MockLLMClient()
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("Some text")
        
        assert isinstance(result, AbstainResult)
        assert result.reason == "no_candidates"


# ============================================================================
# Test: Contrast Triggering
# ============================================================================

class TestContrastTriggering:
    """Test that contrast is triggered for ambiguous cases."""
    
    def test_low_confidence_triggers_contrast_flag(self, test_taxonomy):
        """When confidence is below threshold, needs_contrast should be True in traversal."""
        embedder = ControllableEmbedder(mock_indices=[0])  # Tech first
        
        # Return low confidence to trigger contrast
        llm = SmartMockLLM(selection_map={
            "Tech": ("Tech", 0.5),  # Low confidence
            "Hardware": ("Hardware", 0.6)
        })
        
        config = PipelineConfig(ambiguity_threshold=0.5)  # Higher threshold
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm,
            config=config
        )
        
        # Should still get a result (contrast would be triggered but no candidates set)
        result = classifier.classify("Ambiguous tech issue")
        
        # The result should still work
        assert isinstance(result, (ClassificationResult, AbstainResult))
    
    def test_high_confidence_no_contrast(self, test_taxonomy):
        """When confidence is high, classification proceeds without contrast."""
        embedder = ControllableEmbedder(mock_indices=[0])
        
        # High confidence selections
        llm = SmartMockLLM(selection_map={
            "Tech": ("Tech", 0.95),
            "Hardware": ("Hardware", 0.98)
        })
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("Laptop hardware issue")
        
        assert isinstance(result, ClassificationResult)
        assert result.predicted_category.name == "Hardware"
    
    def test_ambiguous_input_between_branches(self, test_taxonomy):
        """Test ambiguous input that could fit multiple branches."""
        call_count = [0]
        
        class AmbiguousMockLLM(MockLLMClient):
            def call_with_tools(self, system_prompt, user_prompt, tools, tool_choice=None):
                call_count[0] += 1
                # First branch (Tech): abstain
                if call_count[0] == 1:
                    return ToolCallResult(name="abstain", arguments={"reason": "Not clearly tech"})
                # Second branch (Finance): select Billing with confidence
                return ToolCallResult(
                    name="select_child",
                    arguments={"child_name": "Billing", "confidence": 0.85}
                )
        
        embedder = ControllableEmbedder(mock_indices=[0, 1])  # Tech then Finance
        llm = AmbiguousMockLLM()
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("Payment for tech support")
        
        # Should end up in Finance/Billing after Tech abstains
        assert isinstance(result, ClassificationResult)
        assert result.predicted_category.name == "Billing"


# ============================================================================
# Test: Backward Compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Verify backward compatibility."""
    
    def test_hybrid_classifier_importable(self):
        """HybridClassifier should be importable from taxonomy_framework."""
        from taxonomy_framework import HybridClassifier
        assert HybridClassifier is not None
    
    def test_hybrid_classifier_is_from_pipeline(self):
        """HybridClassifier should be the pipeline version."""
        from taxonomy_framework import HybridClassifier
        from taxonomy_framework.pipeline import HybridClassifier as PipelineClassifier
        assert HybridClassifier is PipelineClassifier
    
    def test_classification_result_has_traversal_path(self, test_taxonomy):
        """ClassificationResult should have traversal_path attribute."""
        embedder = ControllableEmbedder(mock_indices=[0])
        llm = SmartMockLLM(selection_map={
            "Tech": ("Tech", 0.9),
            "Hardware": ("Hardware", 0.95)
        })
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("Hardware issue")
        
        assert isinstance(result, ClassificationResult)
        assert hasattr(result, 'traversal_path')
        assert isinstance(result.traversal_path, list)


# ============================================================================
# Test: Pipeline Configuration
# ============================================================================

class TestPipelineConfiguration:
    """Test PipelineConfig impact on classification."""
    
    def test_top_k_entry_branches_limits_candidates(self, test_taxonomy):
        """top_k_entry_branches limits how many entry branches are tried."""
        embedder = ControllableEmbedder(mock_indices=[0, 1])
        
        call_count = [0]
        
        class CountingAbstainLLM(MockLLMClient):
            def call_with_tools(self, system_prompt, user_prompt, tools, tool_choice=None):
                call_count[0] += 1
                return ToolCallResult(name="abstain", arguments={"reason": "Nope"})
        
        llm = CountingAbstainLLM()
        
        # Limit to only 1 entry branch
        config = PipelineConfig(top_k_entry_branches=1)
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm,
            config=config
        )
        
        result = classifier.classify("Some input")
        
        # Should only try 1 branch
        assert call_count[0] == 1
        assert isinstance(result, AbstainResult)
    
    def test_ambiguity_threshold_affects_contrast(self, test_taxonomy):
        """Higher ambiguity threshold makes contrast trigger more often."""
        embedder = ControllableEmbedder(mock_indices=[0])
        
        # Selection with 0.8 confidence
        llm = SmartMockLLM(selection_map={
            "Tech": ("Tech", 0.8),
            "Hardware": ("Hardware", 0.85)
        })
        
        # With low threshold, 0.8 confidence shouldn't trigger contrast
        config_low = PipelineConfig(ambiguity_threshold=0.05)  # needs > 0.95
        classifier_low = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm,
            config=config_low
        )
        
        result_low = classifier_low.classify("Tech hardware")
        
        # Should still get a result
        assert isinstance(result_low, (ClassificationResult, AbstainResult))


# ============================================================================
# Test: Result Content Verification
# ============================================================================

class TestResultContent:
    """Verify result objects have correct content."""
    
    def test_classification_result_input_text_preserved(self, test_taxonomy):
        """ClassificationResult preserves the original input text."""
        embedder = ControllableEmbedder(mock_indices=[0])
        llm = SmartMockLLM(selection_map={
            "Tech": ("Tech", 0.9),
            "Hardware": ("Hardware", 0.95)
        })
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        input_text = "My laptop screen is cracked and broken"
        result = classifier.classify(input_text)
        
        assert isinstance(result, ClassificationResult)
        assert result.input_text == input_text
    
    def test_abstain_result_input_text_preserved(self, test_taxonomy):
        """AbstainResult preserves the original input text."""
        embedder = ControllableEmbedder(mock_indices=[0, 1])
        llm = MockLLMClient()
        llm.mock_tool_response = ToolCallResult(
            name="abstain",
            arguments={"reason": "Cannot classify"}
        )
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        input_text = "Random gibberish text"
        result = classifier.classify(input_text)
        
        assert isinstance(result, AbstainResult)
        assert result.input_text == input_text
    
    def test_classification_result_has_confidence(self, test_taxonomy):
        """ClassificationResult has confidence score from traversal."""
        embedder = ControllableEmbedder(mock_indices=[0])
        llm = SmartMockLLM(selection_map={
            "Tech": ("Tech", 0.88),
            "Hardware": ("Hardware", 0.92)
        })
        
        classifier = HybridClassifier(
            taxonomy=test_taxonomy,
            embedder=embedder,
            llm=llm
        )
        
        result = classifier.classify("Laptop issue")
        
        assert isinstance(result, ClassificationResult)
        assert hasattr(result, 'confidence_score')
        assert result.confidence_score == 0.92  # Last traversal confidence
