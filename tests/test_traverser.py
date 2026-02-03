"""
Tests for ConstrainedTraverser and TraversalResult.

TDD approach: Tests written FIRST before implementation.
"""
import pytest
from taxonomy_framework import CategoryNode
from taxonomy_framework.providers import ToolCallResult
from tests.conftest import MockLLMClient


class TestTraversalResultDataclass:
    """Tests for TraversalResult dataclass creation."""

    def test_traversal_result_basic_creation(self, mock_taxonomy):
        """TraversalResult can be created with required fields."""
        from taxonomy_framework.traverser import TraversalResult
        
        leaf_node = mock_taxonomy.root.children[0].children[0]  # Hardware
        result = TraversalResult(
            final_node=leaf_node,
            path=["Root", "Tech", "Hardware"],
            confidence=0.95,
            needs_contrast=False
        )
        
        assert result.final_node == leaf_node
        assert result.path == ["Root", "Tech", "Hardware"]
        assert result.confidence == 0.95
        assert result.needs_contrast is False

    def test_traversal_result_optional_fields_default(self, mock_taxonomy):
        """TraversalResult optional fields have correct defaults."""
        from taxonomy_framework.traverser import TraversalResult
        
        node = mock_taxonomy.root
        result = TraversalResult(
            final_node=node,
            path=["Root"],
            confidence=0.5,
            needs_contrast=True
        )
        
        assert result.contrast_candidates is None
        assert result.did_abstain is False
        assert result.abstain_reason is None

    def test_traversal_result_with_contrast_candidates(self, mock_taxonomy):
        """TraversalResult can store contrast candidates."""
        from taxonomy_framework.traverser import TraversalResult
        
        tech = mock_taxonomy.root.children[0]
        finance = mock_taxonomy.root.children[1]
        
        result = TraversalResult(
            final_node=tech,
            path=["Root", "Tech"],
            confidence=0.55,
            needs_contrast=True,
            contrast_candidates=[tech, finance]
        )
        
        assert result.needs_contrast is True
        assert result.contrast_candidates == [tech, finance]

    def test_traversal_result_abstain_fields(self, mock_taxonomy):
        """TraversalResult can indicate abstain with reason."""
        from taxonomy_framework.traverser import TraversalResult
        
        node = mock_taxonomy.root
        result = TraversalResult(
            final_node=node,
            path=["Root"],
            confidence=0.0,
            needs_contrast=False,
            did_abstain=True,
            abstain_reason="Input does not match any category"
        )
        
        assert result.did_abstain is True
        assert result.abstain_reason == "Input does not match any category"


class TestConstrainedTraverserInit:
    """Tests for ConstrainedTraverser initialization."""

    def test_traverser_init_with_defaults(self, mock_llm):
        """ConstrainedTraverser can be initialized with default args."""
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        traverser = ConstrainedTraverser(llm=mock_llm)
        
        assert traverser.llm == mock_llm
        assert traverser.ambiguity_threshold == 0.1
        assert traverser.logger is not None

    def test_traverser_init_custom_threshold(self, mock_llm):
        """ConstrainedTraverser accepts custom ambiguity threshold."""
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        traverser = ConstrainedTraverser(llm=mock_llm, ambiguity_threshold=0.2)
        
        assert traverser.ambiguity_threshold == 0.2

    def test_traverser_init_custom_logger(self, mock_llm):
        """ConstrainedTraverser accepts custom logger."""
        import logging
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        custom_logger = logging.getLogger("custom_test_logger")
        traverser = ConstrainedTraverser(llm=mock_llm, logger=custom_logger)
        
        assert traverser.logger == custom_logger


class TestTraverseMethod:
    """Tests for ConstrainedTraverser.traverse() method."""

    def test_traverse_returns_traversal_result(self, mock_taxonomy, mock_llm):
        """traverse() returns a TraversalResult instance."""
        from taxonomy_framework.traverser import ConstrainedTraverser, TraversalResult
        
        # Mock LLM to select "Tech" then "Hardware"
        mock_llm.mock_tool_response = ToolCallResult(
            name="select_child",
            arguments={"child_name": "Tech", "confidence": 0.9}
        )
        
        traverser = ConstrainedTraverser(llm=mock_llm)
        # Start from a leaf to get immediate result
        leaf = mock_taxonomy.root.children[0].children[0]  # Hardware
        result = traverser.traverse("broken screen", leaf)
        
        assert isinstance(result, TraversalResult)

    def test_traverse_stops_at_leaf(self, mock_taxonomy, mock_llm):
        """traverse() stops immediately when entry node is a leaf."""
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        traverser = ConstrainedTraverser(llm=mock_llm)
        leaf = mock_taxonomy.root.children[0].children[0]  # Hardware (leaf)
        
        result = traverser.traverse("test input", leaf)
        
        assert result.final_node == leaf
        assert result.path == ["Hardware"]
        assert result.did_abstain is False

    def test_traverse_single_child_auto_descends(self, mock_taxonomy, mock_llm):
        """traverse() auto-descends when node has single child (no LLM call)."""
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        traverser = ConstrainedTraverser(llm=mock_llm)
        # Finance has single child: Billing
        finance = mock_taxonomy.root.children[1]  # Finance
        
        result = traverser.traverse("invoice question", finance)
        
        # Should reach Billing automatically
        assert result.final_node.name == "Billing"
        assert "Finance" in result.path
        assert "Billing" in result.path

    def test_traverse_llm_selects_child(self, mock_taxonomy, mock_llm):
        """traverse() uses LLM to select among multiple children."""
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        # Mock LLM to select "Hardware"
        mock_llm.mock_tool_response = ToolCallResult(
            name="select_child",
            arguments={"child_name": "Hardware", "confidence": 0.85}
        )
        
        traverser = ConstrainedTraverser(llm=mock_llm)
        tech = mock_taxonomy.root.children[0]  # Tech (has Hardware & Software children)
        
        result = traverser.traverse("broken screen", tech)
        
        assert result.final_node.name == "Hardware"
        assert result.confidence == 0.85
        assert "Tech" in result.path
        assert "Hardware" in result.path

    def test_traverse_full_path_from_root(self, mock_taxonomy):
        """traverse() traverses full path from root to leaf."""
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        # Create mock that returns different responses for each level
        class SequentialMockLLM(MockLLMClient):
            def __init__(self):
                super().__init__()
                self.call_count = 0
                self.responses = [
                    ToolCallResult(name="select_child", arguments={"child_name": "Tech", "confidence": 0.9}),
                    ToolCallResult(name="select_child", arguments={"child_name": "Software", "confidence": 0.8}),
                ]
            
            def call_with_tools(self, system_prompt, user_prompt, tools, tool_choice="auto"):
                response = self.responses[self.call_count]
                self.call_count += 1
                return response
        
        mock_llm = SequentialMockLLM()
        traverser = ConstrainedTraverser(llm=mock_llm)
        
        result = traverser.traverse("app crash bug", mock_taxonomy.root)
        
        assert result.final_node.name == "Software"
        assert result.path == ["Root", "Tech", "Software"]
        assert mock_llm.call_count == 2  # Root→Tech, Tech→Software


class TestAbstainBehavior:
    """Tests for abstain tool handling."""

    def test_traverse_abstain_returns_flag(self, mock_taxonomy, mock_llm):
        """traverse() sets did_abstain=True when LLM calls abstain."""
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        mock_llm.mock_tool_response = ToolCallResult(
            name="abstain",
            arguments={"reason": "Text doesn't match any category"}
        )
        
        traverser = ConstrainedTraverser(llm=mock_llm)
        result = traverser.traverse("random gibberish xyz123", mock_taxonomy.root)
        
        assert result.did_abstain is True
        assert result.abstain_reason == "Text doesn't match any category"
        assert result.final_node == mock_taxonomy.root  # Stopped at root

    def test_traverse_abstain_preserves_path(self, mock_taxonomy):
        """traverse() preserves path when abstaining mid-traversal."""
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        class AbstainMidwayMockLLM(MockLLMClient):
            def __init__(self):
                super().__init__()
                self.call_count = 0
            
            def call_with_tools(self, system_prompt, user_prompt, tools, tool_choice="auto"):
                self.call_count += 1
                if self.call_count == 1:
                    return ToolCallResult(name="select_child", arguments={"child_name": "Tech", "confidence": 0.8})
                else:
                    return ToolCallResult(name="abstain", arguments={"reason": "Ambiguous between Hardware and Software"})
        
        mock_llm = AbstainMidwayMockLLM()
        traverser = ConstrainedTraverser(llm=mock_llm)
        
        result = traverser.traverse("some tech thing", mock_taxonomy.root)
        
        assert result.did_abstain is True
        assert "Root" in result.path
        assert "Tech" in result.path
        assert result.final_node.name == "Tech"  # Stopped at Tech


class TestNeedsContrast:
    """Tests for needs_contrast flag based on confidence threshold."""

    def test_high_confidence_no_contrast_needed(self, mock_taxonomy, mock_llm):
        """High confidence selection doesn't need contrast."""
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        mock_llm.mock_tool_response = ToolCallResult(
            name="select_child",
            arguments={"child_name": "Hardware", "confidence": 0.95}
        )
        
        traverser = ConstrainedTraverser(llm=mock_llm, ambiguity_threshold=0.1)
        tech = mock_taxonomy.root.children[0]
        
        result = traverser.traverse("broken monitor", tech)
        
        assert result.needs_contrast is False

    def test_low_confidence_needs_contrast(self, mock_taxonomy, mock_llm):
        """Low confidence selection needs contrast comparison."""
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        mock_llm.mock_tool_response = ToolCallResult(
            name="select_child",
            arguments={"child_name": "Hardware", "confidence": 0.55}
        )
        
        traverser = ConstrainedTraverser(llm=mock_llm, ambiguity_threshold=0.1)
        tech = mock_taxonomy.root.children[0]
        
        result = traverser.traverse("device issue", tech)
        
        # Confidence 0.55 < (1 - 0.1) = 0.9, so needs_contrast should be True
        assert result.needs_contrast is True


class TestPromptBuilding:
    """Tests for system/user prompt construction."""

    def test_system_prompt_exists(self, mock_llm):
        """ConstrainedTraverser has _build_system_prompt method."""
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        traverser = ConstrainedTraverser(llm=mock_llm)
        
        assert hasattr(traverser, "_build_system_prompt")
        prompt = traverser._build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_user_prompt_includes_text_and_children(self, mock_taxonomy, mock_llm):
        """User prompt includes input text and available children."""
        from taxonomy_framework.traverser import ConstrainedTraverser
        
        traverser = ConstrainedTraverser(llm=mock_llm)
        tech = mock_taxonomy.root.children[0]
        
        prompt = traverser._build_user_prompt("broken screen test", tech)
        
        assert "broken screen test" in prompt
        assert "Tech" in prompt
        assert "Hardware" in prompt
        assert "Software" in prompt
