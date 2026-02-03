"""
Tests for SiblingContrast and ContrastResult.

TDD approach: Tests written FIRST before implementation.
"""
import pytest
from taxonomy_framework import CategoryNode
from taxonomy_framework.providers import ToolCallResult
from tests.conftest import MockLLMClient


class TestContrastResultDataclass:
    """Tests for ContrastResult dataclass creation."""

    def test_contrast_result_basic_creation(self, mock_taxonomy):
        """ContrastResult can be created with required fields."""
        from taxonomy_framework.contrast import ContrastResult
        
        node = mock_taxonomy.root.children[0]  # Tech
        result = ContrastResult(
            choice=node,
            is_neither=False,
            reasoning="Tech is clearly the better fit"
        )
        
        assert result.choice == node
        assert result.is_neither is False
        assert result.reasoning == "Tech is clearly the better fit"

    def test_contrast_result_neither_choice(self):
        """ContrastResult with neither choice has None for choice."""
        from taxonomy_framework.contrast import ContrastResult
        
        result = ContrastResult(
            choice=None,
            is_neither=True,
            reasoning="Neither category fits the input"
        )
        
        assert result.choice is None
        assert result.is_neither is True
        assert result.reasoning == "Neither category fits the input"

    def test_contrast_result_reasoning_optional(self, mock_taxonomy):
        """ContrastResult reasoning field is optional."""
        from taxonomy_framework.contrast import ContrastResult
        
        node = mock_taxonomy.root.children[0]
        result = ContrastResult(choice=node, is_neither=False)
        
        assert result.reasoning is None


class TestSiblingContrastInit:
    """Tests for SiblingContrast initialization."""

    def test_sibling_contrast_init(self, mock_llm):
        """SiblingContrast can be initialized with LLM client."""
        from taxonomy_framework.contrast import SiblingContrast
        
        contrast = SiblingContrast(llm=mock_llm)
        
        assert contrast.llm == mock_llm

    def test_sibling_contrast_has_contrast_method(self, mock_llm):
        """SiblingContrast has contrast() method."""
        from taxonomy_framework.contrast import SiblingContrast
        
        contrast = SiblingContrast(llm=mock_llm)
        
        assert hasattr(contrast, "contrast")
        assert callable(contrast.contrast)


class TestContrastMethod:
    """Tests for SiblingContrast.contrast() method."""

    def test_contrast_returns_contrast_result(self, mock_taxonomy, mock_llm):
        """contrast() returns a ContrastResult instance."""
        from taxonomy_framework.contrast import SiblingContrast, ContrastResult
        
        mock_llm.mock_tool_response = ToolCallResult(
            name="choose_category",
            arguments={"choice": "A", "reasoning": "A fits better"}
        )
        
        contrast = SiblingContrast(llm=mock_llm)
        tech = mock_taxonomy.root.children[0]
        finance = mock_taxonomy.root.children[1]
        
        result = contrast.contrast("tech support query", [tech, finance])
        
        assert isinstance(result, ContrastResult)

    def test_contrast_requires_exactly_two_candidates(self, mock_taxonomy, mock_llm):
        """contrast() raises ValueError if not exactly 2 candidates."""
        from taxonomy_framework.contrast import SiblingContrast
        
        contrast = SiblingContrast(llm=mock_llm)
        tech = mock_taxonomy.root.children[0]
        
        with pytest.raises(ValueError, match="Exactly 2 candidates required"):
            contrast.contrast("test input", [tech])
        
        with pytest.raises(ValueError, match="Exactly 2 candidates required"):
            contrast.contrast("test input", [tech, tech, tech])

    def test_contrast_choice_a_returns_first_candidate(self, mock_taxonomy, mock_llm):
        """LLM choosing 'A' returns the first candidate."""
        from taxonomy_framework.contrast import SiblingContrast
        
        mock_llm.mock_tool_response = ToolCallResult(
            name="choose_category",
            arguments={"choice": "A", "reasoning": "Tech is clearly the match"}
        )
        
        contrast = SiblingContrast(llm=mock_llm)
        tech = mock_taxonomy.root.children[0]
        finance = mock_taxonomy.root.children[1]
        
        result = contrast.contrast("technology question", [tech, finance])
        
        assert result.choice == tech
        assert result.is_neither is False
        assert result.reasoning == "Tech is clearly the match"

    def test_contrast_choice_b_returns_second_candidate(self, mock_taxonomy, mock_llm):
        """LLM choosing 'B' returns the second candidate."""
        from taxonomy_framework.contrast import SiblingContrast
        
        mock_llm.mock_tool_response = ToolCallResult(
            name="choose_category",
            arguments={"choice": "B", "reasoning": "Finance is the better fit"}
        )
        
        contrast = SiblingContrast(llm=mock_llm)
        tech = mock_taxonomy.root.children[0]
        finance = mock_taxonomy.root.children[1]
        
        result = contrast.contrast("billing inquiry", [tech, finance])
        
        assert result.choice == finance
        assert result.is_neither is False
        assert result.reasoning == "Finance is the better fit"

    def test_contrast_choice_neither_returns_none(self, mock_taxonomy, mock_llm):
        """LLM choosing 'neither' returns None with is_neither=True."""
        from taxonomy_framework.contrast import SiblingContrast
        
        mock_llm.mock_tool_response = ToolCallResult(
            name="choose_category",
            arguments={"choice": "neither", "reasoning": "Both categories are poor fits"}
        )
        
        contrast = SiblingContrast(llm=mock_llm)
        tech = mock_taxonomy.root.children[0]
        finance = mock_taxonomy.root.children[1]
        
        result = contrast.contrast("random gibberish", [tech, finance])
        
        assert result.choice is None
        assert result.is_neither is True
        assert result.reasoning == "Both categories are poor fits"

    def test_contrast_reasoning_optional_in_llm_response(self, mock_taxonomy, mock_llm):
        """contrast() handles missing reasoning in LLM response."""
        from taxonomy_framework.contrast import SiblingContrast
        
        mock_llm.mock_tool_response = ToolCallResult(
            name="choose_category",
            arguments={"choice": "A"}  # No reasoning provided
        )
        
        contrast = SiblingContrast(llm=mock_llm)
        tech = mock_taxonomy.root.children[0]
        finance = mock_taxonomy.root.children[1]
        
        result = contrast.contrast("test query", [tech, finance])
        
        assert result.choice == tech
        assert result.reasoning is None


class TestToolSchema:
    """Tests for contrast tool schema format."""

    def test_build_contrast_tools_returns_list(self, mock_llm):
        """_build_contrast_tools returns a list of tool schemas."""
        from taxonomy_framework.contrast import SiblingContrast
        
        contrast = SiblingContrast(llm=mock_llm)
        tools = contrast._build_contrast_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 1

    def test_tool_schema_has_enum_constraint(self, mock_llm):
        """Tool schema constrains choice to ["A", "B", "neither"]."""
        from taxonomy_framework.contrast import SiblingContrast
        
        contrast = SiblingContrast(llm=mock_llm)
        tools = contrast._build_contrast_tools()
        
        tool = tools[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "choose_category"
        
        properties = tool["function"]["parameters"]["properties"]
        assert "choice" in properties
        assert properties["choice"]["enum"] == ["A", "B", "neither"]

    def test_tool_schema_has_reasoning_field(self, mock_llm):
        """Tool schema includes reasoning field."""
        from taxonomy_framework.contrast import SiblingContrast
        
        contrast = SiblingContrast(llm=mock_llm)
        tools = contrast._build_contrast_tools()
        
        properties = tools[0]["function"]["parameters"]["properties"]
        assert "reasoning" in properties
        assert properties["reasoning"]["type"] == "string"


class TestPromptBuilding:
    """Tests for system/user prompt construction."""

    def test_system_prompt_exists(self, mock_llm):
        """SiblingContrast has _build_system_prompt method."""
        from taxonomy_framework.contrast import SiblingContrast
        
        contrast = SiblingContrast(llm=mock_llm)
        
        assert hasattr(contrast, "_build_system_prompt")
        prompt = contrast._build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_user_prompt_includes_text_and_candidates(self, mock_taxonomy, mock_llm):
        """User prompt includes input text and both candidate descriptions."""
        from taxonomy_framework.contrast import SiblingContrast
        
        contrast = SiblingContrast(llm=mock_llm)
        tech = mock_taxonomy.root.children[0]
        finance = mock_taxonomy.root.children[1]
        
        prompt = contrast._build_user_prompt("test query", tech, finance)
        
        assert "test query" in prompt
        assert "Option A:" in prompt
        assert "Option B:" in prompt
        assert "Tech" in prompt
        assert "Finance" in prompt

    def test_user_prompt_includes_descriptions_and_paths(self, mock_taxonomy, mock_llm):
        """User prompt includes category descriptions and paths."""
        from taxonomy_framework.contrast import SiblingContrast
        
        contrast = SiblingContrast(llm=mock_llm)
        tech = mock_taxonomy.root.children[0]
        finance = mock_taxonomy.root.children[1]
        
        prompt = contrast._build_user_prompt("test query", tech, finance)
        
        # Check descriptions are included
        assert "Technology related issues" in prompt
        assert "Money matters" in prompt
        # Check paths are included
        assert tech.path() in prompt
        assert finance.path() in prompt
