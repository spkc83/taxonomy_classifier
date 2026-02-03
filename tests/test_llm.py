"""
Tests for LLM tool/function calling support.

TDD tests for:
- ToolCallResult dataclass
- build_traversal_tools() helper
- MockLLMClient tool calling support
"""
import pytest
from taxonomy_framework.providers import BaseLLMProvider, ToolCallResult, build_traversal_tools
from taxonomy_framework import CategoryNode
from tests.conftest import MockLLMClient


class TestToolCallResult:
    """Tests for ToolCallResult dataclass."""
    
    def test_create_tool_call_result(self):
        """Test basic ToolCallResult creation."""
        result = ToolCallResult(name="select_child", arguments={"child_name": "Hardware"})
        assert result.name == "select_child"
        assert result.arguments["child_name"] == "Hardware"
    
    def test_tool_call_result_with_confidence(self):
        """Test ToolCallResult with confidence argument."""
        result = ToolCallResult(
            name="select_child", 
            arguments={"child_name": "Software", "confidence": 0.85}
        )
        assert result.name == "select_child"
        assert result.arguments["child_name"] == "Software"
        assert result.arguments["confidence"] == 0.85
    
    def test_tool_call_result_abstain(self):
        """Test ToolCallResult for abstain tool."""
        result = ToolCallResult(
            name="abstain",
            arguments={"reason": "No category fits", "closest_options": ["Hardware"]}
        )
        assert result.name == "abstain"
        assert "No category fits" in result.arguments["reason"]


class TestBuildTraversalTools:
    """Tests for build_traversal_tools() helper function."""
    
    def test_generates_select_child_tool(self):
        """Test that select_child tool is generated with enum constraint."""
        children = [
            CategoryNode(name="Hardware", description="Physical devices"),
            CategoryNode(name="Software", description="Code and apps")
        ]
        tools = build_traversal_tools(children)
        
        assert len(tools) >= 1
        select_child = next(t for t in tools if t["function"]["name"] == "select_child")
        assert select_child is not None
        
        # Check enum constraint
        params = select_child["function"]["parameters"]
        assert "child_name" in params["properties"]
        assert params["properties"]["child_name"]["enum"] == ["Hardware", "Software"]
    
    def test_includes_abstain_tool(self):
        """Test that abstain tool is included."""
        children = [CategoryNode(name="Test")]
        tools = build_traversal_tools(children)
        
        abstain_tool = next((t for t in tools if t["function"]["name"] == "abstain"), None)
        assert abstain_tool is not None
    
    def test_select_child_has_confidence_param(self):
        """Test that select_child tool has confidence parameter."""
        children = [CategoryNode(name="Hardware")]
        tools = build_traversal_tools(children)
        
        select_child = next(t for t in tools if t["function"]["name"] == "select_child")
        params = select_child["function"]["parameters"]
        assert "confidence" in params["properties"]
        assert params["properties"]["confidence"]["type"] == "number"
    
    def test_abstain_has_reason_required(self):
        """Test that abstain tool requires reason parameter."""
        children = [CategoryNode(name="Test")]
        tools = build_traversal_tools(children)
        
        abstain_tool = next(t for t in tools if t["function"]["name"] == "abstain")
        params = abstain_tool["function"]["parameters"]
        assert "reason" in params["required"]
    
    def test_tool_format_is_openai_compatible(self):
        """Test that tool format follows OpenAI function calling schema."""
        children = [CategoryNode(name="Test")]
        tools = build_traversal_tools(children)
        
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]


class TestMockLLMClientToolCalling:
    """Tests for MockLLMClient tool calling support."""
    
    def test_call_with_tools_returns_configured_response(self):
        """Test that call_with_tools returns the configured mock response."""
        mock_tool_response = ToolCallResult(
            name="select_child",
            arguments={"child_name": "Hardware", "confidence": 0.9}
        )
        client = MockLLMClient(mock_response={"ignored": True})
        client.mock_tool_response = mock_tool_response
        
        result = client.call_with_tools(
            system_prompt="test",
            user_prompt="test",
            tools=[],
            tool_choice="select_child"
        )
        
        assert result.name == "select_child"
        assert result.arguments["child_name"] == "Hardware"
    
    def test_call_with_tools_default_response(self):
        """Test default response when no mock_tool_response is set."""
        client = MockLLMClient(mock_response={"ignored": True})
        
        tools = [
            {
                "type": "function",
                "function": {"name": "test_tool", "parameters": {}}
            }
        ]
        
        result = client.call_with_tools(
            system_prompt="test",
            user_prompt="test",
            tools=tools,
            tool_choice="auto"
        )
        
        # Should return first tool with empty args
        assert result.name == "test_tool"
        assert result.arguments == {}
    
    def test_call_with_tools_no_tools_provided(self):
        """Test behavior when no tools are provided and no mock set."""
        client = MockLLMClient()
        
        result = client.call_with_tools(
            system_prompt="test",
            user_prompt="test",
            tools=[],
            tool_choice="auto"
        )
        
        # Should return unknown when no tools
        assert result.name == "unknown"


class TestBaseLLMProviderInterface:
    """Tests for BaseLLMProvider.call_with_tools interface (not actual API calls)."""
    
    def test_call_with_tools_method_exists(self):
        """Test that BaseLLMProvider has call_with_tools method."""
        assert hasattr(BaseLLMProvider, 'call_with_tools')
    
    def test_build_traversal_tools_importable(self):
        """Test that build_traversal_tools is importable from providers module."""
        from taxonomy_framework.providers import build_traversal_tools
        assert callable(build_traversal_tools)
