"""Base abstractions for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from taxonomy_framework.models import CategoryNode


@dataclass
class ToolCallResult:
    """Result of a tool/function call from the LLM."""

    name: str
    arguments: Dict[str, Any]


@dataclass
class ProviderCapabilities:
    """Describes the capabilities of an LLM provider."""

    supports_json_mode: bool
    supports_tools: bool
    max_tokens: int
    supports_streaming: bool = False


def build_traversal_tools(children: List["CategoryNode"]) -> List[Dict[str, Any]]:
    """Generate tool schemas for constrained traversal.

    Args:
        children: List of CategoryNode children to choose from.

    Returns:
        List of tool schemas in OpenAI function calling format.
    """
    child_names = [c.name for c in children]

    return [
        {
            "type": "function",
            "function": {
                "name": "select_child",
                "description": "Select the most appropriate child category for the input text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "child_name": {
                            "type": "string",
                            "enum": child_names,
                            "description": "Name of the selected child category",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence score for this selection",
                        },
                    },
                    "required": ["child_name", "confidence"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "abstain",
                "description": "Indicate that none of the options fit well - abstain from classification",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why none of the categories fit",
                        },
                        "closest_options": {
                            "type": "array",
                            "items": {"type": "string", "enum": child_names},
                            "description": "The closest but still unsuitable options",
                        },
                    },
                    "required": ["reason"],
                },
            },
        },
    ]


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM provider implementations must inherit from this class
    and implement the required abstract methods and properties.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier for this provider."""
        pass

    @property
    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Return whether this provider supports tool/function calling."""
        pass

    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Generate a JSON response from the LLM.

        Args:
            system_prompt: System message for the LLM.
            user_prompt: User message for the LLM.
            max_retries: Number of retry attempts for JSON parsing failures.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            ValueError: If valid JSON cannot be generated after retries.
        """
        pass

    @abstractmethod
    def call_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> ToolCallResult:
        """Call LLM with function/tool calling capability.

        Args:
            system_prompt: System message for the LLM.
            user_prompt: User message for the LLM.
            tools: List of tool schemas in OpenAI function calling format.
            tool_choice: Which tool to use ("auto" or specific tool name).

        Returns:
            ToolCallResult with the function name and arguments.

        Raises:
            NotImplementedError: If tool calling is not supported by this provider.
            ValueError: If no tool call is returned in the response.
        """
        pass
