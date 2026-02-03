"""Cohere LLM provider implementation."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import cohere

from .base import BaseLLMProvider, ToolCallResult

logger = logging.getLogger(__name__)


class CohereProvider(BaseLLMProvider):
    """Cohere LLM provider using the official cohere SDK.

    Supports JSON generation and function/tool calling capabilities.
    Compatible with Command-R and Command-R-Plus models.
    """

    def __init__(
        self,
        model: str = "command-r",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """Initialize the Cohere provider.

        Args:
            model: The model identifier (e.g., 'command-r', 'command-r-plus').
            api_key: Cohere API key. If not provided, reads from COHERE_API_KEY
                     or CO_API_KEY environment variables.
            temperature: Sampling temperature for responses.
        """
        self._model = model
        self._temperature = temperature
        self._api_key = api_key or os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")

        self.client = cohere.ClientV2(api_key=self._api_key)

    @property
    def model_name(self) -> str:
        """Return the model name/identifier for this provider."""
        return self._model

    @property
    def supports_tool_calling(self) -> bool:
        """Return whether this provider supports tool/function calling."""
        return True

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Generate a JSON response from the LLM.

        Cohere does not have a native JSON mode like OpenAI, so we instruct
        the model via the prompt to return valid JSON.

        Args:
            system_prompt: System message for the LLM.
            user_prompt: User message for the LLM.
            max_retries: Number of retry attempts for JSON parsing failures.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            ValueError: If valid JSON cannot be generated after retries.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{user_prompt}\n\nRespond with valid JSON only. No markdown, no explanation.",
            },
        ]

        last_error: Optional[Exception] = None
        content = ""

        for attempt in range(max_retries):
            try:
                response = self.client.chat(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                )

                content = self._extract_text_content(response)
                clean_content = self._clean_json_string(content)
                return json.loads(clean_content)

            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parse failure (attempt {attempt + 1}/{max_retries}): {e}"
                )
                last_error = e
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": "The previous response was not valid JSON. Please return ONLY valid JSON with no markdown formatting.",
                    }
                )
            except Exception as e:
                logger.error(f"Cohere API call failed: {e}")
                raise

        raise ValueError(
            f"Failed to generate valid JSON after {max_retries} attempts. Last error: {last_error}"
        )

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
            ValueError: If no tool call is returned in the response.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        cohere_tools = [self._convert_tool_to_cohere(tool) for tool in tools]

        response = self.client.chat(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            tools=cohere_tools,
        )

        if not response.message.tool_calls:
            raise ValueError("No tool call in response")

        tool_call = response.message.tool_calls[0]
        return ToolCallResult(
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )

    def _convert_tool_to_cohere(self, tool: Dict[str, Any]) -> cohere.ToolV2:
        """Convert OpenAI-style tool definition to Cohere ToolV2 format.

        Args:
            tool: Tool definition in OpenAI format with 'type' and 'function' keys.

        Returns:
            Cohere ToolV2 object.
        """
        function = tool["function"]
        return cohere.ToolV2(
            type="function",
            function=cohere.ToolV2Function(
                name=function["name"],
                description=function.get("description", ""),
                parameters=function.get("parameters", {}),
            ),
        )

    def _extract_text_content(self, response: Any) -> str:
        """Extract text content from Cohere response.

        Args:
            response: Cohere chat response object.

        Returns:
            Text content as a string.
        """
        if response.message.content:
            text_parts = []
            for item in response.message.content:
                if hasattr(item, "text"):
                    text_parts.append(item.text)
            return "".join(text_parts)
        return ""

    def _clean_json_string(self, text: str) -> str:
        """Remove markdown JSON code blocks if present.

        Args:
            text: Raw text that may contain markdown formatting.

        Returns:
            Cleaned text with markdown code blocks removed.
        """
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
