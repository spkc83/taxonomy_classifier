"""OpenAI LLM provider implementation."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import BaseLLMProvider, ToolCallResult

logger = logging.getLogger(__name__)

REASONING_MODELS = frozenset({
    "o1", "o1-mini", "o1-preview", "o1-pro",
    "o3", "o3-mini", "o3-pro",
    "o4-mini",
    "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro",
    "gpt-5.1", "gpt-5.1-codex",
    "gpt-5.2", "gpt-5.2-pro", "gpt-5.2-codex",
})


def _is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model that doesn't support temperature."""
    model_lower = model.lower()
    return any(model_lower.startswith(rm) for rm in REASONING_MODELS)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider using the official openai SDK.

    Supports JSON generation and function/tool calling capabilities.
    Compatible with OpenAI API, Azure OpenAI, and other OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """Initialize the OpenAI provider.

        Args:
            model: The model identifier (e.g., 'gpt-4', 'gpt-4o-mini').
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            base_url: Custom base URL for Azure OpenAI or compatible endpoints.
            temperature: Sampling temperature for responses.
        """
        self._model = model
        self._temperature = temperature
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

        client_kwargs: Dict[str, Any] = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

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

        Args:
            system_prompt: System message for the LLM.
            user_prompt: User message for the LLM.
            max_retries: Number of retry attempts for JSON parsing failures.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            ValueError: If valid JSON cannot be generated after retries.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None
        content = ""

        for attempt in range(max_retries):
            try:
                create_kwargs: Dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                }
                if not _is_reasoning_model(self._model):
                    create_kwargs["temperature"] = self._temperature

                response = self.client.chat.completions.create(**create_kwargs)
                content = response.choices[0].message.content or ""

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
                        "content": "The previous response was not valid JSON. Please return ONLY valid JSON.",
                    }
                )
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}")
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
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tool_choice_param: Any = "auto"
        if tool_choice != "auto":
            tool_choice_param = {"type": "function", "function": {"name": tool_choice}}

        create_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice_param,
        }
        if not _is_reasoning_model(self._model):
            create_kwargs["temperature"] = self._temperature

        response = self.client.chat.completions.create(**create_kwargs)

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise ValueError("No tool call in response")

        tool_call = tool_calls[0]
        return ToolCallResult(
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )

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
