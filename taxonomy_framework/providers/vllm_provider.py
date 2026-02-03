"""vLLM server LLM provider implementation."""

import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import BaseLLMProvider, ToolCallResult

logger = logging.getLogger(__name__)


class VLLMProvider(BaseLLMProvider):
    """vLLM server LLM provider using the OpenAI-compatible API.

    vLLM exposes an OpenAI-compatible API endpoint, allowing us to use
    the standard openai SDK for communication. This provider supports
    arbitrary HuggingFace models served through vLLM.

    Supports JSON generation (with guided decoding hints) and tool calling
    (with JSON schema constraints when available).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
    ):
        """Initialize the vLLM provider.

        Args:
            model: The model identifier being served by vLLM
                   (e.g., 'meta-llama/Llama-2-7b-chat-hf').
            base_url: vLLM server URL with /v1 suffix for OpenAI compatibility.
            api_key: API key for the vLLM server. Default is "EMPTY" as vLLM
                     typically doesn't require authentication.
            temperature: Sampling temperature for responses.
        """
        self._model = model
        self._temperature = temperature
        self._base_url = base_url
        self._api_key = api_key

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @property
    def model_name(self) -> str:
        """Return the model name/identifier for this provider."""
        return self._model

    @property
    def base_url(self) -> str:
        """Return the vLLM server base URL."""
        return self._base_url

    @property
    def supports_tool_calling(self) -> bool:
        """Return whether this provider supports tool/function calling.

        vLLM supports tool calling through its OpenAI-compatible API
        when the underlying model supports it.
        """
        return True

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Generate a JSON response from the LLM.

        Uses vLLM's OpenAI-compatible API with JSON mode when available.
        Falls back to prompting for JSON if json_object response format
        is not supported by the server configuration.

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
                # Try with JSON mode first (vLLM supports this with guided decoding)
                try:
                    response = self.client.chat.completions.create(
                        model=self._model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=self._temperature,
                        response_format={"type": "json_object"},
                    )
                except Exception as json_mode_error:
                    # Fall back to regular completion if JSON mode is not supported
                    logger.debug(
                        f"JSON mode not available, falling back to regular mode: {json_mode_error}"
                    )
                    response = self.client.chat.completions.create(
                        model=self._model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=self._temperature,
                    )

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
                logger.error(f"vLLM API call failed: {e}")
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

        Uses vLLM's OpenAI-compatible tool calling API. vLLM supports
        JSON schema constraints for tool calls when configured.

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

        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
            tools=tools,  # type: ignore[arg-type]
            tool_choice=tool_choice_param,
        )

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
