"""Ollama LLM provider implementation for local LLM inference."""

import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import BaseLLMProvider, ToolCallResult

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider using OpenAI-compatible API.

    Ollama exposes an OpenAI-compatible endpoint at /v1, allowing us to
    reuse the OpenAI client for local LLM inference.

    Supports JSON generation and tool calling (for models like llama3, mistral).
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.0,
    ):
        """Initialize the Ollama provider.

        Args:
            model: The model identifier (e.g., 'llama3', 'mistral', 'codellama').
            base_url: Ollama server URL with /v1 endpoint.
            temperature: Sampling temperature for responses.
        """
        self._model = model
        self._temperature = temperature
        self._base_url = base_url

        # Ollama doesn't require a real API key, but OpenAI client needs one
        self.client = OpenAI(api_key="ollama", base_url=base_url)

    @property
    def model_name(self) -> str:
        """Return the model name/identifier for this provider."""
        return self._model

    @property
    def base_url(self) -> str:
        """Return the base URL for the Ollama server."""
        return self._base_url

    @property
    def supports_tool_calling(self) -> bool:
        """Return whether this provider supports tool/function calling.

        Many Ollama models (llama3, mistral, etc.) support tool calling.
        """
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
            ConnectionError: If Ollama server is not reachable.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None
        content = ""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
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
            except ConnectionError as e:
                logger.error(f"Cannot connect to Ollama server at {self._base_url}: {e}")
                raise ConnectionError(
                    f"Ollama server not reachable at {self._base_url}. "
                    "Ensure Ollama is running with 'ollama serve'."
                ) from e
            except Exception as e:
                # Handle connection-related errors from httpx/OpenAI client
                error_str = str(e).lower()
                if "connection" in error_str or "refused" in error_str:
                    logger.error(f"Cannot connect to Ollama server at {self._base_url}: {e}")
                    raise ConnectionError(
                        f"Ollama server not reachable at {self._base_url}. "
                        "Ensure Ollama is running with 'ollama serve'."
                    ) from e
                logger.error(f"Ollama API call failed: {e}")
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
            ConnectionError: If Ollama server is not reachable.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tool_choice_param: Any = "auto"
        if tool_choice != "auto":
            tool_choice_param = {"type": "function", "function": {"name": tool_choice}}

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self._temperature,
                tools=tools,  # type: ignore[arg-type]
                tool_choice=tool_choice_param,
            )
        except ConnectionError as e:
            logger.error(f"Cannot connect to Ollama server at {self._base_url}: {e}")
            raise ConnectionError(
                f"Ollama server not reachable at {self._base_url}. "
                "Ensure Ollama is running with 'ollama serve'."
            ) from e
        except Exception as e:
            # Handle connection-related errors from httpx/OpenAI client
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str:
                logger.error(f"Cannot connect to Ollama server at {self._base_url}: {e}")
                raise ConnectionError(
                    f"Ollama server not reachable at {self._base_url}. "
                    "Ensure Ollama is running with 'ollama serve'."
                ) from e
            raise

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
