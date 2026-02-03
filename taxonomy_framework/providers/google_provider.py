"""Google Gemini LLM provider implementation."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .base import BaseLLMProvider, ToolCallResult

logger = logging.getLogger(__name__)


class GoogleProvider(BaseLLMProvider):
    """Google Gemini LLM provider.

    Supports gemini-1.5-pro and gemini-1.5-flash models with JSON generation
    and function/tool calling capabilities.
    """

    # Permissive safety settings for classification tasks
    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """Initialize the Google Gemini provider.

        Args:
            model: Model name (e.g., "gemini-1.5-flash", "gemini-1.5-pro").
            api_key: Google API key. If not provided, reads from GOOGLE_API_KEY env var.
            temperature: Sampling temperature (0.0-1.0). Default is 0.0 for deterministic output.
        """
        self._model = model
        self._temperature = temperature

        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key must be provided either directly or via GOOGLE_API_KEY environment variable"
            )

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    @property
    def model_name(self) -> str:
        """Return the model name/identifier for this provider."""
        return self._model

    @property
    def supports_tool_calling(self) -> bool:
        """Return whether this provider supports tool/function calling."""
        return True

    def _clean_json_string(self, text: str) -> str:
        """Remove markdown JSON code blocks if present."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

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
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        generation_config = genai.GenerationConfig(
            temperature=self._temperature,
            response_mime_type="application/json",
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(
                    combined_prompt,
                    generation_config=generation_config,
                    safety_settings=self.SAFETY_SETTINGS,
                )

                if not response.text:
                    raise ValueError("Empty response from Gemini")

                cleaned_text = self._clean_json_string(response.text)
                return json.loads(cleaned_text)

            except json.JSONDecodeError as e:
                last_error = f"JSON parsing failed: {e}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {last_error}")

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
        gemini_tools = self._convert_tools_to_gemini_format(tools)
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        generation_config = genai.GenerationConfig(
            temperature=self._temperature,
        )

        response = self.client.generate_content(
            combined_prompt,
            generation_config=generation_config,
            tools=gemini_tools,
            safety_settings=self.SAFETY_SETTINGS,
        )

        if not response.candidates:
            raise ValueError("No response candidates from Gemini")

        candidate = response.candidates[0]
        if not candidate.content.parts:
            raise ValueError("No content parts in Gemini response")

        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                func_call = part.function_call
                return ToolCallResult(
                    name=func_call.name,
                    arguments=dict(func_call.args),
                )

        raise ValueError("No tool call returned in Gemini response")

    def _convert_tools_to_gemini_format(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tool schemas to Gemini format.

        Args:
            tools: Tools in OpenAI function calling format.

        Returns:
            Tools in Gemini-compatible format.
        """
        gemini_functions = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                gemini_func = {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }
                gemini_functions.append(gemini_func)

        return [genai.protos.Tool(function_declarations=gemini_functions)]
