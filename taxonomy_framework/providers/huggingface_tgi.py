"""HuggingFace Text Generation Inference (TGI) provider implementation."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .base import BaseLLMProvider, ToolCallResult

logger = logging.getLogger(__name__)


class HuggingFaceTGIProvider(BaseLLMProvider):
    """HuggingFace TGI provider supporting both HF Inference API and local TGI servers.

    This provider supports two modes:
    1. HuggingFace Inference API (when base_url is not provided)
    2. Local TGI server with OpenAI-compatible endpoint (when base_url is provided)

    TGI has limited tool calling support, so this provider uses JSON-based
    fallback for tool calls.
    """

    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """Initialize the HuggingFace TGI provider.

        Args:
            model: The model identifier (e.g., 'mistralai/Mistral-7B-Instruct-v0.2').
            base_url: Optional URL for local TGI server. If provided, uses OpenAI-compatible API.
            api_key: Optional API key. If not provided, reads from HF_TOKEN or HUGGINGFACE_API_KEY env var.
            temperature: Sampling temperature for responses.
        """
        self._model = model
        self._temperature = temperature
        self._base_url = base_url

        # Resolve API key from env vars if not provided
        self._api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

        if base_url:
            # Use OpenAI-compatible endpoint for local TGI server
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self._api_key or "EMPTY",
                base_url=base_url,
            )
            self._use_openai_compat = True
        else:
            # Use HuggingFace Inference API
            from huggingface_hub import InferenceClient

            self.client = InferenceClient(model=model, token=self._api_key)
            self._use_openai_compat = False

    @property
    def model_name(self) -> str:
        """Return the model name/identifier for this provider."""
        return self._model

    @property
    def base_url(self) -> Optional[str]:
        """Return the base URL for the TGI server, if configured."""
        return self._base_url

    @property
    def supports_tool_calling(self) -> bool:
        """Return whether this provider supports tool/function calling.

        TGI has limited tool support, so we return False as a safer default.
        Tool calls are handled via JSON-based fallback.
        """
        return False

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
            ConnectionError: If the TGI server is not reachable.
        """
        # Enhance prompt to request JSON output
        json_system_prompt = f"{system_prompt}\n\nIMPORTANT: Respond ONLY with valid JSON. No explanations or markdown."

        last_error: Optional[Exception] = None
        content = ""

        for attempt in range(max_retries):
            try:
                if self._use_openai_compat:
                    content = self._generate_openai_compat(json_system_prompt, user_prompt)
                else:
                    content = self._generate_hf_inference(json_system_prompt, user_prompt)

                clean_content = self._clean_json_string(content)
                return json.loads(clean_content)

            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parse failure (attempt {attempt + 1}/{max_retries}): {e}"
                )
                last_error = e
                # Add retry instruction to prompt for next attempt
                user_prompt = (
                    f"{user_prompt}\n\nPrevious response was not valid JSON. "
                    "Please return ONLY valid JSON with no additional text."
                )
            except Exception as e:
                error_str = str(e).lower()
                if "connection" in error_str or "refused" in error_str:
                    endpoint = self._base_url or "HuggingFace Inference API"
                    logger.error(f"Cannot connect to {endpoint}: {e}")
                    raise ConnectionError(
                        f"Cannot connect to {endpoint}. "
                        "Ensure the server is running and accessible."
                    ) from e
                logger.error(f"HuggingFace TGI API call failed: {e}")
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

        Since TGI has limited tool support, this method uses a JSON-based fallback
        approach where the tool schemas are included in the prompt.

        Args:
            system_prompt: System message for the LLM.
            user_prompt: User message for the LLM.
            tools: List of tool schemas in OpenAI function calling format.
            tool_choice: Which tool to use ("auto" or specific tool name).

        Returns:
            ToolCallResult with the function name and arguments.

        Raises:
            ValueError: If no valid tool call can be extracted from the response.
        """
        # Build a prompt that includes tool definitions and requests JSON output
        tool_descriptions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                tool_descriptions.append(
                    f"- {func['name']}: {func.get('description', 'No description')}\n"
                    f"  Parameters: {json.dumps(func.get('parameters', {}))}"
                )

        tools_text = "\n".join(tool_descriptions)

        tool_system_prompt = f"""{system_prompt}

You have access to the following tools:
{tools_text}

To use a tool, respond with ONLY a JSON object in this exact format:
{{"tool": "<tool_name>", "arguments": {{<tool_arguments>}}}}

Do not include any other text, explanation, or markdown formatting."""

        if tool_choice != "auto":
            tool_system_prompt += f"\n\nYou MUST use the tool: {tool_choice}"

        try:
            if self._use_openai_compat:
                content = self._generate_openai_compat(tool_system_prompt, user_prompt)
            else:
                content = self._generate_hf_inference(tool_system_prompt, user_prompt)

            clean_content = self._clean_json_string(content)
            result = json.loads(clean_content)

            # Extract tool name and arguments from the response
            tool_name = result.get("tool") or result.get("name") or result.get("function")
            arguments = result.get("arguments") or result.get("parameters") or {}

            if not tool_name:
                raise ValueError(f"No tool name found in response: {content}")

            return ToolCallResult(name=tool_name, arguments=arguments)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse tool call response as JSON: {e}. Response: {content}")
        except Exception as e:
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str:
                endpoint = self._base_url or "HuggingFace Inference API"
                raise ConnectionError(
                    f"Cannot connect to {endpoint}. "
                    "Ensure the server is running and accessible."
                ) from e
            raise

    def _generate_openai_compat(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using OpenAI-compatible API (for local TGI server).

        Args:
            system_prompt: System message for the LLM.
            user_prompt: User message for the LLM.

        Returns:
            Generated text content.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
        )

        return response.choices[0].message.content or ""

    def _generate_hf_inference(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using HuggingFace Inference API.

        Args:
            system_prompt: System message for the LLM.
            user_prompt: User message for the LLM.

        Returns:
            Generated text content.
        """
        # Combine system and user prompts for text generation
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

        response = self.client.text_generation(
            prompt=full_prompt,
            max_new_tokens=2048,
            temperature=self._temperature if self._temperature > 0 else None,
            do_sample=self._temperature > 0,
        )

        return response

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
