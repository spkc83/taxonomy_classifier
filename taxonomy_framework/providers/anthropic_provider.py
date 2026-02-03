"""Anthropic Claude LLM provider implementation."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

from .base import BaseLLMProvider, ToolCallResult

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """LLM provider for Anthropic Claude models.
    
    Supports claude-3-opus, claude-3-sonnet, claude-3-haiku models.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_tool_calling(self) -> bool:
        return True

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        json_system_prompt = f"{system_prompt}\n\nYou MUST respond with valid JSON only. No additional text or explanation."
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    system=json_system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                )
                
                content = response.content[0].text
                cleaned = self._clean_json_string(content)
                
                return json.loads(cleaned)
                
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {last_error}")
            except Exception as e:
                logger.error(f"LLM Call failed: {e}")
                raise e
        
        raise ValueError(f"Failed to generate valid JSON after {max_retries} attempts. Last error: {last_error}")

    def _clean_json_string(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def call_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
    ) -> ToolCallResult:
        anthropic_tools = self._convert_tools_to_anthropic_format(tools)
        
        if tool_choice == "auto":
            anthropic_tool_choice = {"type": "auto"}
        elif tool_choice == "any":
            anthropic_tool_choice = {"type": "any"}
        else:
            anthropic_tool_choice = {"type": "tool", "name": tool_choice}
        
        response = self.client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            tools=anthropic_tools,
            tool_choice=anthropic_tool_choice,
        )
        
        for block in response.content:
            if block.type == "tool_use":
                return ToolCallResult(
                    name=block.name,
                    arguments=block.input,
                )
        
        raise ValueError("No tool call returned in response")

    def _convert_tools_to_anthropic_format(
        self, 
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI function calling format to Anthropic tool format.
        
        OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}
        Anthropic: {"name", "description", "input_schema"}
        """
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                anthropic_tools.append(tool)
        
        return anthropic_tools
