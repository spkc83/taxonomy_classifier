"""
SiblingContrast module for pairwise comparison between ambiguous sibling categories.

When the traverser identifies that the top-2 candidates are too close,
this module helps decide between them using constrained LLM tool calling.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .models import CategoryNode
from .providers import BaseLLMProvider


@dataclass
class ContrastResult:
    """Result of sibling contrast comparison."""
    choice: Optional[CategoryNode]  # The chosen category, None if "neither"
    is_neither: bool  # True if LLM said "neither" fits
    reasoning: Optional[str] = None  # Explanation from LLM


class SiblingContrast:
    """Performs pairwise comparison between ambiguous sibling categories."""
    
    def __init__(self, llm: BaseLLMProvider):
        """
        Initialize with LLM client.
        
        Args:
            llm: LLM client for making contrast decisions.
        """
        self.llm = llm
    
    def contrast(
        self, 
        text: str, 
        candidates: List[CategoryNode]
    ) -> ContrastResult:
        """
        Compare two candidates and decide which better fits the text.
        
        Args:
            text: Input text to classify.
            candidates: Exactly 2 CategoryNodes to compare.
            
        Returns:
            ContrastResult with choice or neither indication.
            
        Raises:
            ValueError: If not exactly 2 candidates provided.
        """
        if len(candidates) != 2:
            raise ValueError("Exactly 2 candidates required for contrast comparison")
        
        candidate_a, candidate_b = candidates
        
        tools = self._build_contrast_tools()
        result = self.llm.call_with_tools(
            system_prompt=self._build_system_prompt(),
            user_prompt=self._build_user_prompt(text, candidate_a, candidate_b),
            tools=tools
        )
        
        choice_str = result.arguments["choice"]
        reasoning = result.arguments.get("reasoning")
        
        if choice_str == "A":
            return ContrastResult(choice=candidate_a, is_neither=False, reasoning=reasoning)
        elif choice_str == "B":
            return ContrastResult(choice=candidate_b, is_neither=False, reasoning=reasoning)
        else:  # "neither"
            return ContrastResult(choice=None, is_neither=True, reasoning=reasoning)
    
    def _build_contrast_tools(self) -> List[Dict[str, Any]]:
        """Build tool schema for contrast decision.
        
        Returns:
            List containing single tool schema with A/B/neither enum.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "choose_category",
                    "description": "Choose which category better fits the input text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "choice": {
                                "type": "string",
                                "enum": ["A", "B", "neither"],
                                "description": "Your choice: A, B, or neither if both are bad fits"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation for your choice"
                            }
                        },
                        "required": ["choice"]
                    }
                }
            }
        ]
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for contrast comparison.
        
        Returns:
            System prompt string.
        """
        return (
            "You are a classification arbitrator. Given an input text and two category options, "
            "determine which category is the better fit. If neither category fits well, say 'neither'."
        )
    
    def _build_user_prompt(
        self, text: str, candidate_a: CategoryNode, candidate_b: CategoryNode
    ) -> str:
        """Build user prompt with input text and candidate details.
        
        Args:
            text: Input text to classify.
            candidate_a: First category option.
            candidate_b: Second category option.
            
        Returns:
            User prompt string.
        """
        return (
            f"Input text: \"{text}\"\n\n"
            f"Option A: {candidate_a.name}\n"
            f"  Description: {candidate_a.description or 'No description'}\n"
            f"  Path: {candidate_a.path()}\n\n"
            f"Option B: {candidate_b.name}\n"
            f"  Description: {candidate_b.description or 'No description'}\n"
            f"  Path: {candidate_b.path()}\n\n"
            "Which category better fits the input text? Choose A, B, or neither."
        )
