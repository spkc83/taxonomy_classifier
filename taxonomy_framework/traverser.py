"""
ConstrainedTraverser - Core component for constrained LLM tree traversal.

Replaces the old "retrieval + free generation + retry" pattern with
"constrained LLM tree traversal" where at each level the LLM chooses
from a constrained enum of child names - it CANNOT hallucinate categories.
"""
from dataclasses import dataclass
from typing import List, Optional
import logging

from .models import CategoryNode
from .providers import BaseLLMProvider, build_traversal_tools


@dataclass
class TraversalResult:
    """Result of a constrained tree traversal."""
    
    final_node: CategoryNode
    path: List[str]  # e.g. ["Root", "Tech", "Hardware"]
    confidence: float  # Final confidence from last LLM call
    needs_contrast: bool  # True if top-2 confidence gap < ambiguity_threshold
    contrast_candidates: Optional[List[CategoryNode]] = None  # Populated when needs_contrast=True
    did_abstain: bool = False  # True if LLM called abstain tool
    abstain_reason: Optional[str] = None


class ConstrainedTraverser:
    """Traverses taxonomy tree with LLM making constrained choices at each level."""
    
    def __init__(
        self,
        llm: BaseLLMProvider,
        ambiguity_threshold: float = 0.1,  # If gap between top-2 < this, needs contrast
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the constrained traverser.
        
        Args:
            llm: LLM client for making traversal decisions.
            ambiguity_threshold: Threshold for determining if contrast comparison is needed.
            logger: Optional logger for traversal logging.
        """
        self.llm = llm
        self.ambiguity_threshold = ambiguity_threshold
        self.logger = logger or logging.getLogger(__name__)
    
    def traverse(self, text: str, entry_node: CategoryNode) -> TraversalResult:
        """
        Traverse from entry_node to a leaf/stopping point.
        
        At each level:
        1. Build tools with build_traversal_tools(current_node.children)
        2. Call llm.call_with_tools() with system+user prompt
        3. Handle response:
           - select_child -> descend to that child
           - abstain -> stop with abstain flag
        4. Stop when leaf reached OR single child (auto-descend)
        
        Args:
            text: Input text to classify.
            entry_node: Starting node for traversal.
            
        Returns:
            TraversalResult with final node, path, confidence, and flags.
        """
        path: List[str] = []
        current = entry_node
        confidence = 0.0
        needs_contrast = False
        
        while True:
            path.append(current.name)
            self.logger.info(
                f"At node: {current.name}, children: {[c.name for c in current.children]}"
            )
            
            # Stop condition: leaf
            if current.is_leaf():
                self.logger.info(f"Reached leaf node: {current.name}")
                return TraversalResult(
                    final_node=current,
                    path=path,
                    confidence=confidence,
                    needs_contrast=needs_contrast
                )
            
            # Stop condition: single child -> auto-descend
            if len(current.children) == 1:
                self.logger.info(
                    f"Auto-descending (single child): {current.children[0].name}"
                )
                current = current.children[0]
                continue
            
            # Build tools and call LLM
            tools = build_traversal_tools(current.children)
            result = self.llm.call_with_tools(
                system_prompt=self._build_system_prompt(),
                user_prompt=self._build_user_prompt(text, current),
                tools=tools
            )
            
            # Handle abstain
            if result.name == "abstain":
                self.logger.info(
                    f"LLM abstained: {result.arguments.get('reason')}"
                )
                return TraversalResult(
                    final_node=current,
                    path=path,
                    confidence=0.0,
                    needs_contrast=False,
                    did_abstain=True,
                    abstain_reason=result.arguments.get("reason")
                )
            
            # Handle select_child
            child_name = result.arguments["child_name"]
            confidence = result.arguments.get("confidence", 0.0)
            self.logger.info(f"LLM selected: {child_name} (confidence: {confidence})")
            
            # Find child node
            child = next((c for c in current.children if c.name == child_name), None)
            if child is None:
                # Should never happen with proper enum constraints
                raise ValueError(f"Invalid child: {child_name}")
            
            # Check if needs contrast (confidence below threshold)
            # If confidence < (1 - ambiguity_threshold), mark as needing contrast
            needs_contrast = confidence < (1 - self.ambiguity_threshold)
            
            current = child
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM traversal decisions."""
        return (
            "You are a classification engine. Your task is to categorize input text by "
            "selecting the most appropriate category from the options provided. "
            "Use select_child to choose a category, or abstain if none fit."
        )
    
    def _build_user_prompt(self, text: str, current_node: CategoryNode) -> str:
        """Build user prompt with input text and available children."""
        children_desc = "\n".join([
            f"- {c.name}: {c.description or 'No description'}"
            for c in current_node.children
        ])
        return (
            f"Input text: \"{text}\"\n\n"
            f"Current category: {current_node.name}\n"
            f"Available sub-categories:\n{children_desc}\n\n"
            "Select the most appropriate sub-category."
        )
