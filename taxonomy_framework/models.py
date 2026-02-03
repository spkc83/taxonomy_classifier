from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Literal, Tuple
import json
import numpy as np

# Type aliases for abstain reasons and suggested actions
AbstainReason = Literal["no_candidates", "neither_in_contrast", "low_confidence", "explicit_abstain"]
SuggestedAction = Literal["request_clarification", "manual_review", "use_best_guess"]

@dataclass
class CategoryNode:
    """A node in a hierarchical taxonomy."""
    name: str
    description: Optional[str] = None
    children: List["CategoryNode"] = field(default_factory=list)
    parent: Optional["CategoryNode"] = field(default=None, repr=False)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        """Return True if this node has no children."""
        return not self.children

    def path(self) -> str:
        """Return the full category path from the root to this node."""
        node = self
        parts = []
        while node is not None:
            if node.name != "Root": # Skip arbitrary virtual root if named Root
                parts.append(node.name)
            node = node.parent
        return " > ".join(reversed(parts))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary (recursive for children)."""
        data = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata
        }
        if self.children:
            data["children"] = [child.to_dict() for child in self.children]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: Optional["CategoryNode"] = None) -> "CategoryNode":
        """Deserialize from dictionary."""
        node = cls(
            name=data["name"],
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            parent=parent
        )
        children_data = data.get("children", [])
        for child_data in children_data:
            child_node = cls.from_dict(child_data, parent=node)
            node.children.append(child_node)
        return node
    
    def __str__(self) -> str:
        return self.name

@dataclass
class ClassificationResult:
    """Standardized output for classification tasks."""
    input_text: str
    predicted_category: CategoryNode
    confidence_score: float
    reasoning: Optional[str] = None
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    traversal_path: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_text": self.input_text,
            "predicted_path": self.predicted_category.path(),
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
            "traversal_path": self.traversal_path
        }


@dataclass
class AbstainResult:
    """Result when classification cannot be determined with confidence."""
    reason: AbstainReason
    top_candidates: List[Tuple["CategoryNode", float]]
    suggested_action: SuggestedAction
    partial_path: Optional[str]
    input_text: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "top_candidates": [
                {"path": node.path(), "confidence": conf}
                for node, conf in self.top_candidates
            ],
            "suggested_action": self.suggested_action,
            "partial_path": self.partial_path,
            "input_text": self.input_text
        }

class Taxonomy:
    """Container for the taxonomy tree."""
    def __init__(self, root: CategoryNode):
        self.root = root
        
    def get_all_nodes(self) -> List[CategoryNode]:
        """Return a flat list of all nodes."""
        nodes = []
        def traverse(node):
            nodes.append(node)
            for child in node.children:
                traverse(child)
        traverse(self.root)
        return nodes
    
    def to_json(self) -> str:
        return json.dumps(self.root.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Taxonomy":
        data = json.loads(json_str)
        root = CategoryNode.from_dict(data)
        return cls(root)
