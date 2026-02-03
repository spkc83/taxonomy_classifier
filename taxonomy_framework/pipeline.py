"""
HybridClassifier Pipeline - Main orchestration component.

Implements the new hybrid classification pipeline:
semantic recall → constrained traversal → sibling contrast → abstain/result

Replaces the old HybridClassifier in core.py.
"""
from dataclasses import dataclass
from typing import Union, Optional, List
import logging

from .models import Taxonomy, CategoryNode, ClassificationResult, AbstainResult
from .embeddings import EnsembleEmbedder
from .providers import BaseLLMProvider
from .traverser import ConstrainedTraverser
from .contrast import SiblingContrast


@dataclass
class PipelineConfig:
    """Configuration for hybrid classification pipeline."""
    top_k_entry_branches: int = 3  # How many entry points to try
    ambiguity_threshold: float = 0.1  # For contrast triggering


class HybridClassifier:
    """
    Hybrid classification pipeline: semantic recall → constrained traversal → sibling contrast.
    
    When llm is provided:
        Full pipeline: semantic recall → constrained traversal → sibling contrast
    
    When llm is None (embedding-only mode):
        Embedding-only: semantic recall → best matching leaf category
    
    Replaces the old HybridClassifier in core.py.
    """
    
    def __init__(
        self,
        taxonomy: Taxonomy,
        embedder: EnsembleEmbedder,
        llm: Optional[BaseLLMProvider] = None,
        config: Optional[PipelineConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the hybrid classification pipeline.
        
        Args:
            taxonomy: The taxonomy tree structure.
            embedder: EnsembleEmbedder for semantic recall.
            llm: Optional LLM client for traversal and contrast decisions.
                 If None, runs in embedding-only mode using semantic similarity.
            config: Optional pipeline configuration.
            logger: Optional logger for pipeline logging.
        """
        self.taxonomy = taxonomy
        self.embedder = embedder
        self.llm = llm
        self.config = config or PipelineConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate taxonomy
        self._validate_taxonomy()
        
        if llm is not None:
            self.traverser: Optional[ConstrainedTraverser] = ConstrainedTraverser(
                llm=llm,
                ambiguity_threshold=self.config.ambiguity_threshold,
                logger=self.logger
            )
            self.contrast: Optional[SiblingContrast] = SiblingContrast(llm=llm)
        else:
            self.traverser = None
            self.contrast = None
            self.logger.info("Running in embedding-only mode (no LLM configured)")
        
        self.entry_branches = self.taxonomy.root.children
        self.entry_texts = [
            f"{n.name}: {n.description or ''}" for n in self.entry_branches
        ]
        
        self._leaf_nodes: List[CategoryNode] = []
        self._leaf_texts: List[str] = []
        if llm is None:
            self._cache_leaf_nodes()
    
    def _validate_taxonomy(self) -> None:
        """Validate taxonomy structure on construction.
        
        Raises:
            ValueError: If taxonomy is invalid (e.g., circular references).
        """
        # Check for circular references
        visited = set()
        
        def check_node(node: CategoryNode, path: List[str]) -> None:
            node_id = id(node)
            if node_id in visited:
                raise ValueError(f"Circular reference detected: {' > '.join(path)}")
            visited.add(node_id)
            for child in node.children:
                check_node(child, path + [child.name])
        
        check_node(self.taxonomy.root, [self.taxonomy.root.name])
    
    def _cache_leaf_nodes(self) -> None:
        """Cache all leaf nodes for embedding-only mode."""
        def collect_leaves(node: CategoryNode) -> None:
            if node.is_leaf():
                self._leaf_nodes.append(node)
                path = node.path()
                desc = node.description or node.name
                self._leaf_texts.append(f"{path}: {desc}")
            else:
                for child in node.children:
                    collect_leaves(child)
        
        collect_leaves(self.taxonomy.root)
        self.logger.info(f"Cached {len(self._leaf_nodes)} leaf categories for embedding-only mode")
    
    def _classify_embedding_only(self, text: str) -> Union[ClassificationResult, AbstainResult]:
        """Classify using only semantic similarity (no LLM traversal)."""
        if not self._leaf_nodes:
            return AbstainResult(
                reason="no_candidates",
                top_candidates=[],
                suggested_action="manual_review",
                partial_path=None,
                input_text=text
            )
        
        top_indices = self.embedder.retrieve_candidates(
            query=text,
            candidates_texts=self._leaf_texts,
            top_k=3
        )
        
        if not top_indices:
            return AbstainResult(
                reason="no_candidates",
                top_candidates=[],
                suggested_action="manual_review",
                partial_path=None,
                input_text=text
            )
        
        best_node = self._leaf_nodes[top_indices[0]]
        
        path: List[str] = []
        node: Optional[CategoryNode] = best_node
        while node is not None and node.name != "Root":
            path.insert(0, node.name)
            node = node.parent
        
        return ClassificationResult(
            input_text=text,
            predicted_category=best_node,
            confidence_score=0.5,
            reasoning=None,
            alternatives=[
                {"path": self._leaf_nodes[i].path(), "confidence": 0.3}
                for i in top_indices[1:] if i < len(self._leaf_nodes)
            ],
            traversal_path=path
        )
    
    def classify(self, text: str) -> Union[ClassificationResult, AbstainResult]:
        """
        Classify text using the hybrid pipeline.
        
        Pipeline:
        1. Semantic recall: Use embedder to find top-K entry branches
        2. For each entry branch (in priority order):
           a. Traverse using ConstrainedTraverser
           b. If traverser abstains → try next branch
           c. If needs_contrast → call SiblingContrast
              - If contrast says "neither" → try next branch
           d. If valid → return ClassificationResult
        3. If all branches exhausted → return AbstainResult
        
        Args:
            text: Input text to classify.
            
        Returns:
            ClassificationResult on success, AbstainResult if cannot classify.
        """
        self.logger.info(f"Classifying: '{text[:50]}...'")
        
        if self.traverser is None:
            return self._classify_embedding_only(text)
        
        top_indices = self.embedder.retrieve_candidates(
            query=text,
            candidates_texts=self.entry_texts,
            top_k=self.config.top_k_entry_branches
        )
        
        if not top_indices:
            return AbstainResult(
                reason="no_candidates",
                top_candidates=[],
                suggested_action="manual_review",
                partial_path=None,
                input_text=text
            )
        
        candidate_branches = [self.entry_branches[i] for i in top_indices]
        self.logger.info(f"Entry candidates: {[b.name for b in candidate_branches]}")
        
        for entry_branch in candidate_branches:
            self.logger.info(f"Trying entry branch: {entry_branch.name}")
            
            traversal_result = self.traverser.traverse(text, entry_branch)
            
            if traversal_result.did_abstain:
                self.logger.info(f"Traverser abstained: {traversal_result.abstain_reason}")
                continue
            
            if traversal_result.needs_contrast and traversal_result.contrast_candidates:
                self.logger.info("Running sibling contrast")
                if self.contrast is None:
                    final_node = traversal_result.final_node
                else:
                    contrast_result = self.contrast.contrast(
                        text, traversal_result.contrast_candidates
                    )
                    
                    if contrast_result.is_neither or contrast_result.choice is None:
                        self.logger.info("Contrast returned 'neither', trying next branch")
                        continue
                    
                    final_node = contrast_result.choice
            else:
                final_node = traversal_result.final_node
            
            return ClassificationResult(
                input_text=text,
                predicted_category=final_node,
                confidence_score=traversal_result.confidence,
                reasoning=None,
                alternatives=[],
                traversal_path=traversal_result.path
            )
        
        return AbstainResult(
            reason="low_confidence",
            top_candidates=[
                (self.entry_branches[i], 1.0 / (idx + 1))
                for idx, i in enumerate(top_indices[:2])
            ] if top_indices else [],
            suggested_action="request_clarification",
            partial_path=" > ".join(
                [self.taxonomy.root.name] + [b.name for b in candidate_branches[:1]]
            ) if candidate_branches else None,
            input_text=text
        )
