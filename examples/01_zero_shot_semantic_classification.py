"""
Zero-Shot Semantic Classification - No Training Required

This example demonstrates immediate classification using pre-trained
SentenceTransformer models without any fine-tuning.

What you'll learn:
1. Loading a pre-trained SentenceTransformer model directly
2. Building a simple taxonomy for demonstration
3. Performing semantic similarity-based classification
4. Zero-shot classification without any training data

Key Features:
- LIGHTWEIGHT: Uses only sentence-transformers (no SetFit, no training)
- FAST: Pure inference, no model training required
- MEMORY-EFFICIENT: Uses all-MiniLM-L6-v2 (~80MB model)
- SIMPLE: Easy to understand and adapt

Best For:
- Quick prototyping and experimentation
- Domains with clear semantic distinctions
- When you don't have labeled training data
- Understanding the basic classification workflow

Requirements:
    pip install sentence-transformers numpy

Usage:
    python examples/01_zero_shot_semantic_classification.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers not installed. Run: pip install sentence-transformers"
    )

from taxonomy_framework.models import CategoryNode, Taxonomy


@dataclass
class InferenceResult:
    """Result of semantic similarity classification."""
    input_text: str
    predicted_category: str
    predicted_path: str
    confidence: float
    top_k_results: List[Tuple[str, float]]


class SimpleSemanticClassifier:
    """
    Simple zero-shot classifier using SentenceTransformer embeddings.
    
    Classifies text by computing cosine similarity between the input
    and all category descriptions/names in the taxonomy.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            model_name: SentenceTransformer model to use. Defaults to lightweight
                        all-MiniLM-L6-v2 (~80MB, fast inference).
            device: Device to run on ('cpu', 'cuda', or None for auto-detect).
        """
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.taxonomy: Optional[Taxonomy] = None
        self.leaf_nodes: List[CategoryNode] = []
        self.leaf_embeddings: Optional[np.ndarray] = None
        self.category_texts: List[str] = []
        
    def load_taxonomy(self, taxonomy: Taxonomy) -> None:
        """
        Load taxonomy and pre-compute embeddings for all leaf categories.
        
        Args:
            taxonomy: The taxonomy structure to use for classification.
        """
        self.taxonomy = taxonomy
        self.leaf_nodes = []
        self._collect_leaves(taxonomy.root)
        
        print(f"Found {len(self.leaf_nodes)} leaf categories")
        
        self.category_texts = []
        for node in self.leaf_nodes:
            path = node.path()
            desc = node.description or node.name
            text = f"{path}: {desc}"
            self.category_texts.append(text)
        
        print("Computing category embeddings...")
        self.leaf_embeddings = self.model.encode(
            self.category_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print("Category embeddings computed successfully")
    
    def _collect_leaves(self, node: CategoryNode) -> None:
        """Recursively collect all leaf nodes."""
        if node.is_leaf():
            self.leaf_nodes.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child)
    
    def classify(
        self,
        text: str,
        top_k: int = 3
    ) -> InferenceResult:
        """
        Classify input text using semantic similarity.
        
        Args:
            text: Input text to classify.
            top_k: Number of top candidates to return.
            
        Returns:
            InferenceResult with predicted category and confidence.
        """
        if self.leaf_embeddings is None:
            raise ValueError("Taxonomy not loaded. Call load_taxonomy() first.")
        
        text_embedding = self.model.encode(text, convert_to_numpy=True)
        similarities = self._cosine_similarity(text_embedding, self.leaf_embeddings)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        top_k_results = [
            (self.leaf_nodes[idx].path(), float(similarities[idx]))
            for idx in top_indices
        ]
        
        best_idx = top_indices[0]
        best_node = self.leaf_nodes[best_idx]
        
        return InferenceResult(
            input_text=text,
            predicted_category=best_node.name,
            predicted_path=best_node.path(),
            confidence=float(similarities[best_idx]),
            top_k_results=top_k_results
        )
    
    def classify_batch(
        self,
        texts: List[str],
        top_k: int = 3,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[InferenceResult]:
        """
        Classify multiple texts efficiently using batched encoding.
        
        Args:
            texts: List of input texts to classify.
            top_k: Number of top candidates per text.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
            
        Returns:
            List of InferenceResult objects.
        """
        if self.leaf_embeddings is None:
            raise ValueError("Taxonomy not loaded. Call load_taxonomy() first.")
        
        text_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        results = []
        for i, text_emb in enumerate(text_embeddings):
            similarities = self._cosine_similarity(text_emb, self.leaf_embeddings)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            top_k_results = [
                (self.leaf_nodes[idx].path(), float(similarities[idx]))
                for idx in top_indices
            ]
            
            best_idx = top_indices[0]
            best_node = self.leaf_nodes[best_idx]
            
            results.append(InferenceResult(
                input_text=texts[i],
                predicted_category=best_node.name,
                predicted_path=best_node.path(),
                confidence=float(similarities[best_idx]),
                top_k_results=top_k_results
            ))
        
        return results
    
    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a vector and a matrix of vectors."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
        matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        return np.dot(matrix_norms, vec_norm)


def build_demo_taxonomy() -> Taxonomy:
    """Build a simple demonstration taxonomy."""
    root = CategoryNode(
        name="Root",
        description="Product Categories"
    )
    
    electronics = CategoryNode(
        name="Electronics",
        description="Electronic devices and gadgets",
        parent=root
    )
    root.children.append(electronics)
    
    phones = CategoryNode(
        name="Smartphones",
        description="Mobile phones and smartphones with touchscreens",
        parent=electronics
    )
    laptops = CategoryNode(
        name="Laptops",
        description="Portable computers and notebooks",
        parent=electronics
    )
    headphones = CategoryNode(
        name="Headphones",
        description="Audio devices worn on ears, earbuds, and headsets",
        parent=electronics
    )
    electronics.children.extend([phones, laptops, headphones])
    
    clothing = CategoryNode(
        name="Clothing",
        description="Apparel and wearable fashion items",
        parent=root
    )
    root.children.append(clothing)
    
    shirts = CategoryNode(
        name="Shirts",
        description="Tops, t-shirts, and button-down shirts",
        parent=clothing
    )
    pants = CategoryNode(
        name="Pants",
        description="Trousers, jeans, and legwear",
        parent=clothing
    )
    shoes = CategoryNode(
        name="Shoes",
        description="Footwear including sneakers, boots, and sandals",
        parent=clothing
    )
    clothing.children.extend([shirts, pants, shoes])
    
    books = CategoryNode(
        name="Books",
        description="Written publications and literature",
        parent=root
    )
    root.children.append(books)
    
    fiction = CategoryNode(
        name="Fiction",
        description="Novels, stories, and imaginative literature",
        parent=books
    )
    nonfiction = CategoryNode(
        name="Non-Fiction",
        description="Factual books, biographies, and educational texts",
        parent=books
    )
    textbooks = CategoryNode(
        name="Textbooks",
        description="Academic and educational course materials",
        parent=books
    )
    books.children.extend([fiction, nonfiction, textbooks])
    
    return Taxonomy(root)


def print_taxonomy_tree(node: CategoryNode, indent: int = 0) -> None:
    """Print taxonomy structure as a tree."""
    prefix = "  " * indent + ("└─ " if indent > 0 else "")
    print(f"{prefix}{node.name}")
    for child in node.children:
        print_taxonomy_tree(child, indent + 1)


def main():
    """Run the simple inference example."""
    print("=" * 70)
    print("Simple Inference Example - Zero-Shot Classification")
    print("=" * 70)
    
    print("\n[1/3] Building demonstration taxonomy...")
    taxonomy = build_demo_taxonomy()
    print("\nTaxonomy structure:")
    print_taxonomy_tree(taxonomy.root)
    
    print("\n[2/3] Initializing classifier...")
    classifier = SimpleSemanticClassifier(model_name="all-MiniLM-L6-v2")
    classifier.load_taxonomy(taxonomy)
    
    print("\n[3/3] Running classification demo...")
    demo_texts = [
        "iPhone 15 Pro with A17 chip and titanium design",
        "Classic blue denim jeans with five pockets",
        "The Great Gatsby by F. Scott Fitzgerald",
        "MacBook Air M2 with 16GB RAM",
        "Introduction to Machine Learning textbook",
        "Nike Air Jordan basketball sneakers",
        "Wireless Bluetooth earbuds with noise cancellation",
        "Biography of Albert Einstein",
        "Cotton polo shirt in navy blue",
    ]
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)
    
    for text in demo_texts:
        result = classifier.classify(text, top_k=3)
        
        print(f"\n📝 Input: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        print(f"✅ Predicted: {result.predicted_path}")
        print(f"📊 Confidence: {result.confidence:.4f}")
        print("   Top alternatives:")
        for path, conf in result.top_k_results[1:]:
            print(f"      - {path}: {conf:.4f}")
    
    print("\n" + "=" * 70)
    print("BATCH CLASSIFICATION (More Efficient)")
    print("=" * 70)
    
    batch_results = classifier.classify_batch(demo_texts, top_k=1)
    
    print(f"\nProcessed {len(batch_results)} texts in batch mode:")
    for result in batch_results:
        status = "✓" if result.confidence > 0.3 else "?"
        print(f"  {status} {result.predicted_category:15} ({result.confidence:.2f}) - {result.input_text[:40]}...")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nNotes:")
    print("- This example uses zero-shot classification (no training required)")
    print("- Works well for general semantic similarity matching")
    print("- For better accuracy on specific domains, consider fine-tuning")


if __name__ == "__main__":
    main()
