from abc import ABC, abstractmethod
from typing import List, Union, Dict
import os
import numpy as np
from .utils import logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from setfit import SetFitModel
except ImportError:
    SetFitModel = None

class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        pass

class SentenceTransformerBackend(EmbeddingModel):
    """Wrapper for local SentenceTransformers."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("sentence_transformers not installed. Run: pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)


class SetFitBackend(EmbeddingModel):
    """Wrapper for pre-trained SetFit models as embedders."""
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2"):
        if SetFitModel is None:
            raise ImportError("setfit not installed. Run: pip install setfit")
        self.model = SetFitModel.from_pretrained(model_name)

    @classmethod
    def from_finetuned(cls, path: str) -> "SetFitBackend":
        """Load a fine-tuned SetFit model from a local directory."""
        if not os.path.exists(path):
            raise ValueError(f"Model path does not exist: {path}")
        
        has_config = os.path.isfile(os.path.join(path, "config_setfit.json"))
        has_model_head = os.path.isfile(os.path.join(path, "model_head.pkl"))
        if not (has_config or has_model_head):
            raise ValueError("Invalid model directory: missing model files")
        
        instance = cls.__new__(cls)
        if SetFitModel is None:
            raise ImportError("setfit not installed. Run: pip install setfit")
        instance.model = SetFitModel.from_pretrained(path)
        return instance
    
    def embed_text(self, text: str) -> np.ndarray:
        return self.model.model_body.encode(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.model_body.encode(texts)


class EnsembleEmbedder:
    """
    Mixture of Experts (MoE) for embeddings / Retrieval.
    Uses Reciprocal Rank Fusion (RRF) to combine results from multiple embedding models.
    """
    def __init__(self, models: List[EmbeddingModel]):
        self.models = models
        if not self.models:
            raise ValueError("At least one embedding model is required for Ensemble.")

    def retrieve_candidates(
        self, 
        query: str, 
        candidates_texts: List[str], 
        top_k: int = 5,
        k_rrf: int = 60
    ) -> List[int]:
        """
        Retrieve top_k indices from candidates_texts using RRF across all models.
        
        Args:
            query: Input text.
            candidates_texts: List of strings to search against.
            top_k: Number of results to return.
            k_rrf: Constant for RRF (default 60).
            
        Returns:
            List of indices of the top candidates.
        """
        # Store scores for RRF: index -> score
        rrf_scores: Dict[int, float] = {}
        
        for model in self.models:
            # 1. Compute embeddings
            query_vec = model.embed_text(query)
            cand_vecs = model.embed_batch(candidates_texts)
            
            # 2. Compute Cosine Similarity
            # Normalize first
            norm_q = np.linalg.norm(query_vec)
            norm_c = np.linalg.norm(cand_vecs, axis=1)
            
            # Avoid division by zero
            if norm_q == 0:
                continue
            
            # Dot product
            scores = np.dot(cand_vecs, query_vec) / (norm_c * norm_q + 1e-10)
            
            # 3. Get rank
            # Argsort returns indices of sorted array (ascending), so we take tail and reverse
            ranked_indices = np.argsort(scores)[::-1]
            
            # 4. Compute RRF score
            for rank, idx in enumerate(ranked_indices):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank + 1))
                
        # Sort by RRF score descending
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return sorted_indices[:top_k]
