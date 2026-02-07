from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional, Tuple
import hashlib
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

    @property
    def model_id(self) -> str:
        """Unique identifier for cache keying. Override in subclasses."""
        return self.__class__.__name__

class SentenceTransformerBackend(EmbeddingModel):
    """Wrapper for local SentenceTransformers."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", query_prefix: str = "", device: Optional[str] = None):
        """
        Initialize the SentenceTransformer backend.

        Args:
            model_name: SentenceTransformer model to use.
            query_prefix: Prefix to prepend to query texts (e.g. "Represent this sentence: "
                          for BGE models). NOT applied to candidate/passage texts.
            device: Device to load model on (e.g. "cuda", "cpu"). None for auto-detect.
        """
        if SentenceTransformer is None:
            raise ImportError("sentence_transformers not installed. Run: pip install sentence-transformers")
        self._model_name = model_name
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name, device=device)

    @property
    def model_id(self) -> str:
        return self._model_name
    
    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)


class SetFitBackend(EmbeddingModel):
    """Wrapper for pre-trained SetFit models as embedders."""
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2"):
        if SetFitModel is None:
            raise ImportError("setfit not installed. Run: pip install setfit")
        self._model_name = model_name
        self.model = SetFitModel.from_pretrained(model_name)

    @property
    def model_id(self) -> str:
        return self._model_name

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
        instance._model_name = path
        instance.model = SetFitModel.from_pretrained(path)
        return instance
    
    def embed_text(self, text: str) -> np.ndarray:
        return self.model.model_body.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.model_body.encode(texts, normalize_embeddings=True)


def _texts_cache_key(texts: List[str]) -> str:
    """Compute a stable hash for a list of candidate texts."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:16]


class EnsembleEmbedder:
    """
    Mixture of Experts (MoE) for embeddings / Retrieval.
    Uses Reciprocal Rank Fusion (RRF) to combine results from multiple embedding models.

    Supports pre-computed candidate embedding caches to avoid redundant work.
    Call `build_index(candidates_texts)` once, then every `retrieve_candidates`
    call reuses the cached vectors and only embeds the query.
    """
    def __init__(self, models: List[EmbeddingModel]):
        self.models = models
        if not self.models:
            raise ValueError("At least one embedding model is required for Ensemble.")
        self._index_cache: Dict[int, Tuple[str, np.ndarray]] = {}

    def build_index(self, candidates_texts: List[str]) -> None:
        """Pre-compute and cache candidate embeddings for all models.

        Call this once after taxonomy loading. Subsequent `retrieve_candidates`
        calls with the same candidates will skip re-embedding.
        """
        cache_key = _texts_cache_key(candidates_texts)
        for i, model in enumerate(self.models):
            existing = self._index_cache.get(i)
            if existing is not None and existing[0] == cache_key:
                continue
            vecs = model.embed_batch(candidates_texts)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            vecs_normed = vecs / norms
            self._index_cache[i] = (cache_key, vecs_normed)
        logger.info(f"Built embedding index for {len(candidates_texts)} candidates")

    def clear_index(self) -> None:
        """Drop cached candidate embeddings."""
        self._index_cache.clear()

    def retrieve_candidates(
        self, 
        query: str, 
        candidates_texts: List[str], 
        top_k: int = 5,
        k_rrf: int = 60
    ) -> List[int]:
        """
        Retrieve top_k indices from candidates_texts using RRF across all models.

        If `build_index` was called with the same candidates_texts, cached
        embeddings are reused and only the query is embedded per call.
        """
        cache_key = _texts_cache_key(candidates_texts)
        rrf_scores: Dict[int, float] = {}
        
        for i, model in enumerate(self.models):
            query_vec = model.embed_text(query)

            cached = self._index_cache.get(i)
            if cached is not None and cached[0] == cache_key:
                cand_vecs_normed = cached[1]
            else:
                cand_vecs = model.embed_batch(candidates_texts)
                norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                cand_vecs_normed = cand_vecs / norms

            norm_q = np.linalg.norm(query_vec)
            if norm_q == 0:
                continue
            query_vec_normed = query_vec / norm_q

            scores = cand_vecs_normed @ query_vec_normed

            ranked_indices = np.argsort(scores)[::-1]
            
            for rank, idx in enumerate(ranked_indices):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank + 1))
                
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return sorted_indices[:top_k]
