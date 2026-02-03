# Taxonomy Framework - Python API Reference

This document describes the core Python API for the taxonomy classification framework.

## Core Classes

### CategoryNode

Represents a node in the taxonomy tree.

```python
from taxonomy_framework.models import CategoryNode

node = CategoryNode(
    name="Electronics",
    description="Electronic devices and gadgets",
    parent=root_node,  # Optional, None for root
    metadata={"icon": "🔌"}  # Optional
)
```

**Attributes:**
- `name: str` - Category name
- `description: Optional[str]` - Human-readable description
- `parent: Optional[CategoryNode]` - Parent node reference
- `children: List[CategoryNode]` - Child nodes
- `metadata: Dict[str, Any]` - Additional metadata

**Methods:**
- `is_leaf() -> bool` - Returns True if node has no children
- `path() -> str` - Returns full path (e.g., "Root > Electronics > Phones")
- `depth() -> int` - Returns depth in tree (root = 0)

### Taxonomy

Container for the taxonomy tree.

```python
from taxonomy_framework.models import Taxonomy

taxonomy = Taxonomy(root=root_node)
```

**Attributes:**
- `root: CategoryNode` - Root node of the taxonomy

**Methods:**
- `get_all_leaves() -> List[CategoryNode]` - Returns all leaf nodes
- `get_node_by_path(path: str) -> Optional[CategoryNode]` - Find node by path
- `to_dict() -> Dict` - Serialize to dictionary

### ClassificationResult

Result of a successful classification.

```python
from taxonomy_framework.models import ClassificationResult

# Returned by classifier.classify()
result = ClassificationResult(
    predicted_category=leaf_node,
    confidence_score=0.92,
    path_confidence=[0.98, 0.95, 0.92],
    alternatives=[other_node1, other_node2]
)
```

**Attributes:**
- `predicted_category: CategoryNode` - The predicted leaf category
- `confidence_score: float` - Overall confidence (0-1)
- `path_confidence: List[float]` - Confidence at each level
- `alternatives: List[CategoryNode]` - Alternative candidates

### AbstainResult

Returned when the classifier cannot make a confident prediction.

```python
from taxonomy_framework.models import AbstainResult

result = AbstainResult(
    reason="Ambiguous between multiple categories",
    partial_path="Root > Electronics",
    candidates=[node1, node2],
    confidence_scores=[0.45, 0.42]
)
```

**Attributes:**
- `reason: str` - Why classification was abstained
- `partial_path: Optional[str]` - How far traversal got
- `candidates: List[CategoryNode]` - Potential categories
- `confidence_scores: List[float]` - Scores for each candidate

---

## HybridClassifier

The main classification orchestrator combining embeddings and LLM traversal.

```python
from taxonomy_framework.pipeline import HybridClassifier
from taxonomy_framework.embeddings import SentenceTransformerBackend, EnsembleEmbedder
from taxonomy_framework.providers import ProviderFactory

# Create embedder
backend = SentenceTransformerBackend(model_name="all-MiniLM-L6-v2")
embedder = EnsembleEmbedder([backend])

# Create LLM provider
llm = ProviderFactory.create("openai", model="gpt-4o-mini", api_key="sk-...")

# Create classifier
classifier = HybridClassifier(
    taxonomy=taxonomy,
    embedder=embedder,
    llm=llm,
    top_k=3,                    # Top-K candidates from embedding retrieval
    confidence_threshold=0.7,   # Minimum confidence to classify
    max_depth=None              # Maximum traversal depth (None = no limit)
)

# Classify
result = classifier.classify("iPhone 15 Pro with titanium design")

if isinstance(result, ClassificationResult):
    print(f"Category: {result.predicted_category.path()}")
    print(f"Confidence: {result.confidence_score}")
else:  # AbstainResult
    print(f"Abstained: {result.reason}")
    print(f"Partial path: {result.partial_path}")
```

**Constructor Parameters:**
- `taxonomy: Taxonomy` - The taxonomy to classify into
- `embedder: EnsembleEmbedder` - Embedding model for retrieval
- `llm: Optional[BaseLLMProvider]` - LLM for traversal (None = embedding-only)
- `top_k: int = 3` - Number of candidates from retrieval
- `confidence_threshold: float = 0.7` - Minimum confidence
- `max_depth: Optional[int] = None` - Max traversal depth

**Methods:**
- `classify(text: str) -> Union[ClassificationResult, AbstainResult]`

---

## Embedding Backends

### SentenceTransformerBackend

Uses pre-trained sentence transformers.

```python
from taxonomy_framework.embeddings import SentenceTransformerBackend

backend = SentenceTransformerBackend(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"  # or "cpu"
)

# Get embeddings
embeddings = backend.encode(["text 1", "text 2"])
```

### SetFitBackend

Uses fine-tuned SetFit models.

```python
from taxonomy_framework.embeddings import SetFitBackend

# Load from fine-tuned model
backend = SetFitBackend.from_finetuned("./models/my-setfit-model")

# Or wrap existing SetFit model
backend = SetFitBackend(model=setfit_model)
```

### EnsembleEmbedder

Combines multiple embedding backends.

```python
from taxonomy_framework.embeddings import EnsembleEmbedder

embedder = EnsembleEmbedder(
    backends=[backend1, backend2],
    weights=[0.7, 0.3]  # Optional weighting
)

# Retrieve top candidates
candidates = embedder.retrieve(
    query="iPhone 15",
    candidates=leaf_nodes,
    top_k=5
)
```

---

## LLM Providers

### ProviderFactory

Factory for creating LLM provider instances.

```python
from taxonomy_framework.providers import ProviderFactory

# List available providers
print(ProviderFactory.list_providers())
# ['openai', 'anthropic', 'google', 'cohere', 'ollama', 'vllm', 'huggingface_tgi']

# Create provider
provider = ProviderFactory.create(
    "openai",
    model="gpt-4o-mini",
    api_key="sk-...",
    api_base=None  # Optional custom endpoint
)

# Check capabilities
print(provider.supports_tool_calling)  # True
print(provider.capabilities)  # ProviderCapabilities dataclass
```

### BaseLLMProvider Interface

All providers implement this interface:

```python
from taxonomy_framework.providers import BaseLLMProvider

class BaseLLMProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str: ...
    
    @property
    @abstractmethod
    def supports_tool_calling(self) -> bool: ...
    
    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...
    
    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3
    ) -> Dict[str, Any]: ...
    
    @abstractmethod
    def call_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_choice: Optional[str] = None
    ) -> ToolCallResult: ...
```

---

## Training

### SetFitTrainer

Fine-tune SetFit models for improved classification.

```python
from taxonomy_framework.training import SetFitTrainer

trainer = SetFitTrainer(
    base_model="sentence-transformers/paraphrase-mpnet-base-v2",
    output_dir="./models/my-setfit"
)

# Train on dataset
trainer.train(
    dataset=train_dataset,
    text_column="text",
    label_column="category",
    num_epochs=1,
    batch_size=16
)

# Save model
trainer.save()

# Predict
predictions = trainer.predict(["text 1", "text 2"])
```

---

## Evaluation

### Hierarchical Metrics

```python
from taxonomy_framework.evaluation import compute_hierarchical_metrics

# y_true and y_pred are lists of paths
y_true = [["L1", "L2", "L3"], ["L1", "L2", "L4"]]
y_pred = [["L1", "L2", "L3"], ["L1", "L2", "L5"]]

metrics = compute_hierarchical_metrics(
    y_true=y_true,
    y_pred=y_pred,
    level_names=["L1", "L2", "L3"]
)

print(f"Hierarchical F1: {metrics.hierarchical_f1:.4f}")
print(f"Exact Match: {metrics.exact_match_accuracy:.4f}")
print(f"Per-level accuracy: {metrics.per_level_accuracy}")
```

**HierarchicalMetrics Attributes:**
- `hierarchical_precision: float`
- `hierarchical_recall: float`
- `hierarchical_f1: float`
- `per_level_accuracy: Dict[str, float]`
- `per_level_precision: Dict[str, float]`
- `per_level_recall: Dict[str, float]`
- `per_level_f1: Dict[str, float]`
- `exact_match_accuracy: float`
- `depth_weighted_accuracy: float`
- `num_samples: int`

---

## Configuration

### Settings

Uses Pydantic Settings for configuration management.

```python
from taxonomy_framework.config import Settings

# Loads from environment variables
settings = Settings()

print(settings.llm_provider)      # "openai"
print(settings.llm_model)         # "gpt-4o-mini"
print(settings.openai_api_key)    # "sk-..."
```

**Environment Variables:**
- `LLM_PROVIDER` - Provider name
- `LLM_MODEL` - Model identifier
- `API_BASE_URL` - Custom API endpoint
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. - API keys
- `JWT_SECRET_KEY` - For API authentication
- `API_KEYS` - Comma-separated valid API keys

---

## Error Handling

```python
from taxonomy_framework.models import ClassificationResult, AbstainResult

result = classifier.classify(text)

if isinstance(result, ClassificationResult):
    # Successful classification
    category = result.predicted_category
    confidence = result.confidence_score
elif isinstance(result, AbstainResult):
    # Classification abstained
    reason = result.reason
    partial = result.partial_path
    candidates = result.candidates
```

Common abstain reasons:
- "Ambiguous between multiple categories"
- "Confidence below threshold"
- "No matching categories found"
- "Maximum depth reached without leaf"
