# Examples

This directory contains example scripts demonstrating different aspects of the taxonomy classification framework.

## Overview

| Example | Description | Requirements | Time |
|---------|-------------|--------------|------|
| [01_zero_shot_semantic_classification.py](#1-zero-shot-semantic-classification) | Basic classification using pre-trained embeddings | `sentence-transformers` | ~1 min |
| [02_setfit_finetuning_comparison.py](#2-setfit-fine-tuning-comparison) | Compare baseline vs fine-tuned performance | `setfit`, `datasets` | ~30 min |
| [03_memory_efficient_training.py](#3-memory-efficient-training) | TREC question classification with BGE-small | `setfit`, `datasets` | ~10 min |

## Prerequisites

```bash
# Basic requirements
pip install sentence-transformers numpy

# For fine-tuning examples
pip install setfit datasets torch

# Optional: LLM provider for hybrid classification
pip install openai  # or anthropic, google-generativeai, etc.
```

## 1. Zero-Shot Semantic Classification

**File:** `01_zero_shot_semantic_classification.py`

Demonstrates immediate classification using pre-trained SentenceTransformer models without any fine-tuning.

**Key Concepts:**
- Load pre-trained embedding model
- Build taxonomy from scratch
- Classify via cosine similarity
- No training data required

**Run:**
```bash
python examples/01_zero_shot_semantic_classification.py
```

**Output:**
```
📝 Input: "iPhone 15 Pro with A17 chip and titanium design"
✅ Predicted: Root > Electronics > Smartphones
📊 Confidence: 0.8234
   Top alternatives:
      - Root > Electronics > Laptops: 0.5123
      - Root > Electronics > Headphones: 0.4891
```

**Best For:**
- Quick prototyping
- Domains with clear semantic distinctions
- When you don't have labeled training data

---

## 2. SetFit Fine-tuning Comparison

**File:** `02_setfit_finetuning_comparison.py`

Trains a SetFit model on the DBpedia dataset and compares against a pre-trained baseline.

**Key Concepts:**
- Load real-world dataset (DBpedia Classes)
- Stratified train/test split
- SetFit few-shot fine-tuning
- Hierarchical evaluation metrics

**Run:**
```bash
python examples/02_setfit_finetuning_comparison.py
```

**Output:**
```
==========================================================================
COMPARISON: Baseline (Pre-trained) vs Fine-tuned SetFit
==========================================================================

Metric                              Baseline     Fine-tuned     Δ Change
-----------------------------------------------------------------------
Hierarchical Precision               65.2%         82.4%       +17.2%
Hierarchical Recall                  64.8%         81.9%       +17.1%
Hierarchical F1                      65.0%         82.1%       +17.1%
-----------------------------------------------------------------------
L1 Accuracy                          78.5%         91.2%       +12.7%
L2 Accuracy                          62.3%         79.4%       +17.1%
L3 Accuracy                          54.1%         75.8%       +21.7%
-----------------------------------------------------------------------
Exact Match Accuracy                 54.1%         75.8%       +21.7%

✓ Fine-tuning IMPROVED classification performance
```

**Best For:**
- Maximizing accuracy on specific domains
- When you have labeled training data
- Production systems requiring high precision

---

## 3. Memory-Efficient Training

**File:** `03_memory_efficient_training.py`

Uses the TREC Question Classification dataset (6 coarse → 50 fine classes) with `BAAI/bge-small-en-v1.5` for high-quality embeddings in a small footprint. Baseline accuracy exceeds 80% on coarse labels without any fine-tuning.

**Key Concepts:**
- High-quality model (`BAAI/bge-small-en-v1.5`, 33M params)
- TREC dataset with natural 2-level hierarchy
- Pre-computed embedding index (candidates embedded once, reused per query)
- Sequential model loading to stay within 4GB VRAM
- Explicit memory management with garbage collection

**Run:**
```bash
python examples/03_memory_efficient_training.py            # uses saved model if exists
python examples/03_memory_efficient_training.py --retrain  # force retraining
```

**Best For:**
- Demonstrating strong baseline performance (80%+ without fine-tuning)
- Development on laptops with limited GPU (4GB)
- Understanding the training → evaluation workflow
- CI/CD pipelines

---

## Environment Variables

All examples support configuration via environment variables:

```bash
# LLM Provider (optional - enables hybrid classification)
export LLM_PROVIDER=openai        # or anthropic, google, cohere, ollama
export LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...      # Set appropriate API key
```

Without LLM configuration, examples run in **embedding-only mode** (semantic similarity only, no constrained traversal).

---

## Example Taxonomy Structure

The examples use a simple product taxonomy:

```
Root
├── Electronics
│   ├── Smartphones
│   ├── Laptops
│   └── Headphones
├── Clothing
│   ├── Shirts
│   ├── Pants
│   └── Shoes
└── Books
    ├── Fiction
    ├── Non-Fiction
    └── Textbooks
```

---

## Understanding the Output

### Classification Result

```python
result = classifier.classify("iPhone 15 Pro")

if isinstance(result, ClassificationResult):
    print(result.predicted_category.name)  # "Smartphones"
    print(result.predicted_category.path())  # "Root > Electronics > Smartphones"
    print(result.confidence_score)  # 0.92
```

### Abstain Result

```python
if isinstance(result, AbstainResult):
    print(result.reason)  # "Ambiguous between multiple categories"
    print(result.partial_path)  # "Root > Electronics"
    print(result.candidates)  # [node1, node2]
```

### Hierarchical Metrics

```
Hierarchical Precision: 82.4%
  - Fraction of predicted path that matches true path

Hierarchical Recall: 81.9%
  - Fraction of true path that was predicted

Hierarchical F1: 82.1%
  - Harmonic mean of precision and recall

Per-Level Accuracy:
  - L1: How often top-level is correct
  - L2: How often mid-level is correct (given L1)
  - L3: How often leaf is correct (given L2)

Exact Match: 75.8%
  - How often the entire path is correct
```

---

## Customizing Examples

### Use Your Own Taxonomy

```python
from taxonomy_framework.models import CategoryNode, Taxonomy

# Build your taxonomy
root = CategoryNode(name="Root", description="My Categories")
category1 = CategoryNode(name="Category1", description="...", parent=root)
root.children.append(category1)
# ... add more

taxonomy = Taxonomy(root)

# Use in classifier
classifier = HybridClassifier(taxonomy=taxonomy, embedder=embedder, llm=llm)
```

### Use Your Own Dataset

```python
from datasets import Dataset

# Create dataset
data = {
    "text": ["product description 1", "product description 2"],
    "label": ["category1", "category2"]
}
dataset = Dataset.from_dict(data)

# Use for training
trainer.train(dataset=dataset, text_column="text", label_column="label")
```

### Use Different Embedding Model

```python
from taxonomy_framework.embeddings import SentenceTransformerBackend

# BGE model with query prefix (recommended for retrieval tasks)
backend = SentenceTransformerBackend(
    model_name="BAAI/bge-small-en-v1.5",
    query_prefix="Represent this sentence: ",
    device="cuda",
)

# Or a larger model without prefix
backend = SentenceTransformerBackend(
    model_name="sentence-transformers/all-mpnet-base-v2",
)
```

---

## Troubleshooting

### Out of Memory (OOM)

Use `03_memory_efficient_training.py` as a template, or:

```python
# Reduce batch sizes
TRAINING_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16

# Limit dataset
MAX_SAMPLES_PER_CLASS = 2
MAX_TOTAL_TRAIN_SAMPLES = 100

# Clear memory frequently
import gc
import torch

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Missing API Key Warning

```
Warning: No LLM provider configured. Running in embedding-only mode.
```

This is normal - the examples work without an LLM. To enable hybrid classification:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-your-key
```

### Slow Performance

- Use `BAAI/bge-small-en-v1.5` (good quality/speed balance) or `all-MiniLM-L6-v2` (fastest)
- Enable GPU: pass `device="cuda"` to `SentenceTransformerBackend`
- Call `embedder.build_index()` once to pre-compute candidate embeddings
- Reduce `top_k` in retrieval
- Use batch classification

---

## Next Steps

1. **Start with Example 1** to understand basic concepts
2. **Run Example 2** to see fine-tuning benefits
3. **Adapt examples** for your own taxonomy and data
4. **Deploy** using the FastAPI service (see `taxonomy_framework/api/README.md`)
