"""
Memory-Efficient Training - For Resource-Constrained Environments

This example demonstrates optimized training for systems with limited
memory (8GB RAM, 4GB GPU). Perfect for development on laptops or CI/CD.

Uses the TREC Question Classification dataset (Li & Roth, 2002) which
has a natural 2-level hierarchy: 6 coarse classes -> 50 fine classes.
Baseline (pre-trained) accuracy on coarse labels typically exceeds 80%.

Key Optimizations:
1. BGE-small-en-v1.5 — high quality embeddings in a small footprint
2. Pre-computed embedding index — candidates embedded once, reused per query
3. 16 samples per class for SetFit contrastive learning
4. Explicit garbage collection between heavy operations

System Requirements:
- Minimum 8GB RAM (works comfortably with 16GB)
- Optional GPU (4GB+ VRAM) - falls back to CPU gracefully

Requirements:
    pip install setfit datasets sentence-transformers torch

Usage:
    python examples/03_memory_efficient_training.py           # Load saved model if exists
    python examples/03_memory_efficient_training.py --retrain # Force retraining
"""

import argparse
import sys
import warnings
from pathlib import Path

# transformers 4.54 deprecated encoder_attention_mask but its own BertEncoder still passes it internally.
# Fixed upstream in 4.55. Safe to suppress until then.
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*is deprecated.*", category=FutureWarning, module=r"transformers\.utils\.deprecation")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import time

from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass

try:
    from datasets import load_dataset, Dataset
except ImportError:
    raise ImportError("datasets not installed. Run: pip install datasets")

try:
    from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
except ImportError:
    raise ImportError("setfit not installed. Run: pip install setfit")

try:
    import torch
except ImportError:
    torch = None

from taxonomy_framework.models import CategoryNode, Taxonomy, ClassificationResult
from taxonomy_framework.embeddings import SentenceTransformerBackend, SetFitBackend, EnsembleEmbedder
from taxonomy_framework.pipeline import HybridClassifier
from taxonomy_framework.evaluation import compute_hierarchical_metrics, HierarchicalMetrics
from taxonomy_framework.providers import ProviderFactory
from taxonomy_framework.config import Settings


# ---------------------------------------------------------------------------
# TREC coarse label descriptions for taxonomy node quality
# ---------------------------------------------------------------------------

COARSE_DESCRIPTIONS = {
    "ABBR": "Questions about abbreviations and their expansions",
    "ENTY": "Questions about entities such as animals, colors, events, food, products, and more",
    "DESC": "Questions seeking a definition, description, manner, or reason",
    "HUM": "Questions about people, groups, titles, or descriptions of individuals",
    "LOC": "Questions about places including cities, countries, mountains, and states",
    "NUM": "Questions expecting a numeric answer like dates, counts, distances, or percentages",
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EFFICIENT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence: "
SAMPLES_PER_CLASS = 16
NUM_EPOCHS = 2
TRAINING_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 32


def get_llm_client():
    settings = Settings()
    
    api_key_map = {
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "google": settings.google_api_key,
        "cohere": settings.cohere_api_key,
        "ollama": None,
    }
    
    provider = settings.llm_provider.lower() if settings.llm_provider else None
    api_key = api_key_map.get(provider)
    
    if not provider:
        print("Warning: No LLM provider configured. Running in embedding-only mode.")
        print("Set LLM_PROVIDER and appropriate API key environment variables.")
        return None
    
    if provider not in ["ollama", "local"] and not api_key:
        print(f"Warning: No API key found for provider '{provider}'. Running in embedding-only mode.")
        print(f"Set {provider.upper()}_API_KEY environment variable.")
        return None
    
    try:
        config = {"model": settings.llm_model}
        if api_key:
            config["api_key"] = api_key
        if settings.api_base_url:
            config["api_base"] = settings.api_base_url
            
        return ProviderFactory.create(provider, **config)
    except Exception as e:
        print(f"Warning: Could not create LLM client: {e}")
        print("Running in embedding-only mode.")
        return None


def _detect_device() -> str:
    if torch and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available() if torch else False,
        "device": _detect_device()
    }
    
    if info["cuda_available"] and torch:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info


def clear_memory():
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class EvalResult:
    name: str
    metrics: HierarchicalMetrics
    eval_time: float
    num_samples: int


# ---------------------------------------------------------------------------
# Dataset loading — TREC Question Classification (via SetFit/TREC-QC)
# ---------------------------------------------------------------------------

def load_trec_data() -> Tuple[Dataset, Dataset, Dict]:
    print("Loading TREC Question Classification dataset (SetFit/TREC-QC)...")
    ds = load_dataset("SetFit/TREC-QC")
    train_ds = ds["train"]
    test_ds = ds["test"]

    coarse_classes = sorted(set(train_ds["label_coarse_original"]))
    fine_classes = sorted(set(train_ds["label_original"]))

    stats = {
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "num_coarse": len(coarse_classes),
        "num_fine": len(fine_classes),
    }

    print(f"Train: {stats['train_size']} samples")
    print(f"Test:  {stats['test_size']} samples")
    print(f"Coarse classes: {stats['num_coarse']}, Fine classes: {stats['num_fine']}")

    return train_ds, test_ds, stats


# ---------------------------------------------------------------------------
# Taxonomy construction from TREC labels
# ---------------------------------------------------------------------------

def build_taxonomy_from_trec(train_ds: Dataset, test_ds: Dataset) -> Taxonomy:
    root = CategoryNode(name="Root", description="TREC question types")

    coarse_nodes: Dict[str, CategoryNode] = {}
    fine_nodes: Dict[str, CategoryNode] = {}

    all_data = list(train_ds) + list(test_ds)

    for item in all_data:
        coarse_code = item["label_coarse_original"]
        fine_code = item["label_original"]
        fine_text = item["label_text"]

        if coarse_code not in coarse_nodes:
            desc = COARSE_DESCRIPTIONS.get(coarse_code, coarse_code)
            node = CategoryNode(name=coarse_code, description=desc, parent=root)
            root.children.append(node)
            coarse_nodes[coarse_code] = node

        if fine_code not in fine_nodes:
            parent = coarse_nodes[coarse_code]
            node = CategoryNode(name=fine_code, description=fine_text, parent=parent)
            parent.children.append(node)
            fine_nodes[fine_code] = node

    return Taxonomy(root)


# ---------------------------------------------------------------------------
# SetFit trainer (memory-efficient)
# ---------------------------------------------------------------------------

class EfficientSetFitTrainer:
    def __init__(self, base_model: str = EFFICIENT_MODEL, output_dir: str = "./models/efficient-setfit", device: Optional[str] = None):
        self.base_model = base_model
        self.output_dir = output_dir
        self.device = device or _detect_device()
        self.model: Optional[SetFitModel] = None
        
    def train(
        self, 
        train_dataset: Dataset,
        text_column: str = "text",
        label_column: str = "label_coarse_original",
        num_epochs: int = NUM_EPOCHS,
        batch_size: int = TRAINING_BATCH_SIZE
    ) -> float:
        print(f"Loading base model: {self.base_model} (device={self.device})")
        clear_memory()
        
        self.model = SetFitModel.from_pretrained(self.base_model, device=self.device)
        
        args = TrainingArguments(
            batch_size=batch_size,
            num_epochs=num_epochs,
            eval_strategy="no",
            logging_steps=50,
        )
        
        cols_to_drop = [
            c for c in train_dataset.column_names
            if c not in (text_column, label_column)
        ]
        clean_dataset = train_dataset.remove_columns(cols_to_drop)
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=clean_dataset,
            column_mapping={text_column: "text", label_column: "label"}
        )
        
        print(f"Starting training with {len(train_dataset)} samples...")
        start_time = time.time()
        trainer.train()
        train_time = time.time() - start_time
        
        print(f"Training completed in {train_time:.1f}s")
        
        del trainer
        clear_memory()
        
        return train_time
    
    def save(self):
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        print(f"Model saved to {self.output_dir}")

    def load(self) -> bool:
        model_path = Path(self.output_dir)
        if not model_path.exists():
            return False
        
        config_file = model_path / "config.json"
        if not config_file.exists():
            return False
        
        print(f"Loading saved model from {self.output_dir} (device={self.device})...")
        self.model = SetFitModel.from_pretrained(self.output_dir, device=self.device)
        return True

    def model_exists(self) -> bool:
        model_path = Path(self.output_dir)
        return model_path.exists() and (model_path / "config.json").exists()
    
    def predict(self, texts: List[str]) -> List[str]:
        if self.model is None:
            raise ValueError("No model loaded.")
        raw_predictions = self.model.predict(texts)
        if isinstance(raw_predictions, (list, tuple)):
            return [str(p) for p in raw_predictions]
        try:
            return [str(p) for p in iter(raw_predictions)]
        except TypeError:
            return [str(raw_predictions)]


# ---------------------------------------------------------------------------
# Classifier factories
# ---------------------------------------------------------------------------

def create_classifier(taxonomy: Taxonomy, device: Optional[str] = None) -> HybridClassifier:
    backend = SentenceTransformerBackend(
        model_name=EFFICIENT_MODEL,
        query_prefix=BGE_QUERY_PREFIX,
        device=device,
    )
    embedder = EnsembleEmbedder([backend])
    llm_client = get_llm_client()
    
    return HybridClassifier(
        taxonomy=taxonomy,
        embedder=embedder,
        llm=llm_client
    )


def create_finetuned_classifier(taxonomy: Taxonomy, model_path: str) -> HybridClassifier:
    backend = SetFitBackend.from_finetuned(model_path)
    embedder = EnsembleEmbedder([backend])
    llm_client = get_llm_client()
    
    return HybridClassifier(
        taxonomy=taxonomy,
        embedder=embedder,
        llm=llm_client
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    classifier: HybridClassifier,
    test_dataset: Dataset,
    name: str,
    batch_size: int = EVAL_BATCH_SIZE
) -> EvalResult:
    print(f"\nEvaluating {name} on {len(test_dataset)} samples...")
    
    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []
    
    start_time = time.time()
    
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        true_path = [sample["label_coarse_original"], sample["label_original"]]
        
        result = classifier.classify(sample["text"])
        
        if isinstance(result, ClassificationResult):
            pred_node = result.predicted_category
            pred_path = []
            node = pred_node
            while node is not None and node.name != "Root":
                pred_path.insert(0, node.name)
                node = node.parent
        else:
            pred_path = []
        
        y_true.append(true_path)
        y_pred.append(pred_path)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_dataset)}")
            clear_memory()
    
    eval_time = time.time() - start_time
    
    metrics = compute_hierarchical_metrics(
        y_true=y_true,
        y_pred=y_pred,
        level_names=["Coarse", "Fine"]
    )
    
    return EvalResult(
        name=name,
        metrics=metrics,
        eval_time=eval_time,
        num_samples=len(test_dataset)
    )


def print_results(result: EvalResult):
    m = result.metrics
    
    print(f"\n{'='*60}")
    print(f"Results: {result.name}")
    print(f"{'='*60}")
    print(f"Samples: {result.num_samples} | Time: {result.eval_time:.1f}s")
    print("\nHierarchical Metrics:")
    print(f"  Precision: {m.hierarchical_precision:.4f}")
    print(f"  Recall:    {m.hierarchical_recall:.4f}")
    print(f"  F1:        {m.hierarchical_f1:.4f}")
    print("\nPer-Level Accuracy:")
    for level in ["Coarse", "Fine"]:
        acc = m.per_level_accuracy.get(level, 0)
        print(f"  {level}: {acc:.4f}")
    print(f"\nExact Match: {m.exact_match_accuracy:.4f}")


def print_comparison(baseline: EvalResult, finetuned: EvalResult):
    def delta(base: float, fine: float) -> str:
        diff = fine - base
        return f"+{diff*100:.1f}%" if diff > 0 else f"{diff*100:.1f}%"
    
    def pct(val: float) -> str:
        return f"{val*100:.1f}%"
    
    b, f = baseline.metrics, finetuned.metrics
    
    print("\n" + "=" * 75)
    print("COMPARISON: Baseline (Pre-trained) vs Fine-tuned SetFit")
    print("=" * 75)
    
    print(f"\n{'Metric':<30} {'Baseline':>12} {'Fine-tuned':>12} {'Δ Change':>12}")
    print("-" * 66)
    
    print(f"{'Hierarchical Precision':<30} {pct(b.hierarchical_precision):>12} {pct(f.hierarchical_precision):>12} {delta(b.hierarchical_precision, f.hierarchical_precision):>12}")
    print(f"{'Hierarchical Recall':<30} {pct(b.hierarchical_recall):>12} {pct(f.hierarchical_recall):>12} {delta(b.hierarchical_recall, f.hierarchical_recall):>12}")
    print(f"{'Hierarchical F1':<30} {pct(b.hierarchical_f1):>12} {pct(f.hierarchical_f1):>12} {delta(b.hierarchical_f1, f.hierarchical_f1):>12}")
    
    print("-" * 66)
    
    for level in ["Coarse", "Fine"]:
        b_acc = b.per_level_accuracy.get(level, 0)
        f_acc = f.per_level_accuracy.get(level, 0)
        print(f"{f'{level} Accuracy':<30} {pct(b_acc):>12} {pct(f_acc):>12} {delta(b_acc, f_acc):>12}")
    
    print("-" * 66)
    print(f"{'Exact Match Accuracy':<30} {pct(b.exact_match_accuracy):>12} {pct(f.exact_match_accuracy):>12} {delta(b.exact_match_accuracy, f.exact_match_accuracy):>12}")
    
    print(f"\n{'Eval Time':<30} {baseline.eval_time:>11.1f}s {finetuned.eval_time:>11.1f}s")
    
    print("\n" + "=" * 75)
    
    if f.hierarchical_f1 > b.hierarchical_f1:
        improvement = (f.hierarchical_f1 - b.hierarchical_f1) / b.hierarchical_f1 * 100 if b.hierarchical_f1 > 0 else 0
        print(f"\n✓ Fine-tuning improved Hierarchical F1 by {improvement:.1f}% relative gain")


def parse_args():
    parser = argparse.ArgumentParser(description="Memory-efficient SetFit training example")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if a saved model exists"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("Memory-Efficient Classification Example")
    print("Dataset: TREC Question Classification (6 coarse → 50 fine)")
    print(f"Model:   {EFFICIENT_MODEL}")
    print("=" * 70)
    
    device_info = get_device_info()
    device = device_info["device"]
    print(f"\nDevice: {device}")
    if device_info['cuda_available']:
        print(f"GPU: {device_info['gpu_name']} ({device_info['gpu_memory_gb']:.1f}GB)")
    
    model_output_dir = "./models/trec-setfit"
    trainer = EfficientSetFitTrainer(
        base_model=EFFICIENT_MODEL,
        output_dir=model_output_dir,
        device=device,
    )
    
    should_train = args.retrain or not trainer.model_exists()
    
    if trainer.model_exists() and not args.retrain:
        print(f"\n[INFO] Found saved model at {model_output_dir}")
        print("       Use --retrain flag to force retraining")
    
    print("\n[1/6] Loading dataset...")
    train_dataset, test_dataset, stats = load_trec_data()
    
    print("\n[2/6] Building taxonomy...")
    taxonomy = build_taxonomy_from_trec(train_dataset, test_dataset)
    l1_count = len(taxonomy.root.children)
    l2_count = sum(len(l1.children) for l1 in taxonomy.root.children)
    print(f"Taxonomy: {l1_count} coarse → {l2_count} fine")
    
    if should_train:
        print("\n[3/6] Training SetFit model on coarse labels...")
        train_time = trainer.train(
            train_dataset=train_dataset,
            text_column="text",
            label_column="label_coarse_original",
            num_epochs=NUM_EPOCHS,
            batch_size=TRAINING_BATCH_SIZE
        )
        trainer.save()
        trainer.model = None
        clear_memory()
        print(f"Training completed in {train_time/60:.1f} minutes")
    else:
        print("\n[3/6] Using saved SetFit model (skipping training)...")
        print(f"Model loaded from {model_output_dir}")
    
    print("\n[4/6] Evaluating baseline classifier (pre-trained, no fine-tuning)...")
    baseline_classifier = create_classifier(taxonomy, device=device)
    baseline_result = evaluate_classifier(
        baseline_classifier,
        test_dataset,
        "Baseline (Pre-trained BGE-small)"
    )
    del baseline_classifier
    clear_memory()
    
    print("\n[5/6] Evaluating fine-tuned classifier...")
    trainer.load()
    finetuned_classifier = create_finetuned_classifier(taxonomy, model_output_dir)
    finetuned_result = evaluate_classifier(
        finetuned_classifier,
        test_dataset,
        "Fine-tuned SetFit"
    )
    del finetuned_classifier
    clear_memory()
    
    print_comparison(baseline_result, finetuned_result)
    
    print("\n[6/6] Sample predictions (fine-tuned model)...")
    print("=" * 70)
    
    sample_texts = [test_dataset[i]["text"] for i in range(min(5, len(test_dataset)))]
    sample_labels = [test_dataset[i]["label_coarse_original"] for i in range(min(5, len(test_dataset)))]
    
    setfit_predictions = trainer.predict(sample_texts)
    
    for i, (text, true_label, pred_label) in enumerate(zip(sample_texts, sample_labels, setfit_predictions)):
        match = "✓" if true_label == pred_label else "✗"
        print(f"\n[{i+1}] {text[:80]}...")
        print(f"    True:  {true_label}")
        print(f"    Pred:  {pred_label} {match}")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print(f"\nKey settings:")
    print(f"- Model: {EFFICIENT_MODEL} (33M params, high quality/size ratio)")
    print(f"- Dataset: TREC (6 coarse, 50 fine — well-separated classes)")
    print(f"- Embedding index pre-computed once (fast repeated queries)")


if __name__ == "__main__":
    main()
