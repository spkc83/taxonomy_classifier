"""
Memory-Efficient Training - For Resource-Constrained Environments

This example demonstrates optimized training for systems with limited
memory (8GB RAM, 4GB GPU). Perfect for development on laptops or CI/CD.

Key Optimizations:
1. Better base model (paraphrase-MiniLM-L6-v2 for quality/speed balance)
2. Adequate samples per class (16) for SetFit contrastive learning
3. Multiple training epochs for better convergence
4. Batch processing with explicit memory cleanup
5. Garbage collection between heavy operations

System Requirements:
- Minimum 8GB RAM (works comfortably with 16GB)
- Optional GPU (4GB+ VRAM) - falls back to CPU gracefully
- Works on GTX 970 (4GB) and similar

Training Configuration:
- 16 samples per class (SetFit sweet spot for few-shot)
- 2 training epochs (contrastive + classifier head)
- ~800 training samples, ~200 test samples
- Batch size: 16

Requirements:
    pip install setfit datasets sentence-transformers torch

Environment Variables (optional):
    LLM_PROVIDER, LLM_MODEL, OPENAI_API_KEY, etc.

Usage:
    python examples/03_memory_efficient_training.py           # Load saved model if exists
    python examples/03_memory_efficient_training.py --retrain # Force retraining
"""

import argparse
import sys
from pathlib import Path
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


def get_llm_client():
    """Get LLM client from environment configuration.
    
    Returns None if no provider is configured or if creation fails.
    Examples will run in embedding-only mode without an LLM.
    """
    settings = Settings()
    
    api_key_map = {
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "google": settings.google_api_key,
        "cohere": settings.cohere_api_key,
        "ollama": None,  # Ollama doesn't require API key
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


EFFICIENT_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
SAMPLES_PER_CLASS = 16
NUM_CLASSES = 50
NUM_EPOCHS = 2
TRAINING_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


def get_device_info() -> Dict[str, Any]:
    """Get available compute resources."""
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available() if torch else False,
        "device": "cpu"
    }
    
    if info["cuda_available"] and torch:
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info


def clear_memory():
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class EvalResult:
    """Container for evaluation results."""
    name: str
    metrics: HierarchicalMetrics
    eval_time: float
    num_samples: int


def load_dbpedia_data() -> Tuple[Dataset, Dataset, Dict]:
    print("Loading DBpedia Classes dataset...")
    dataset = load_dataset("DeveloperOats/DBPedia_Classes", split="train")
    
    print(f"Original dataset size: {len(dataset)} samples")
    
    l3_labels = list(set(dataset["l3"]))
    print(f"Total L3 classes: {len(l3_labels)}")
    
    selected_labels = sorted(l3_labels)[:NUM_CLASSES]
    
    print(f"Selecting {len(selected_labels)} classes with {SAMPLES_PER_CLASS} samples each")
    
    filtered_indices = [
        i for i, label in enumerate(dataset["l3"]) 
        if label in selected_labels
    ]
    filtered_dataset = dataset.select(filtered_indices)
    
    sampled = sample_dataset(
        filtered_dataset, 
        label_column="l3", 
        num_samples=SAMPLES_PER_CLASS
    )
    
    label_to_indices: Dict[str, List[int]] = {}
    for idx in range(len(sampled)):
        label = sampled[idx]["l3"]
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    train_indices, test_indices = [], []
    for label, indices in label_to_indices.items():
        n_train = max(1, int(len(indices) * 0.75))
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])
    
    train_dataset = sampled.select(train_indices)
    test_dataset = sampled.select(test_indices)
    
    stats = {
        "original_size": len(dataset),
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "num_classes": len(set(train_dataset["l3"]))
    }
    
    print(f"Final train size: {stats['train_size']} samples")
    print(f"Final test size: {stats['test_size']} samples")
    print(f"Number of classes: {stats['num_classes']}")
    
    del dataset, filtered_dataset
    clear_memory()
    
    return train_dataset, test_dataset, stats


def build_taxonomy_from_data(train_dataset, test_dataset) -> Taxonomy:
    """Build taxonomy from L1/L2/L3 labels in datasets."""
    root = CategoryNode(name="Root", description="DBpedia taxonomy root")
    
    l1_nodes: Dict[str, CategoryNode] = {}
    l2_nodes: Dict[Tuple[str, str], CategoryNode] = {}
    l3_nodes: Dict[Tuple[str, str, str], CategoryNode] = {}
    
    all_data = list(train_dataset) + list(test_dataset)
    
    for item in all_data:
        l1, l2, l3 = item["l1"], item["l2"], item["l3"]
        
        if l1 not in l1_nodes:
            l1_node = CategoryNode(name=l1, description=l1, parent=root)
            root.children.append(l1_node)
            l1_nodes[l1] = l1_node
        else:
            l1_node = l1_nodes[l1]
        
        l2_key = (l1, l2)
        if l2_key not in l2_nodes:
            l2_node = CategoryNode(name=l2, description=l2, parent=l1_node)
            l1_node.children.append(l2_node)
            l2_nodes[l2_key] = l2_node
        else:
            l2_node = l2_nodes[l2_key]
        
        l3_key = (l1, l2, l3)
        if l3_key not in l3_nodes:
            l3_node = CategoryNode(name=l3, description=l3, parent=l2_node)
            l2_node.children.append(l3_node)
            l3_nodes[l3_key] = l3_node
    
    return Taxonomy(root)


class EfficientSetFitTrainer:
    """
    Memory-efficient SetFit trainer with resource management.
    
    Key optimizations:
    - Uses smaller base model
    - Reduced batch sizes
    - Explicit memory cleanup
    - Progress monitoring
    """
    
    def __init__(self, base_model: str = EFFICIENT_MODEL, output_dir: str = "./models/efficient-setfit"):
        self.base_model = base_model
        self.output_dir = output_dir
        self.model: Optional[SetFitModel] = None
        
    def train(
        self, 
        train_dataset: Dataset,
        text_column: str = "text",
        label_column: str = "l3",
        num_epochs: int = 1,
        batch_size: int = TRAINING_BATCH_SIZE
    ) -> float:
        """Train with memory-efficient settings. Returns training time."""
        print(f"Loading base model: {self.base_model}")
        clear_memory()
        
        self.model = SetFitModel.from_pretrained(self.base_model)
        
        args = TrainingArguments(
            batch_size=batch_size,
            num_epochs=num_epochs,
            evaluation_strategy="no",
            logging_steps=50,
        )
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
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
        """Save trained model."""
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
        
        print(f"Loading saved model from {self.output_dir}...")
        self.model = SetFitModel.from_pretrained(self.output_dir)
        return True

    def model_exists(self) -> bool:
        model_path = Path(self.output_dir)
        return model_path.exists() and (model_path / "config.json").exists()
    
    def predict(self, texts: List[str]) -> List[str]:
        """Predict labels for texts."""
        if self.model is None:
            raise ValueError("No model loaded.")
        raw_predictions = self.model.predict(texts)  # type: ignore[union-attr]
        if isinstance(raw_predictions, (list, tuple)):
            return [str(p) for p in raw_predictions]
        try:
            return [str(p) for p in iter(raw_predictions)]  # type: ignore[call-overload]
        except TypeError:
            return [str(raw_predictions)]


def create_classifier(taxonomy: Taxonomy) -> HybridClassifier:
    backend = SentenceTransformerBackend(model_name=EFFICIENT_MODEL)
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


def evaluate_classifier(
    classifier: HybridClassifier,
    test_dataset: Dataset,
    name: str,
    batch_size: int = EVAL_BATCH_SIZE
) -> EvalResult:
    """Evaluate classifier with batched processing."""
    print(f"\nEvaluating {name} on {len(test_dataset)} samples...")
    
    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []
    
    start_time = time.time()
    
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        true_path = [sample["l1"], sample["l2"], sample["l3"]]
        
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
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(test_dataset)}")
            clear_memory()
    
    eval_time = time.time() - start_time
    
    metrics = compute_hierarchical_metrics(
        y_true=y_true,
        y_pred=y_pred,
        level_names=["L1", "L2", "L3"]
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
    for level in ["L1", "L2", "L3"]:
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
    
    for level in ["L1", "L2", "L3"]:
        b_acc = b.per_level_accuracy.get(level, 0)
        f_acc = f.per_level_accuracy.get(level, 0)
        print(f"{f'{level} Accuracy':<30} {pct(b_acc):>12} {pct(f_acc):>12} {delta(b_acc, f_acc):>12}")
    
    print("-" * 66)
    print(f"{'Exact Match Accuracy':<30} {pct(b.exact_match_accuracy):>12} {pct(f.exact_match_accuracy):>12} {delta(b.exact_match_accuracy, f.exact_match_accuracy):>12}")
    
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
    print("=" * 70)
    
    device_info = get_device_info()
    print(f"\nDevice: {device_info['device']}")
    if device_info['cuda_available']:
        print(f"GPU: {device_info['gpu_name']} ({device_info['gpu_memory_gb']:.1f}GB)")
    
    print("\nTraining configuration:")
    print(f"  Samples per class: {SAMPLES_PER_CLASS}")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Training epochs:   {NUM_EPOCHS}")
    print(f"  Base model:        {EFFICIENT_MODEL}")
    
    model_output_dir = "./models/efficient-setfit-v2"
    trainer = EfficientSetFitTrainer(
        base_model=EFFICIENT_MODEL,
        output_dir=model_output_dir
    )
    
    should_train = args.retrain or not trainer.model_exists()
    
    if trainer.model_exists() and not args.retrain:
        print(f"\n[INFO] Found saved model at {model_output_dir}")
        print("       Use --retrain flag to force retraining")
    
    print("\n[1/6] Loading dataset...")
    train_dataset, test_dataset, stats = load_dbpedia_data()
    
    print("\n[2/6] Building taxonomy...")
    taxonomy = build_taxonomy_from_data(train_dataset, test_dataset)
    l1_count = len(taxonomy.root.children)
    l2_count = sum(len(l1.children) for l1 in taxonomy.root.children)
    l3_count = sum(len(l2.children) for l1 in taxonomy.root.children for l2 in l1.children)
    print(f"Taxonomy: {l1_count} L1 → {l2_count} L2 → {l3_count} L3")
    
    print("\n[3/6] Creating baseline classifier (pre-trained, no fine-tuning)...")
    baseline_classifier = create_classifier(taxonomy)
    
    if should_train:
        print("\n[4/6] Training SetFit model...")
        train_time = trainer.train(
            train_dataset=train_dataset,
            text_column="text",
            label_column="l3",
            num_epochs=NUM_EPOCHS,
            batch_size=TRAINING_BATCH_SIZE
        )
        trainer.save()
        print(f"Training completed in {train_time/60:.1f} minutes")
    else:
        print("\n[4/6] Loading saved SetFit model (skipping training)...")
        trainer.load()
        print(f"Model loaded from {model_output_dir}")
    
    clear_memory()
    
    print("\n[5/6] Creating fine-tuned classifier...")
    finetuned_classifier = create_finetuned_classifier(taxonomy, model_output_dir)
    
    print("\n[6/6] Evaluating classifiers...")
    
    baseline_result = evaluate_classifier(
        baseline_classifier,
        test_dataset,
        "Baseline (Pre-trained)"
    )
    
    clear_memory()
    
    finetuned_result = evaluate_classifier(
        finetuned_classifier,
        test_dataset,
        "Fine-tuned SetFit"
    )
    
    print_comparison(baseline_result, finetuned_result)
    
    clear_memory()
    
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS (Fine-tuned model)")
    print("=" * 70)
    
    sample_texts = [test_dataset[i]["text"] for i in range(min(5, len(test_dataset)))]
    sample_labels = [test_dataset[i]["l3"] for i in range(min(5, len(test_dataset)))]
    
    setfit_predictions = trainer.predict(sample_texts)
    
    for i, (text, true_label, pred_label) in enumerate(zip(sample_texts, sample_labels, setfit_predictions)):
        match = "✓" if true_label == pred_label else "✗"
        print(f"\n[{i+1}] {text[:60]}...")
        print(f"    True:  {true_label}")
        print(f"    Pred:  {pred_label} {match}")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nKey settings for good performance:")
    print(f"- {SAMPLES_PER_CLASS} samples per class (SetFit sweet spot)")
    print(f"- {NUM_EPOCHS} training epochs")
    print(f"- {EFFICIENT_MODEL} (good quality/speed balance)")
    print("- Explicit garbage collection between heavy operations")


if __name__ == "__main__":
    main()
