"""
SetFit Fine-tuning Comparison - Baseline vs Fine-tuned Performance

This example trains a SetFit model on the DBpedia Classes dataset and
compares performance against a pre-trained baseline to demonstrate
the value of domain-specific fine-tuning.

What you'll learn:
1. Loading the DBpedia Classes hierarchical dataset (9 L1 -> 70 L2 -> 219 L3)
2. Stratified train/test split to prevent data leakage
3. Building a taxonomy from the hierarchical labels
4. Comparing classification performance:
   - BASELINE: Pre-trained SentenceTransformer (no fine-tuning)
   - FINE-TUNED: SetFit model trained on taxonomy labels
5. Evaluating both on held-out test set with hierarchical metrics

Expected Results:
- Fine-tuning typically improves Hierarchical F1 by 15-25%
- Leaf-level (L3) accuracy sees the biggest improvement
- Shows when and why fine-tuning matters

Expected runtime: ~30-40 minutes (includes training)

Requirements:
    pip install setfit datasets sentence-transformers

Environment Variables (optional):
    LLM_PROVIDER: Provider name ('openai', 'anthropic', 'google', 'cohere', 'ollama')
    LLM_MODEL: Model name (e.g., 'gpt-4o-mini', 'claude-3-haiku-20240307')
    OPENAI_API_KEY, ANTHROPIC_API_KEY, etc: API keys for providers

Usage:
    python examples/02_setfit_finetuning_comparison.py           # Load saved model if exists
    python examples/02_setfit_finetuning_comparison.py --retrain # Force retraining
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from datasets import load_dataset, Dataset
except ImportError:
    raise ImportError(
        "datasets not installed. Run: pip install datasets"
    )

try:
    from setfit import sample_dataset
except ImportError:
    raise ImportError(
        "setfit not installed. Run: pip install setfit"
    )

import time
from collections import Counter
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

from taxonomy_framework.models import CategoryNode, Taxonomy, ClassificationResult, AbstainResult
from taxonomy_framework.training import SetFitTrainer
from taxonomy_framework.embeddings import (
    SetFitBackend, 
    SentenceTransformerBackend, 
    EnsembleEmbedder,
    EmbeddingModel
)
from taxonomy_framework.pipeline import HybridClassifier
from taxonomy_framework.evaluation import (
    compute_hierarchical_metrics, 
    HierarchicalMetrics,
    mean_lca_distance
)
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


@dataclass
class EvaluationResult:
    """Container for evaluation results with timing info."""
    name: str
    metrics: HierarchicalMetrics
    eval_time: float
    num_classified: int
    num_abstained: int


def load_dbpedia_data(num_samples_per_class: int = 16) -> Tuple[Dataset, Dataset]:
    """Load and stratify DBpedia Classes dataset into train/test splits."""
    print("Loading DBpedia Classes dataset from HuggingFace...")
    dataset = load_dataset("DeveloperOats/DBPedia_Classes", split="train")
    
    print(f"Sampling {num_samples_per_class} examples per L3 class...")
    sampled = sample_dataset(dataset, label_column="l3", num_samples=num_samples_per_class)
    
    print("Performing stratified train/test split (75/25)...")
    
    label_to_indices: Dict[str, List[int]] = {}
    for idx in range(len(sampled)):
        label = sampled[idx]["l3"]
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    train_indices = []
    test_indices = []
    
    for label, indices in label_to_indices.items():
        n_train = max(1, int(len(indices) * 0.75))
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])
    
    train_dataset = sampled.select(train_indices)
    test_dataset = sampled.select(test_indices)
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set:  {len(test_dataset)} samples (HELD OUT)")
    
    return train_dataset, test_dataset


def print_dataset_stats(train_dataset, test_dataset):
    """Print statistics about the train/test datasets."""
    all_l1 = set(train_dataset["l1"]) | set(test_dataset["l1"])
    all_l2 = set(train_dataset["l2"]) | set(test_dataset["l2"])
    all_l3 = set(train_dataset["l3"]) | set(test_dataset["l3"])
    
    print("\n=== DBpedia Classes Dataset Statistics ===")
    print(f"L1 categories: {len(all_l1)}")
    print(f"L2 categories: {len(all_l2)}")
    print(f"L3 categories: {len(all_l3)}")
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    
    print("\n--- L1 Distribution ---")
    train_l1_counts = Counter(train_dataset["l1"])
    for l1 in sorted(all_l1):
        print(f"  {l1}: {train_l1_counts.get(l1, 0)} train")


def build_taxonomy_from_dbpedia(train_dataset, test_dataset) -> Taxonomy:
    """Build a taxonomy tree from DBpedia hierarchical labels."""
    root = CategoryNode(name="Root", description="DBpedia Classes taxonomy root")
    
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


def print_taxonomy_stats(taxonomy: Taxonomy):
    """Print taxonomy statistics."""
    l1_count = len(taxonomy.root.children)
    l2_count = sum(len(l1.children) for l1 in taxonomy.root.children)
    l3_count = sum(
        len(l2.children) 
        for l1 in taxonomy.root.children 
        for l2 in l1.children
    )
    
    print(f"\n=== Taxonomy Structure ===")
    print(f"Hierarchy: {l1_count} L1 → {l2_count} L2 → {l3_count} L3 categories")


def train_setfit_model(train_dataset, output_dir: str) -> Tuple[SetFitTrainer, float]:
    """Train a SetFit model on the L3 (leaf) labels."""
    print(f"\nTraining on {len(train_dataset)} samples...")
    
    trainer = SetFitTrainer(
        base_model="sentence-transformers/paraphrase-mpnet-base-v2",
        output_dir=output_dir
    )
    
    start_time = time.time()
    trainer.train(
        dataset=train_dataset,
        text_column="text",
        label_column="l3",
        auto_split=False
    )
    train_time = time.time() - start_time
    
    trainer.save()
    
    return trainer, train_time


def model_exists(model_path: str) -> bool:
    path = Path(model_path)
    return path.exists() and (path / "config.json").exists()


def create_baseline_pipeline(taxonomy: Taxonomy) -> HybridClassifier:
    """Create pipeline with pre-trained SentenceTransformer (NO fine-tuning)."""
    backend = SentenceTransformerBackend(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )
    embedder = EnsembleEmbedder([backend])
    llm_client = get_llm_client()
    
    return HybridClassifier(
        taxonomy=taxonomy,
        embedder=embedder,
        llm=llm_client
    )


def create_finetuned_pipeline(taxonomy: Taxonomy, model_path: str) -> HybridClassifier:
    """Create pipeline with fine-tuned SetFit model."""
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
    test_dataset,
    name: str
) -> EvaluationResult:
    """Evaluate a classifier on test set and return metrics."""
    print(f"\nEvaluating {name}...")
    
    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []
    num_classified = 0
    num_abstained = 0
    
    start_time = time.time()
    
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        text = sample["text"]
        true_path = [sample["l1"], sample["l2"], sample["l3"]]
        
        result = classifier.classify(text)
        
        if isinstance(result, ClassificationResult):
            pred_node = result.predicted_category
            pred_path = []
            
            node = pred_node
            while node is not None and node.name != "Root":
                pred_path.insert(0, node.name)
                node = node.parent
            
            y_true.append(true_path)
            y_pred.append(pred_path)
            num_classified += 1
            
        elif isinstance(result, AbstainResult):
            y_true.append(true_path)
            
            if result.partial_path:
                pred_path = [p.strip() for p in result.partial_path.split(">")]
            else:
                pred_path = []
            
            y_pred.append(pred_path)
            num_abstained += 1
        
        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(test_dataset)} samples processed...")
    
    eval_time = time.time() - start_time
    
    metrics = compute_hierarchical_metrics(
        y_true=y_true,
        y_pred=y_pred,
        level_names=["L1", "L2", "L3"]
    )
    
    return EvaluationResult(
        name=name,
        metrics=metrics,
        eval_time=eval_time,
        num_classified=num_classified,
        num_abstained=num_abstained
    )


def print_comparison_table(baseline: EvaluationResult, finetuned: EvaluationResult):
    """Print a comparison table of baseline vs fine-tuned metrics."""
    
    def delta_str(baseline_val: float, finetuned_val: float) -> str:
        diff = finetuned_val - baseline_val
        if diff > 0:
            return f"+{diff*100:.1f}%"
        elif diff < 0:
            return f"{diff*100:.1f}%"
        else:
            return "0.0%"
    
    def format_pct(val: float) -> str:
        return f"{val*100:.1f}%"
    
    print("\n" + "=" * 80)
    print("COMPARISON: Baseline (Pre-trained) vs Fine-tuned SetFit")
    print("=" * 80)
    
    b, f = baseline.metrics, finetuned.metrics
    
    print(f"\n{'Metric':<35} {'Baseline':>12} {'Fine-tuned':>12} {'Δ Change':>12}")
    print("-" * 71)
    
    print(f"{'Hierarchical Precision':<35} {format_pct(b.hierarchical_precision):>12} {format_pct(f.hierarchical_precision):>12} {delta_str(b.hierarchical_precision, f.hierarchical_precision):>12}")
    print(f"{'Hierarchical Recall':<35} {format_pct(b.hierarchical_recall):>12} {format_pct(f.hierarchical_recall):>12} {delta_str(b.hierarchical_recall, f.hierarchical_recall):>12}")
    print(f"{'Hierarchical F1':<35} {format_pct(b.hierarchical_f1):>12} {format_pct(f.hierarchical_f1):>12} {delta_str(b.hierarchical_f1, f.hierarchical_f1):>12}")
    
    print("-" * 71)
    
    for level in ["L1", "L2", "L3"]:
        b_acc = b.per_level_accuracy.get(level, 0)
        f_acc = f.per_level_accuracy.get(level, 0)
        print(f"{f'{level} Accuracy':<35} {format_pct(b_acc):>12} {format_pct(f_acc):>12} {delta_str(b_acc, f_acc):>12}")
    
    print("-" * 71)
    
    print(f"{'Exact Match Accuracy':<35} {format_pct(b.exact_match_accuracy):>12} {format_pct(f.exact_match_accuracy):>12} {delta_str(b.exact_match_accuracy, f.exact_match_accuracy):>12}")
    print(f"{'Depth-Weighted Accuracy':<35} {format_pct(b.depth_weighted_accuracy):>12} {format_pct(f.depth_weighted_accuracy):>12} {delta_str(b.depth_weighted_accuracy, f.depth_weighted_accuracy):>12}")
    
    print("-" * 71)
    
    b_class_rate = baseline.num_classified / b.num_samples
    f_class_rate = finetuned.num_classified / f.num_samples
    print(f"{'Classification Rate':<35} {format_pct(b_class_rate):>12} {format_pct(f_class_rate):>12} {delta_str(b_class_rate, f_class_rate):>12}")
    
    print(f"{'Evaluation Time':<35} {baseline.eval_time:>11.1f}s {finetuned.eval_time:>11.1f}s")
    
    print("\n" + "=" * 80)
    
    hf_improvement = (f.hierarchical_f1 - b.hierarchical_f1) / b.hierarchical_f1 * 100 if b.hierarchical_f1 > 0 else float('inf')
    exact_improvement = (f.exact_match_accuracy - b.exact_match_accuracy) / b.exact_match_accuracy * 100 if b.exact_match_accuracy > 0 else float('inf')
    
    print("\n=== KEY FINDINGS ===")
    print(f"• Hierarchical F1 improvement: {hf_improvement:+.1f}% relative gain")
    print(f"• Exact Match improvement: {exact_improvement:+.1f}% relative gain")
    
    if f.hierarchical_f1 > b.hierarchical_f1:
        print("\n✓ Fine-tuning IMPROVED classification performance")
        
        l3_b = b.per_level_accuracy.get("L3", 0)
        l3_f = f.per_level_accuracy.get("L3", 0)
        if l3_f > l3_b:
            print(f"  → Leaf-level (L3) accuracy improved from {format_pct(l3_b)} to {format_pct(l3_f)}")
            print("  → Fine-tuning helps distinguish fine-grained categories")
    else:
        print("\n⚠ Fine-tuning did NOT improve performance")
        print("  → Consider: more training data, different base model, or hyperparameter tuning")


def print_detailed_metrics(result: EvaluationResult):
    """Print detailed metrics for a single evaluation result."""
    m = result.metrics
    print(f"\n--- {result.name} Detailed Metrics ---")
    print(f"Samples: {m.num_samples} | Classified: {result.num_classified} | Abstained: {result.num_abstained}")
    print(f"Evaluation time: {result.eval_time:.1f}s")
    print(f"\nHierarchical: P={m.hierarchical_precision:.4f}, R={m.hierarchical_recall:.4f}, F1={m.hierarchical_f1:.4f}")
    
    print("\nPer-Level Breakdown:")
    for level in sorted(m.per_level_accuracy.keys()):
        acc = m.per_level_accuracy[level]
        p = m.per_level_precision.get(level, float('nan'))
        r = m.per_level_recall.get(level, float('nan'))
        f1 = m.per_level_f1.get(level, float('nan'))
        print(f"  {level}: Acc={acc:.3f}, P={p:.3f}, R={r:.3f}, F1={f1:.3f}")


def demo_side_by_side(
    baseline_clf: HybridClassifier,
    finetuned_clf: HybridClassifier,
    test_dataset,
    num_samples: int = 5
):
    """Show side-by-side classification comparison."""
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE CLASSIFICATION EXAMPLES")
    print("=" * 80)
    
    seen_l1 = set()
    samples = []
    for i in range(len(test_dataset)):
        l1 = test_dataset[i]["l1"]
        if l1 not in seen_l1 and len(samples) < num_samples:
            samples.append(test_dataset[i])
            seen_l1.add(l1)
        if len(samples) >= num_samples:
            break
    
    for i, sample in enumerate(samples):
        text = sample["text"]
        display_text = text[:70] + "..." if len(text) > 70 else text
        actual = f"{sample['l1']} > {sample['l2']} > {sample['l3']}"
        
        print(f"\n[Example {i+1}]")
        print(f"Input: \"{display_text}\"")
        print(f"Actual: {actual}")
        
        baseline_result = baseline_clf.classify(text)
        finetuned_result = finetuned_clf.classify(text)
        
        def format_result(result) -> str:
            if isinstance(result, ClassificationResult):
                return f"{result.predicted_category.path()} (conf: {result.confidence_score:.2f})"
            else:
                return f"ABSTAIN: {result.reason}"
        
        baseline_pred = format_result(baseline_result)
        finetuned_pred = format_result(finetuned_result)
        
        baseline_correct = isinstance(baseline_result, ClassificationResult) and \
            baseline_result.predicted_category.name == sample["l3"]
        finetuned_correct = isinstance(finetuned_result, ClassificationResult) and \
            finetuned_result.predicted_category.name == sample["l3"]
        
        b_mark = "✓" if baseline_correct else "✗"
        f_mark = "✓" if finetuned_correct else "✗"
        
        print(f"Baseline:   {b_mark} {baseline_pred}")
        print(f"Fine-tuned: {f_mark} {finetuned_pred}")


def parse_args():
    parser = argparse.ArgumentParser(description="SetFit fine-tuning comparison example")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if a saved model exists"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("=" * 80)
    print("DBpedia Hierarchical Classification")
    print("COMPARISON: Pre-trained Baseline vs Fine-tuned SetFit")
    print("=" * 80)
    
    model_path = "./models/dbpedia-setfit"
    should_train = args.retrain or not model_exists(model_path)
    
    if model_exists(model_path) and not args.retrain:
        print(f"\n[INFO] Found saved model at {model_path}")
        print("       Use --retrain flag to force retraining")
    
    print("\n[1/6] Loading and stratifying data...")
    train_dataset, test_dataset = load_dbpedia_data(num_samples_per_class=16)
    print_dataset_stats(train_dataset, test_dataset)
    
    print("\n[2/6] Building taxonomy...")
    taxonomy = build_taxonomy_from_dbpedia(train_dataset, test_dataset)
    print_taxonomy_stats(taxonomy)
    
    print("\n[3/6] Creating BASELINE pipeline (pre-trained, no fine-tuning)...")
    baseline_classifier = create_baseline_pipeline(taxonomy)
    print("  Using: sentence-transformers/paraphrase-mpnet-base-v2 (pre-trained)")
    
    if should_train:
        print("\n[4/6] Training SetFit model for FINE-TUNED pipeline...")
        _, train_time = train_setfit_model(train_dataset, model_path)
        print(f"  Training completed in {train_time/60:.1f} minutes")
    else:
        print("\n[4/6] Loading saved SetFit model (skipping training)...")
        print(f"  Model loaded from {model_path}")
    
    print("\n[5/6] Creating FINE-TUNED pipeline...")
    finetuned_classifier = create_finetuned_pipeline(taxonomy, model_path)
    print(f"  Using: {model_path} (fine-tuned on {len(train_dataset)} samples)")
    
    print("\n[6/6] Evaluating both models on held-out test set...")
    print(f"  Test set size: {len(test_dataset)} samples")
    
    baseline_results = evaluate_classifier(
        baseline_classifier, 
        test_dataset, 
        "Baseline (Pre-trained)"
    )
    
    finetuned_results = evaluate_classifier(
        finetuned_classifier, 
        test_dataset, 
        "Fine-tuned SetFit"
    )
    
    print_comparison_table(baseline_results, finetuned_results)
    
    print_detailed_metrics(baseline_results)
    print_detailed_metrics(finetuned_results)
    
    demo_side_by_side(
        baseline_classifier,
        finetuned_classifier,
        test_dataset,
        num_samples=5
    )
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
