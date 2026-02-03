"""
Hierarchical classification evaluation metrics.

Implements standard metrics for taxonomy/ontology classification:
- Hierarchical Precision (hP), Recall (hR), F-score (hF)
- Per-level accuracy
- Depth-weighted accuracy

Based on Kiritchenko et al. (2006) and Kosmopoulos et al. (2014).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import numpy as np


@dataclass
class HierarchicalMetrics:
    """Container for hierarchical classification metrics."""
    
    hierarchical_precision: float
    hierarchical_recall: float
    hierarchical_f1: float
    per_level_accuracy: Dict[str, float]
    per_level_precision: Dict[str, float]
    per_level_recall: Dict[str, float]
    per_level_f1: Dict[str, float]
    depth_weighted_accuracy: float
    exact_match_accuracy: float
    num_samples: int
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for serialization."""
        return {
            "hierarchical_precision": self.hierarchical_precision,
            "hierarchical_recall": self.hierarchical_recall,
            "hierarchical_f1": self.hierarchical_f1,
            "per_level_accuracy": self.per_level_accuracy,
            "per_level_precision": self.per_level_precision,
            "per_level_recall": self.per_level_recall,
            "per_level_f1": self.per_level_f1,
            "depth_weighted_accuracy": self.depth_weighted_accuracy,
            "exact_match_accuracy": self.exact_match_accuracy,
            "num_samples": self.num_samples,
        }
    
    def __str__(self) -> str:
        """Human-readable summary of metrics."""
        lines = [
            "=== Hierarchical Classification Metrics ===",
            f"Samples evaluated: {self.num_samples}",
            "",
            "--- Hierarchical Metrics (with ancestor credit) ---",
            f"  Hierarchical Precision: {self.hierarchical_precision:.4f}",
            f"  Hierarchical Recall:    {self.hierarchical_recall:.4f}",
            f"  Hierarchical F1:        {self.hierarchical_f1:.4f}",
            "",
            "--- Per-Level Metrics ---",
        ]
        
        for level in sorted(self.per_level_accuracy.keys()):
            acc = self.per_level_accuracy[level]
            prec = self.per_level_precision.get(level, float('nan'))
            rec = self.per_level_recall.get(level, float('nan'))
            f1 = self.per_level_f1.get(level, float('nan'))
            lines.append(f"  {level}: Acc={acc:.4f}, P={prec:.4f}, R={rec:.4f}, F1={f1:.4f}")
        
        lines.extend([
            "",
            "--- Aggregate Metrics ---",
            f"  Depth-Weighted Accuracy: {self.depth_weighted_accuracy:.4f}",
            f"  Exact Match Accuracy:    {self.exact_match_accuracy:.4f}",
        ])
        
        return "\n".join(lines)


def _get_ancestors_from_path(path: List[str]) -> Set[str]:
    """
    Get all nodes in a path as ancestor set.
    For path ["A", "B", "C"], returns {"A", "B", "C"}.
    """
    return set(path) if path else set()


def hierarchical_precision_recall_f1(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> Tuple[float, float, float]:
    """
    Compute hierarchical precision, recall, and F1-score.
    
    Uses the set-based formulation from Kiritchenko et al. (2006):
    - hP = Σ|αᵢ ∩ βᵢ| / Σ|αᵢ|
    - hR = Σ|αᵢ ∩ βᵢ| / Σ|βᵢ|
    - hF = 2 * hP * hR / (hP + hR)
    
    Where αᵢ is the predicted path and βᵢ is the true path (both include ancestors).
    
    Args:
        y_true: List of true paths, e.g., [["L1", "L2", "L3"], ...]
        y_pred: List of predicted paths, same format
    
    Returns:
        Tuple of (hierarchical_precision, hierarchical_recall, hierarchical_f1)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}")
    
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    
    sum_intersection = 0
    sum_predicted = 0
    sum_true = 0
    
    for true_path, pred_path in zip(y_true, y_pred):
        # Build ancestor sets
        alpha_i = _get_ancestors_from_path(pred_path)
        beta_i = _get_ancestors_from_path(true_path)
        
        # Compute set operations
        sum_intersection += len(alpha_i & beta_i)
        sum_predicted += len(alpha_i)
        sum_true += len(beta_i)
    
    # Compute metrics
    hP = sum_intersection / sum_predicted if sum_predicted > 0 else 0.0
    hR = sum_intersection / sum_true if sum_true > 0 else 0.0
    hF = 2 * hP * hR / (hP + hR) if (hP + hR) > 0 else 0.0
    
    return hP, hR, hF


def per_level_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    level_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Compute accuracy, precision, recall, and F1 at each level of the hierarchy.
    
    Args:
        y_true: List of true paths
        y_pred: List of predicted paths
        level_names: Optional names for levels (e.g., ["L1", "L2", "L3"]).
                     If None, uses "L1", "L2", etc.
    
    Returns:
        Tuple of (accuracy_dict, precision_dict, recall_dict, f1_dict)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}")
    
    # Determine number of levels
    max_levels = max(
        max((len(p) for p in y_true), default=0),
        max((len(p) for p in y_pred), default=0)
    )
    
    if level_names is None:
        level_names = [f"L{i+1}" for i in range(max_levels)]
    
    accuracy = {}
    precision = {}
    recall = {}
    f1 = {}
    
    for level_idx in range(max_levels):
        level_name = level_names[level_idx] if level_idx < len(level_names) else f"L{level_idx+1}"
        
        true_at_level = []
        pred_at_level = []
        
        for true_path, pred_path in zip(y_true, y_pred):
            true_label = true_path[level_idx] if level_idx < len(true_path) else None
            pred_label = pred_path[level_idx] if level_idx < len(pred_path) else None
            
            # Only include if both have labels at this level
            if true_label is not None and pred_label is not None:
                true_at_level.append(true_label)
                pred_at_level.append(pred_label)
        
        if not true_at_level:
            accuracy[level_name] = float('nan')
            precision[level_name] = float('nan')
            recall[level_name] = float('nan')
            f1[level_name] = float('nan')
            continue
        
        # Compute accuracy
        correct = sum(1 for t, p in zip(true_at_level, pred_at_level) if t == p)
        accuracy[level_name] = correct / len(true_at_level)
        
        # Compute macro-averaged precision, recall, F1
        # Get unique labels
        all_labels = set(true_at_level) | set(pred_at_level)
        
        level_precisions = []
        level_recalls = []
        
        for label in all_labels:
            tp = sum(1 for t, p in zip(true_at_level, pred_at_level) if t == label and p == label)
            fp = sum(1 for t, p in zip(true_at_level, pred_at_level) if t != label and p == label)
            fn = sum(1 for t, p in zip(true_at_level, pred_at_level) if t == label and p != label)
            
            label_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            label_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            level_precisions.append(label_precision)
            level_recalls.append(label_recall)
        
        precision[level_name] = np.mean(level_precisions) if level_precisions else 0.0
        recall[level_name] = np.mean(level_recalls) if level_recalls else 0.0
        
        p, r = precision[level_name], recall[level_name]
        f1[level_name] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    return accuracy, precision, recall, f1


def depth_weighted_accuracy(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    weights: Optional[List[float]] = None,
) -> float:
    """
    Compute depth-weighted accuracy.
    
    Deeper levels are weighted more heavily by default, reflecting that
    fine-grained classification is more valuable/difficult.
    
    Args:
        y_true: List of true paths
        y_pred: List of predicted paths
        weights: Optional weights for each level. If None, uses [1, 2, 3, ...].
    
    Returns:
        Depth-weighted accuracy score
    """
    accuracy, _, _, _ = per_level_metrics(y_true, y_pred)
    
    if not accuracy:
        return 0.0
    
    num_levels = len(accuracy)
    
    if weights is None:
        # Default: deeper levels more important
        weights = list(range(1, num_levels + 1))
    
    if len(weights) < num_levels:
        # Extend weights if needed
        weights = list(weights) + [weights[-1]] * (num_levels - len(weights))
    
    # Normalize weights
    total_weight = sum(weights[:num_levels])
    normalized_weights = [w / total_weight for w in weights[:num_levels]]
    
    # Compute weighted average
    weighted_sum = 0.0
    for i, (level_name, acc) in enumerate(sorted(accuracy.items())):
        if not np.isnan(acc):
            weighted_sum += normalized_weights[i] * acc
    
    return weighted_sum


def exact_match_accuracy(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> float:
    """
    Compute exact match accuracy (full path must match).
    
    Args:
        y_true: List of true paths
        y_pred: List of predicted paths
    
    Returns:
        Proportion of samples where the entire path matches exactly
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}")
    
    if len(y_true) == 0:
        return 0.0
    
    exact_matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return exact_matches / len(y_true)


def compute_hierarchical_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    level_names: Optional[List[str]] = None,
    depth_weights: Optional[List[float]] = None,
) -> HierarchicalMetrics:
    """
    Compute all hierarchical classification metrics.
    
    Args:
        y_true: List of true paths, e.g., [["Electronics", "Phones", "iPhone"], ...]
        y_pred: List of predicted paths, same format
        level_names: Optional names for levels (e.g., ["L1", "L2", "L3"])
        depth_weights: Optional weights for depth-weighted accuracy
    
    Returns:
        HierarchicalMetrics object containing all computed metrics
    
    Example:
        >>> y_true = [["Animal", "Mammal", "Dog"], ["Animal", "Bird", "Eagle"]]
        >>> y_pred = [["Animal", "Mammal", "Cat"], ["Animal", "Bird", "Eagle"]]
        >>> metrics = compute_hierarchical_metrics(y_true, y_pred)
        >>> print(metrics)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}")
    
    # Compute hierarchical metrics
    hP, hR, hF = hierarchical_precision_recall_f1(y_true, y_pred)
    
    # Compute per-level metrics
    acc, prec, rec, f1 = per_level_metrics(y_true, y_pred, level_names)
    
    # Compute aggregate metrics
    dwa = depth_weighted_accuracy(y_true, y_pred, depth_weights)
    ema = exact_match_accuracy(y_true, y_pred)
    
    return HierarchicalMetrics(
        hierarchical_precision=hP,
        hierarchical_recall=hR,
        hierarchical_f1=hF,
        per_level_accuracy=acc,
        per_level_precision=prec,
        per_level_recall=rec,
        per_level_f1=f1,
        depth_weighted_accuracy=dwa,
        exact_match_accuracy=ema,
        num_samples=len(y_true),
    )


def lca_distance(
    true_path: List[str],
    pred_path: List[str],
) -> int:
    """
    Compute the Lowest Common Ancestor (LCA) distance.
    
    Distance = (depth of true - LCA depth) + (depth of pred - LCA depth)
    Lower is better (0 means perfect match).
    
    Args:
        true_path: True path from root to leaf
        pred_path: Predicted path from root to leaf
    
    Returns:
        LCA distance (0 for perfect match)
    """
    # Find LCA depth (how many levels match from root)
    lca_depth = 0
    for t, p in zip(true_path, pred_path):
        if t == p:
            lca_depth += 1
        else:
            break
    
    # Distance = sum of distances from each leaf to LCA
    true_depth = len(true_path)
    pred_depth = len(pred_path)
    
    return (true_depth - lca_depth) + (pred_depth - lca_depth)


def mean_lca_distance(
    y_true: List[List[str]],
    y_pred: List[List[str]],
) -> float:
    """
    Compute mean LCA distance across all samples.
    
    Args:
        y_true: List of true paths
        y_pred: List of predicted paths
    
    Returns:
        Mean LCA distance (lower is better, 0 is perfect)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}")
    
    if len(y_true) == 0:
        return 0.0
    
    distances = [lca_distance(t, p) for t, p in zip(y_true, y_pred)]
    return np.mean(distances)
