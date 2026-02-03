"""Tests for the hierarchical classification evaluation module."""

import pytest
import numpy as np
from taxonomy_framework.evaluation import (
    HierarchicalMetrics,
    compute_hierarchical_metrics,
    hierarchical_precision_recall_f1,
    per_level_metrics,
    depth_weighted_accuracy,
    exact_match_accuracy,
    lca_distance,
    mean_lca_distance,
)


class TestHierarchicalPrecisionRecallF1:
    """Tests for hierarchical precision, recall, and F1 computation."""

    def test_perfect_predictions(self):
        y_true = [
            ["Animal", "Mammal", "Dog"],
            ["Animal", "Bird", "Eagle"],
        ]
        y_pred = [
            ["Animal", "Mammal", "Dog"],
            ["Animal", "Bird", "Eagle"],
        ]
        
        hP, hR, hF = hierarchical_precision_recall_f1(y_true, y_pred)
        
        assert hP == 1.0
        assert hR == 1.0
        assert hF == 1.0

    def test_partial_match_same_branch(self):
        y_true = [["Animal", "Mammal", "Dog"]]
        y_pred = [["Animal", "Mammal", "Cat"]]
        
        hP, hR, hF = hierarchical_precision_recall_f1(y_true, y_pred)
        
        assert hP == pytest.approx(2/3)
        assert hR == pytest.approx(2/3)
        assert hF == pytest.approx(2/3)

    def test_partial_match_different_branch(self):
        y_true = [["Animal", "Mammal", "Dog"]]
        y_pred = [["Animal", "Bird", "Eagle"]]
        
        hP, hR, hF = hierarchical_precision_recall_f1(y_true, y_pred)
        
        assert hP == pytest.approx(1/3)
        assert hR == pytest.approx(1/3)
        assert hF == pytest.approx(1/3)

    def test_empty_inputs(self):
        hP, hR, hF = hierarchical_precision_recall_f1([], [])
        
        assert hP == 0.0
        assert hR == 0.0
        assert hF == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            hierarchical_precision_recall_f1(
                [["A", "B"]],
                [["A", "B"], ["C", "D"]]
            )

    def test_different_path_lengths(self):
        y_true = [["A", "B", "C"]]
        y_pred = [["A", "B"]]
        
        hP, hR, hF = hierarchical_precision_recall_f1(y_true, y_pred)
        
        assert hP == 1.0
        assert hR == pytest.approx(2/3)

    def test_multiple_samples(self):
        y_true = [
            ["Animal", "Mammal", "Dog"],
            ["Animal", "Bird", "Eagle"],
            ["Plant", "Tree", "Oak"],
        ]
        y_pred = [
            ["Animal", "Mammal", "Dog"],
            ["Animal", "Mammal", "Cat"],
            ["Plant", "Flower", "Rose"],
        ]
        
        hP, hR, hF = hierarchical_precision_recall_f1(y_true, y_pred)
        
        assert 0 < hP < 1
        assert 0 < hR < 1
        assert 0 < hF < 1


class TestPerLevelMetrics:
    """Tests for per-level accuracy, precision, recall, and F1."""

    def test_perfect_predictions(self):
        y_true = [["A", "B", "C"], ["D", "E", "F"]]
        y_pred = [["A", "B", "C"], ["D", "E", "F"]]
        
        acc, prec, rec, f1 = per_level_metrics(y_true, y_pred)
        
        assert acc["L1"] == 1.0
        assert acc["L2"] == 1.0
        assert acc["L3"] == 1.0

    def test_partial_accuracy(self):
        y_true = [["A", "B", "C"], ["A", "B", "D"]]
        y_pred = [["A", "B", "C"], ["A", "X", "Y"]]
        
        acc, _, _, _ = per_level_metrics(y_true, y_pred)
        
        assert acc["L1"] == 1.0
        assert acc["L2"] == 0.5
        assert acc["L3"] == 0.5

    def test_custom_level_names(self):
        y_true = [["Cat1", "Cat2"]]
        y_pred = [["Cat1", "Cat2"]]
        
        acc, _, _, _ = per_level_metrics(y_true, y_pred, level_names=["Top", "Mid"])
        
        assert "Top" in acc
        assert "Mid" in acc
        assert "L1" not in acc

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            per_level_metrics([["A"]], [["A"], ["B"]])


class TestDepthWeightedAccuracy:
    """Tests for depth-weighted accuracy."""

    def test_perfect_predictions(self):
        y_true = [["A", "B", "C"]]
        y_pred = [["A", "B", "C"]]
        
        dwa = depth_weighted_accuracy(y_true, y_pred)
        
        assert dwa == 1.0

    def test_deeper_levels_weighted_more(self):
        y_true = [["A", "B", "C"], ["A", "B", "C"]]
        y_pred = [["A", "X", "Y"], ["A", "B", "C"]]
        
        dwa = depth_weighted_accuracy(y_true, y_pred)
        
        assert 0 < dwa < 1

    def test_custom_weights(self):
        y_true = [["A", "B"]]
        y_pred = [["A", "X"]]
        
        equal_weights = depth_weighted_accuracy(y_true, y_pred, weights=[1, 1])
        deeper_weights = depth_weighted_accuracy(y_true, y_pred, weights=[1, 3])
        
        assert deeper_weights < equal_weights

    def test_empty_returns_zero(self):
        assert depth_weighted_accuracy([], []) == 0.0


class TestExactMatchAccuracy:
    """Tests for exact match accuracy."""

    def test_all_match(self):
        y_true = [["A", "B"], ["C", "D"]]
        y_pred = [["A", "B"], ["C", "D"]]
        
        assert exact_match_accuracy(y_true, y_pred) == 1.0

    def test_none_match(self):
        y_true = [["A", "B"], ["C", "D"]]
        y_pred = [["X", "Y"], ["Z", "W"]]
        
        assert exact_match_accuracy(y_true, y_pred) == 0.0

    def test_partial_match(self):
        y_true = [["A", "B"], ["C", "D"]]
        y_pred = [["A", "B"], ["X", "Y"]]
        
        assert exact_match_accuracy(y_true, y_pred) == 0.5

    def test_partial_path_no_match(self):
        y_true = [["A", "B", "C"]]
        y_pred = [["A", "B"]]
        
        assert exact_match_accuracy(y_true, y_pred) == 0.0

    def test_empty_returns_zero(self):
        assert exact_match_accuracy([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            exact_match_accuracy([["A"]], [])


class TestLCADistance:
    """Tests for Lowest Common Ancestor distance."""

    def test_identical_paths(self):
        assert lca_distance(["A", "B", "C"], ["A", "B", "C"]) == 0

    def test_sibling_leaves(self):
        assert lca_distance(["A", "B", "C"], ["A", "B", "D"]) == 2

    def test_different_subtrees(self):
        assert lca_distance(["A", "B", "C"], ["A", "X", "Y"]) == 4

    def test_completely_different(self):
        assert lca_distance(["A", "B"], ["X", "Y"]) == 4

    def test_different_depths(self):
        assert lca_distance(["A", "B", "C"], ["A", "B"]) == 1


class TestMeanLCADistance:
    """Tests for mean LCA distance."""

    def test_perfect_predictions(self):
        y_true = [["A", "B"], ["C", "D"]]
        y_pred = [["A", "B"], ["C", "D"]]
        
        assert mean_lca_distance(y_true, y_pred) == 0.0

    def test_mixed_distances(self):
        y_true = [["A", "B", "C"], ["A", "B", "C"]]
        y_pred = [["A", "B", "C"], ["A", "B", "X"]]
        
        mean_dist = mean_lca_distance(y_true, y_pred)
        
        assert mean_dist == 1.0

    def test_empty_returns_zero(self):
        assert mean_lca_distance([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            mean_lca_distance([["A"]], [])


class TestComputeHierarchicalMetrics:
    """Tests for the main compute_hierarchical_metrics function."""

    def test_returns_hierarchical_metrics_object(self):
        y_true = [["A", "B", "C"]]
        y_pred = [["A", "B", "C"]]
        
        metrics = compute_hierarchical_metrics(y_true, y_pred)
        
        assert isinstance(metrics, HierarchicalMetrics)

    def test_contains_all_metrics(self):
        y_true = [["A", "B", "C"], ["D", "E", "F"]]
        y_pred = [["A", "B", "X"], ["D", "Y", "Z"]]
        
        metrics = compute_hierarchical_metrics(y_true, y_pred)
        
        assert hasattr(metrics, "hierarchical_precision")
        assert hasattr(metrics, "hierarchical_recall")
        assert hasattr(metrics, "hierarchical_f1")
        assert hasattr(metrics, "per_level_accuracy")
        assert hasattr(metrics, "depth_weighted_accuracy")
        assert hasattr(metrics, "exact_match_accuracy")
        assert hasattr(metrics, "num_samples")

    def test_num_samples_correct(self):
        y_true = [["A"], ["B"], ["C"]]
        y_pred = [["A"], ["B"], ["C"]]
        
        metrics = compute_hierarchical_metrics(y_true, y_pred)
        
        assert metrics.num_samples == 3

    def test_custom_level_names(self):
        y_true = [["Top", "Mid", "Bottom"]]
        y_pred = [["Top", "Mid", "Bottom"]]
        
        metrics = compute_hierarchical_metrics(
            y_true, y_pred,
            level_names=["Category", "Subcategory", "Item"]
        )
        
        assert "Category" in metrics.per_level_accuracy
        assert "Subcategory" in metrics.per_level_accuracy
        assert "Item" in metrics.per_level_accuracy

    def test_to_dict(self):
        y_true = [["A", "B"]]
        y_pred = [["A", "B"]]
        
        metrics = compute_hierarchical_metrics(y_true, y_pred)
        result = metrics.to_dict()
        
        assert isinstance(result, dict)
        assert "hierarchical_f1" in result
        assert "per_level_accuracy" in result

    def test_str_representation(self):
        y_true = [["A", "B"]]
        y_pred = [["A", "X"]]
        
        metrics = compute_hierarchical_metrics(y_true, y_pred)
        result = str(metrics)
        
        assert "Hierarchical" in result
        assert "Per-Level" in result


class TestHierarchicalMetricsDataclass:
    """Tests for the HierarchicalMetrics dataclass."""

    def test_to_dict_serializable(self):
        metrics = HierarchicalMetrics(
            hierarchical_precision=0.8,
            hierarchical_recall=0.7,
            hierarchical_f1=0.75,
            per_level_accuracy={"L1": 0.9, "L2": 0.8},
            per_level_precision={"L1": 0.85, "L2": 0.75},
            per_level_recall={"L1": 0.88, "L2": 0.78},
            per_level_f1={"L1": 0.86, "L2": 0.76},
            depth_weighted_accuracy=0.85,
            exact_match_accuracy=0.6,
            num_samples=100,
        )
        
        result = metrics.to_dict()
        
        import json
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_str_output_readable(self):
        metrics = HierarchicalMetrics(
            hierarchical_precision=0.8,
            hierarchical_recall=0.7,
            hierarchical_f1=0.75,
            per_level_accuracy={"L1": 0.9},
            per_level_precision={"L1": 0.85},
            per_level_recall={"L1": 0.88},
            per_level_f1={"L1": 0.86},
            depth_weighted_accuracy=0.85,
            exact_match_accuracy=0.6,
            num_samples=100,
        )
        
        output = str(metrics)
        
        assert "0.8" in output or "0.80" in output
        assert "100" in output
        assert "L1" in output


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_level_hierarchy(self):
        y_true = [["A"], ["B"], ["A"]]
        y_pred = [["A"], ["B"], ["B"]]
        
        metrics = compute_hierarchical_metrics(y_true, y_pred)
        
        assert metrics.hierarchical_f1 == pytest.approx(2/3)

    def test_very_deep_hierarchy(self):
        deep_path = ["L" + str(i) for i in range(10)]
        y_true = [deep_path]
        y_pred = [deep_path]
        
        metrics = compute_hierarchical_metrics(y_true, y_pred)
        
        assert metrics.hierarchical_f1 == 1.0
        assert len(metrics.per_level_accuracy) == 10

    def test_empty_predicted_path(self):
        y_true = [["A", "B", "C"]]
        y_pred = [[]]
        
        hP, hR, hF = hierarchical_precision_recall_f1(y_true, y_pred)
        
        assert hP == 0.0
        assert hR == 0.0
        assert hF == 0.0

    def test_many_classes_same_level(self):
        y_true = [[f"Class{i}", "Sub", "Leaf"] for i in range(100)]
        y_pred = [[f"Class{i}", "Sub", "Leaf"] for i in range(100)]
        
        metrics = compute_hierarchical_metrics(y_true, y_pred)
        
        assert metrics.exact_match_accuracy == 1.0
