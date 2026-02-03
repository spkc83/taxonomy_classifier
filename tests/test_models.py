"""
Tests for AbstainResult dataclass and ClassificationResult extension.

TDD tests written first before implementation.
"""
from taxonomy_framework import ClassificationResult
from taxonomy_framework.models import AbstainResult


class TestAbstainResult:
    def test_create_with_no_candidates_reason(self, mock_taxonomy):
        """Test AbstainResult with no_candidates reason."""
        result = AbstainResult(
            reason="no_candidates",
            top_candidates=[],
            suggested_action="manual_review",
            partial_path=None,
            input_text="random gibberish"
        )
        assert result.reason == "no_candidates"
        assert result.top_candidates == []
        assert result.suggested_action == "manual_review"
    
    def test_create_with_top_candidates(self, mock_taxonomy):
        """Test AbstainResult with top candidate nodes."""
        tech = mock_taxonomy.root.children[0]  # Tech node
        result = AbstainResult(
            reason="low_confidence",
            top_candidates=[(tech, 0.45)],
            suggested_action="use_best_guess",
            partial_path="Tech",
            input_text="some ambiguous text"
        )
        assert result.reason == "low_confidence"
        assert len(result.top_candidates) == 1
        assert result.top_candidates[0][0].name == "Tech"
        assert result.top_candidates[0][1] == 0.45
        assert result.partial_path == "Tech"
    
    def test_to_dict_serialization(self, mock_taxonomy):
        """Test AbstainResult serializes correctly."""
        tech = mock_taxonomy.root.children[0]
        result = AbstainResult(
            reason="neither_in_contrast",
            top_candidates=[(tech, 0.6)],
            suggested_action="request_clarification",
            partial_path="Tech",
            input_text="test input"
        )
        d = result.to_dict()
        assert d["reason"] == "neither_in_contrast"
        assert d["input_text"] == "test input"
        assert d["suggested_action"] == "request_clarification"
        # top_candidates should serialize node paths, not objects
        assert "top_candidates" in d

    def test_all_valid_reasons(self):
        """Test all 4 valid reason types."""
        for reason in ("no_candidates", "neither_in_contrast", "low_confidence", "explicit_abstain"):
            result = AbstainResult(
                reason=reason,  # type: ignore[arg-type]
                top_candidates=[],
                suggested_action="manual_review",
                partial_path=None,
                input_text="test"
            )
            assert result.reason == reason


class TestClassificationResultExtension:
    def test_traversal_path_default_empty(self, mock_taxonomy):
        """Test traversal_path defaults to empty list."""
        tech = mock_taxonomy.root.children[0]
        result = ClassificationResult(
            input_text="test",
            predicted_category=tech,
            confidence_score=0.9
        )
        assert result.traversal_path == []
    
    def test_traversal_path_with_path(self, mock_taxonomy):
        """Test traversal_path stores the path."""
        hardware = mock_taxonomy.root.children[0].children[0]  # Tech > Hardware
        result = ClassificationResult(
            input_text="laptop issue",
            predicted_category=hardware,
            confidence_score=0.95,
            traversal_path=["Tech", "Hardware"]
        )
        assert result.traversal_path == ["Tech", "Hardware"]
    
    def test_to_dict_includes_traversal_path(self, mock_taxonomy):
        """Test to_dict includes traversal_path."""
        hardware = mock_taxonomy.root.children[0].children[0]
        result = ClassificationResult(
            input_text="test",
            predicted_category=hardware,
            confidence_score=0.9,
            traversal_path=["Tech", "Hardware"]
        )
        d = result.to_dict()
        assert "traversal_path" in d
        assert d["traversal_path"] == ["Tech", "Hardware"]
