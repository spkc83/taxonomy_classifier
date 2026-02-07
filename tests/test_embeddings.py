"""Tests for embedding models including SetFitBackend."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from taxonomy_framework.embeddings import (
    EmbeddingModel,
    SentenceTransformerBackend,
    EnsembleEmbedder,
)
from tests.conftest import MockEmbeddingModel


# ============================================================================
# SetFitBackend Tests
# ============================================================================

class TestSetFitBackendImportError:
    """Test SetFitBackend behavior when setfit is not installed."""

    def test_setfit_import_error_raised_when_not_installed(self):
        """SetFitBackend raises ImportError when setfit library is not available."""
        # Patch SetFitModel to None to simulate missing library
        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': None}):
            # Need to re-import to get the patched version
            from taxonomy_framework.embeddings import SetFitBackend
            with pytest.raises(ImportError, match="setfit not installed"):
                SetFitBackend("some-model")


class TestSetFitBackendInterface:
    """Test SetFitBackend implements EmbeddingModel interface correctly."""

    def test_setfit_backend_is_embedding_model_subclass(self):
        """SetFitBackend should be a subclass of EmbeddingModel."""
        from taxonomy_framework.embeddings import SetFitBackend
        assert issubclass(SetFitBackend, EmbeddingModel)

    def test_setfit_backend_has_embed_text_method(self):
        """SetFitBackend should have embed_text method."""
        from taxonomy_framework.embeddings import SetFitBackend
        assert hasattr(SetFitBackend, 'embed_text')

    def test_setfit_backend_has_embed_batch_method(self):
        """SetFitBackend should have embed_batch method."""
        from taxonomy_framework.embeddings import SetFitBackend
        assert hasattr(SetFitBackend, 'embed_batch')


class TestSetFitBackendWithMock:
    """Test SetFitBackend functionality with mocked SetFitModel."""

    def test_embed_text_returns_numpy_array(self):
        """embed_text should return a numpy array."""
        mock_model = MagicMock()
        mock_model.model_body.encode.return_value = np.array([0.1, 0.2, 0.3])
        
        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': MagicMock()}):
            from taxonomy_framework.embeddings import SetFitBackend
            with patch.object(SetFitBackend, '__init__', lambda self, model_name: None):
                backend = SetFitBackend.__new__(SetFitBackend)
                backend.model = mock_model
                
                result = backend.embed_text("test text")
                assert isinstance(result, np.ndarray)

    def test_embed_batch_returns_numpy_array(self):
        """embed_batch should return a 2D numpy array."""
        mock_model = MagicMock()
        mock_model.model_body.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': MagicMock()}):
            from taxonomy_framework.embeddings import SetFitBackend
            with patch.object(SetFitBackend, '__init__', lambda self, model_name: None):
                backend = SetFitBackend.__new__(SetFitBackend)
                backend.model = mock_model
                
                result = backend.embed_batch(["text1", "text2"])
                assert isinstance(result, np.ndarray)

    def test_embed_text_calls_model_body_encode(self):
        """embed_text should use model.model_body.encode()."""
        mock_model = MagicMock()
        mock_model.model_body.encode.return_value = np.array([0.1, 0.2, 0.3])
        
        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': MagicMock()}):
            from taxonomy_framework.embeddings import SetFitBackend
            with patch.object(SetFitBackend, '__init__', lambda self, model_name: None):
                backend = SetFitBackend.__new__(SetFitBackend)
                backend.model = mock_model
                
                backend.embed_text("hello world")
                mock_model.model_body.encode.assert_called_once_with("hello world", normalize_embeddings=True)

    def test_embed_batch_calls_model_body_encode(self):
        """embed_batch should use model.model_body.encode()."""
        mock_model = MagicMock()
        mock_model.model_body.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': MagicMock()}):
            from taxonomy_framework.embeddings import SetFitBackend
            with patch.object(SetFitBackend, '__init__', lambda self, model_name: None):
                backend = SetFitBackend.__new__(SetFitBackend)
                backend.model = mock_model
                
                texts = ["text1", "text2"]
                backend.embed_batch(texts)
                mock_model.model_body.encode.assert_called_once_with(texts, normalize_embeddings=True)


class TestSetFitBackendInitialization:
    """Test SetFitBackend initialization with mocked SetFitModel."""

    def test_init_loads_model_from_pretrained(self):
        """__init__ should load model using SetFitModel.from_pretrained()."""
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = MagicMock()
        
        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': mock_setfit_model_cls}):
            from taxonomy_framework.embeddings import SetFitBackend
            # Reload to get patched version
            backend = SetFitBackend("test-model-name")
            mock_setfit_model_cls.from_pretrained.assert_called_once_with("test-model-name")

    def test_init_default_model_name(self):
        """__init__ should have a sensible default model name."""
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = MagicMock()
        
        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': mock_setfit_model_cls}):
            from taxonomy_framework.embeddings import SetFitBackend
            # Call without args to use default
            backend = SetFitBackend()
            # Just verify it was called with some default
            mock_setfit_model_cls.from_pretrained.assert_called_once()


# ============================================================================
# Existing Embeddings Tests (for completeness)
# ============================================================================

class TestMockEmbeddingModel:
    """Tests for MockEmbeddingModel."""

    def test_embed_text_returns_correct_dimension(self):
        """embed_text should return array with specified dimension."""
        model = MockEmbeddingModel(dim=128)
        result = model.embed_text("test")
        assert result.shape == (128,)

    def test_embed_batch_returns_correct_shape(self):
        """embed_batch should return 2D array with correct shape."""
        model = MockEmbeddingModel(dim=64)
        result = model.embed_batch(["a", "b", "c"])
        assert result.shape == (3, 64)


class TestSetFitBackendFromFinetuned:
    """Test SetFitBackend.from_finetuned() class method."""

    def test_from_finetuned_loads_model_from_path(self):
        """from_finetuned should load model using SetFitModel.from_pretrained()."""
        mock_setfit_model_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = mock_model_instance

        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': mock_setfit_model_cls}):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.isfile', return_value=True):
                    from taxonomy_framework.embeddings import SetFitBackend
                    backend = SetFitBackend.from_finetuned("/path/to/model")
                    mock_setfit_model_cls.from_pretrained.assert_called_once_with("/path/to/model")

    def test_from_finetuned_raises_for_nonexistent_path(self):
        """from_finetuned should raise ValueError for nonexistent path."""
        mock_setfit_model_cls = MagicMock()

        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': mock_setfit_model_cls}):
            with patch('os.path.exists', return_value=False):
                from taxonomy_framework.embeddings import SetFitBackend
                with pytest.raises(ValueError, match="Model path does not exist: /nonexistent/path"):
                    SetFitBackend.from_finetuned("/nonexistent/path")

    def test_from_finetuned_raises_for_invalid_model_dir(self):
        """from_finetuned should raise ValueError for directory without model files."""
        mock_setfit_model_cls = MagicMock()

        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': mock_setfit_model_cls}):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.isfile', return_value=False):
                    from taxonomy_framework.embeddings import SetFitBackend
                    with pytest.raises(ValueError, match="Invalid model directory: missing model files"):
                        SetFitBackend.from_finetuned("/path/with/no/model")

    def test_from_finetuned_returns_setfit_backend(self):
        """from_finetuned should return a SetFitBackend instance."""
        mock_setfit_model_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = mock_model_instance

        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': mock_setfit_model_cls}):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.isfile', return_value=True):
                    from taxonomy_framework.embeddings import SetFitBackend
                    backend = SetFitBackend.from_finetuned("/path/to/model")
                    assert isinstance(backend, SetFitBackend)

    def test_from_finetuned_model_works_for_embedding(self):
        """Loaded model should work with embed_text and embed_batch."""
        mock_setfit_model_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.model_body.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_setfit_model_cls.from_pretrained.return_value = mock_model_instance

        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': mock_setfit_model_cls}):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.isfile', return_value=True):
                    from taxonomy_framework.embeddings import SetFitBackend
                    backend = SetFitBackend.from_finetuned("/path/to/model")
                    
                    result = backend.embed_text("test text")
                    assert isinstance(result, np.ndarray)
                    
                    mock_model_instance.model_body.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
                    result = backend.embed_batch(["text1", "text2"])
                    assert isinstance(result, np.ndarray)


class TestEnsembleEmbedder:
    """Tests for EnsembleEmbedder."""

    def test_ensemble_requires_at_least_one_model(self):
        """EnsembleEmbedder should raise ValueError for empty model list."""
        with pytest.raises(ValueError, match="At least one embedding model"):
            EnsembleEmbedder([])

    def test_retrieve_candidates_returns_indices(self):
        """retrieve_candidates should return list of indices."""
        model = MockEmbeddingModel(dim=10)
        ensemble = EnsembleEmbedder([model])
        candidates = ["cat", "dog", "bird"]
        result = ensemble.retrieve_candidates("animal", candidates, top_k=2)
        assert len(result) == 2
        assert all(isinstance(idx, (int, np.integer)) for idx in result)
