"""Tests for SetFitTrainer training module."""

import pytest
from unittest.mock import MagicMock, patch


def _create_valid_mock_dataset():
    """Create a mock dataset that passes validation (16 samples, 2 classes, 8 each)."""
    texts = [f"text{i}" for i in range(16)]
    labels = ["class_a"] * 8 + ["class_b"] * 8
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=16)
    mock_dataset.__getitem__ = MagicMock(side_effect=lambda x: 
        texts if x == "text" else labels if x == "label" else None
    )
    return mock_dataset


# ============================================================================
# SetFitTrainer Import Tests
# ============================================================================

class TestSetFitTrainerImportError:
    """Test SetFitTrainer behavior when setfit is not installed."""

    def test_setfit_import_error_raised_when_not_installed(self):
        """SetFitTrainer raises ImportError when setfit library is not available."""
        with patch.dict('taxonomy_framework.training.__dict__', {'Trainer': None, 'SetFitModel': None}):
            from taxonomy_framework.training import SetFitTrainer
            with pytest.raises(ImportError, match="setfit not installed. Run: pip install setfit"):
                SetFitTrainer()


class TestSetFitTrainerInterface:
    """Test SetFitTrainer has required interface."""

    def test_setfit_trainer_has_train_method(self):
        """SetFitTrainer should have train method."""
        from taxonomy_framework.training import SetFitTrainer
        assert hasattr(SetFitTrainer, 'train')

    def test_setfit_trainer_has_save_method(self):
        """SetFitTrainer should have save method."""
        from taxonomy_framework.training import SetFitTrainer
        assert hasattr(SetFitTrainer, 'save')


# ============================================================================
# SetFitTrainer Initialization Tests
# ============================================================================

class TestSetFitTrainerInit:
    """Test SetFitTrainer initialization with mocked setfit."""

    def test_init_default_base_model(self):
        """__init__ should use default base_model if not provided."""
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = MagicMock()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mock_setfit_model_cls,
            'Trainer': MagicMock(),
            'TrainingArguments': MagicMock(),
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            mock_setfit_model_cls.from_pretrained.assert_called_once_with(
                "sentence-transformers/paraphrase-mpnet-base-v2"
            )

    def test_init_custom_base_model(self):
        """__init__ should use provided base_model."""
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = MagicMock()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mock_setfit_model_cls,
            'Trainer': MagicMock(),
            'TrainingArguments': MagicMock(),
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer(base_model="custom/model-name")
            mock_setfit_model_cls.from_pretrained.assert_called_once_with("custom/model-name")

    def test_init_default_output_dir(self):
        """__init__ should use default output_dir if not provided."""
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = MagicMock()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mock_setfit_model_cls,
            'Trainer': MagicMock(),
            'TrainingArguments': MagicMock(),
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            assert trainer.output_dir == "./models/setfit-model"

    def test_init_custom_output_dir(self):
        """__init__ should use provided output_dir."""
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = MagicMock()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mock_setfit_model_cls,
            'Trainer': MagicMock(),
            'TrainingArguments': MagicMock(),
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer(output_dir="/custom/path")
            assert trainer.output_dir == "/custom/path"


# ============================================================================
# SetFitTrainer Train Method Tests
# ============================================================================

class TestSetFitTrainerTrain:
    """Test SetFitTrainer.train() method with mocked setfit."""

    def test_train_creates_setfit_trainer(self):
        """train() should create a SetFit Trainer instance."""
        mock_setfit_model = MagicMock()
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = mock_setfit_model
        
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        
        mock_training_args_cls = MagicMock()
        mock_training_args = MagicMock()
        mock_training_args_cls.return_value = mock_training_args
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mock_setfit_model_cls,
            'Trainer': mock_trainer_cls,
            'TrainingArguments': mock_training_args_cls,
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            mock_dataset = _create_valid_mock_dataset()
            trainer.train(mock_dataset, text_column="text", label_column="label", auto_split=False)
            
            mock_trainer_cls.assert_called_once()

    def test_train_calls_trainer_train(self):
        """train() should call the internal Trainer.train()."""
        mock_setfit_model = MagicMock()
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = mock_setfit_model
        
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        
        mock_training_args_cls = MagicMock()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mock_setfit_model_cls,
            'Trainer': mock_trainer_cls,
            'TrainingArguments': mock_training_args_cls,
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            mock_dataset = _create_valid_mock_dataset()
            trainer.train(mock_dataset, text_column="text", label_column="label", auto_split=False)
            
            mock_trainer_instance.train.assert_called_once()

    def test_train_with_eval_dataset(self):
        """train() should accept optional eval_dataset."""
        mock_setfit_model = MagicMock()
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = mock_setfit_model
        
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        
        mock_training_args_cls = MagicMock()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mock_setfit_model_cls,
            'Trainer': mock_trainer_cls,
            'TrainingArguments': mock_training_args_cls,
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            mock_train_dataset = _create_valid_mock_dataset()
            mock_eval_dataset = MagicMock()
            trainer.train(
                mock_train_dataset,
                text_column="text",
                label_column="label",
                eval_dataset=mock_eval_dataset
            )
            
            call_kwargs = mock_trainer_cls.call_args[1]
            assert 'eval_dataset' in call_kwargs
            assert call_kwargs['eval_dataset'] == mock_eval_dataset

    def test_train_passes_training_args(self):
        """train() should pass **training_args to TrainingArguments."""
        mock_setfit_model = MagicMock()
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = mock_setfit_model
        
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        
        mock_training_args_cls = MagicMock()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mock_setfit_model_cls,
            'Trainer': mock_trainer_cls,
            'TrainingArguments': mock_training_args_cls,
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            mock_dataset = _create_valid_mock_dataset()
            trainer.train(
                mock_dataset,
                text_column="text",
                label_column="label",
                batch_size=32,
                num_epochs=10,
                auto_split=False
            )
            
            call_kwargs = mock_training_args_cls.call_args[1]
            assert call_kwargs.get('batch_size') == 32
            assert call_kwargs.get('num_epochs') == 10


# ============================================================================
# SetFitTrainer Save Method Tests
# ============================================================================

class TestSetFitTrainerSave:
    """Test SetFitTrainer.save() method with mocked setfit."""

    def test_save_calls_model_save_pretrained(self):
        """save() should call model.save_pretrained() with output_dir."""
        mock_setfit_model = MagicMock()
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = mock_setfit_model
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mock_setfit_model_cls,
            'Trainer': MagicMock(),
            'TrainingArguments': MagicMock(),
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer(output_dir="/my/output/dir")
            trainer.save()
            
            mock_setfit_model.save_pretrained.assert_called_once_with("/my/output/dir")

    def test_save_uses_default_output_dir(self):
        """save() should use default output_dir when not specified."""
        mock_setfit_model = MagicMock()
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = mock_setfit_model
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mock_setfit_model_cls,
            'Trainer': MagicMock(),
            'TrainingArguments': MagicMock(),
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            trainer.save()
            
            mock_setfit_model.save_pretrained.assert_called_once_with("./models/setfit-model")


# ============================================================================
# SetFitTrainer Data Format Support Tests
# ============================================================================

class TestDataFormats:
    """Test SetFitTrainer data format support (HuggingFace Dataset and Python lists)."""

    def _create_trainer_with_mocks(self):
        """Helper to create a mocked SetFitTrainer."""
        mock_setfit_model = MagicMock()
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = mock_setfit_model
        
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        
        mock_training_args_cls = MagicMock()
        mock_training_args = MagicMock()
        mock_training_args_cls.return_value = mock_training_args
        
        return {
            'setfit_model': mock_setfit_model,
            'setfit_model_cls': mock_setfit_model_cls,
            'trainer_cls': mock_trainer_cls,
            'trainer_instance': mock_trainer_instance,
            'training_args_cls': mock_training_args_cls,
            'training_args': mock_training_args,
        }

    def test_train_with_huggingface_dataset(self):
        """train() should accept HuggingFace Dataset directly."""
        mocks = self._create_trainer_with_mocks()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            mock_hf_dataset = _create_valid_mock_dataset()
            trainer.train(dataset=mock_hf_dataset, auto_split=False)
            
            mocks['trainer_instance'].train.assert_called_once()
            call_kwargs = mocks['trainer_cls'].call_args[1]
            assert call_kwargs['train_dataset'] == mock_hf_dataset

    def test_train_with_python_lists(self):
        """train() should accept Python lists (texts and labels) and convert to HuggingFace Dataset."""
        mocks = self._create_trainer_with_mocks()
        
        texts = [f"text{i}" for i in range(16)]
        labels = ["class_a"] * 8 + ["class_b"] * 8
        
        mock_dataset_cls = MagicMock()
        mock_created_dataset = MagicMock()
        mock_created_dataset.__len__ = MagicMock(return_value=16)
        mock_created_dataset.__getitem__ = MagicMock(side_effect=lambda x: 
            texts if x == "text" else labels if x == "label" else None
        )
        mock_dataset_cls.from_dict.return_value = mock_created_dataset
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
            'Dataset': mock_dataset_cls,
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            trainer.train(texts=texts, labels=labels, auto_split=False)
            
            mock_dataset_cls.from_dict.assert_called_once_with({
                "text": texts,
                "label": labels
            })
            call_kwargs = mocks['trainer_cls'].call_args[1]
            assert call_kwargs['train_dataset'] == mock_created_dataset

    def test_train_with_column_mapping(self):
        """train() should apply column_mapping for non-standard column names."""
        mocks = self._create_trainer_with_mocks()
        
        texts = [f"text{i}" for i in range(16)]
        labels = ["class_a"] * 8 + ["class_b"] * 8
        
        mock_hf_dataset = MagicMock()
        mock_hf_dataset.__len__ = MagicMock(return_value=16)
        mock_hf_dataset.__getitem__ = MagicMock(side_effect=lambda x: 
            texts if x == "my_text" else labels if x == "my_label" else None
        )
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            trainer.train(
                dataset=mock_hf_dataset,
                text_column="my_text",
                label_column="my_label",
                auto_split=False
            )
            
            call_kwargs = mocks['trainer_cls'].call_args[1]
            assert call_kwargs['column_mapping'] == {"my_text": "text", "my_label": "label"}

    def test_train_error_when_no_dataset_or_lists(self):
        """train() should raise ValueError when neither dataset nor texts/labels provided."""
        mocks = self._create_trainer_with_mocks()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            with pytest.raises(ValueError, match="Either 'dataset' or both 'texts' and 'labels' must be provided"):
                trainer.train()

    def test_train_error_when_only_texts_provided(self):
        """train() should raise ValueError when only texts provided without labels."""
        mocks = self._create_trainer_with_mocks()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            with pytest.raises(ValueError, match="Both 'texts' and 'labels' must be provided together"):
                trainer.train(texts=["text1", "text2"])

    def test_train_error_when_only_labels_provided(self):
        """train() should raise ValueError when only labels provided without texts."""
        mocks = self._create_trainer_with_mocks()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            with pytest.raises(ValueError, match="Both 'texts' and 'labels' must be provided together"):
                trainer.train(labels=["label1", "label2"])

    def test_datasets_import_error_message(self):
        """train() with lists should raise ImportError with actionable message when datasets not installed."""
        mocks = self._create_trainer_with_mocks()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
            'Dataset': None,  # Simulate datasets not installed
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            with pytest.raises(ImportError, match="datasets not installed. Run: pip install datasets"):
                trainer.train(texts=["text1"], labels=["label1"])


# ============================================================================
# SetFitTrainer Validation Tests
# ============================================================================

class TestValidation:
    """Test SetFitTrainer dataset validation."""

    def _create_trainer_with_mocks(self):
        """Helper to create a mocked SetFitTrainer."""
        mock_setfit_model = MagicMock()
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = mock_setfit_model
        
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        
        mock_training_args_cls = MagicMock()
        mock_training_args = MagicMock()
        mock_training_args_cls.return_value = mock_training_args
        
        return {
            'setfit_model': mock_setfit_model,
            'setfit_model_cls': mock_setfit_model_cls,
            'trainer_cls': mock_trainer_cls,
            'trainer_instance': mock_trainer_instance,
            'training_args_cls': mock_training_args_cls,
            'training_args': mock_training_args,
        }

    def test_validation_empty_dataset(self):
        """train() should raise ValueError for empty dataset."""
        mocks = self._create_trainer_with_mocks()
        
        # Create mock empty dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=0)
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda x: [] if x in ["text", "label"] else None)
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            with pytest.raises(ValueError, match="Dataset cannot be empty"):
                trainer.train(dataset=mock_dataset)

    def test_validation_single_class(self):
        """train() should raise ValueError when dataset has only one class."""
        mocks = self._create_trainer_with_mocks()
        
        # Create mock dataset with single class
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda x: 
            ["text"] * 10 if x == "text" else ["label_a"] * 10 if x == "label" else None
        )
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            with pytest.raises(ValueError, match="Need at least 2 classes for classification"):
                trainer.train(dataset=mock_dataset)

    def test_validation_insufficient_samples(self):
        """train() should raise ValueError when a class has fewer than 8 samples."""
        mocks = self._create_trainer_with_mocks()
        
        # Create mock dataset with one class having only 5 samples
        texts = ["text"] * 15  # 10 for class A, 5 for class B
        labels = ["class_a"] * 10 + ["class_b"] * 5
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=15)
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda x: 
            texts if x == "text" else labels if x == "label" else None
        )
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            with pytest.raises(ValueError, match="Class 'class_b' has only 5 samples. Minimum 8 required."):
                trainer.train(dataset=mock_dataset)

    def test_validation_null_text(self):
        """train() should raise ValueError when dataset contains null/empty text."""
        mocks = self._create_trainer_with_mocks()
        
        # Create mock dataset with null text at index 2
        texts = ["text1", "text2", None, "text4", "text5", "text6", "text7", "text8",
                 "text9", "text10", "text11", "text12", "text13", "text14", "text15", "text16"]
        labels = ["class_a"] * 8 + ["class_b"] * 8
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=16)
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda x: 
            texts if x == "text" else labels if x == "label" else None
        )
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            with pytest.raises(ValueError, match="Found null/empty text at index 2"):
                trainer.train(dataset=mock_dataset)

    def test_validation_empty_string_text(self):
        """train() should raise ValueError when dataset contains empty string text."""
        mocks = self._create_trainer_with_mocks()
        
        # Create mock dataset with empty string text at index 5
        texts = ["text1", "text2", "text3", "text4", "text5", "", "text7", "text8",
                 "text9", "text10", "text11", "text12", "text13", "text14", "text15", "text16"]
        labels = ["class_a"] * 8 + ["class_b"] * 8
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=16)
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda x: 
            texts if x == "text" else labels if x == "label" else None
        )
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            with pytest.raises(ValueError, match="Found null/empty text at index 5"):
                trainer.train(dataset=mock_dataset)

    def test_validation_whitespace_only_text(self):
        """train() should raise ValueError when dataset contains whitespace-only text."""
        mocks = self._create_trainer_with_mocks()
        
        # Create mock dataset with whitespace-only text at index 3
        texts = ["text1", "text2", "text3", "   ", "text5", "text6", "text7", "text8",
                 "text9", "text10", "text11", "text12", "text13", "text14", "text15", "text16"]
        labels = ["class_a"] * 8 + ["class_b"] * 8
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=16)
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda x: 
            texts if x == "text" else labels if x == "label" else None
        )
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            with pytest.raises(ValueError, match="Found null/empty text at index 3"):
                trainer.train(dataset=mock_dataset)

    def test_validation_passes_valid_dataset(self):
        """train() should proceed with valid dataset (2+ classes, 8+ samples each, no null texts)."""
        mocks = self._create_trainer_with_mocks()
        
        # Create valid mock dataset
        texts = [f"text{i}" for i in range(16)]
        labels = ["class_a"] * 8 + ["class_b"] * 8
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=16)
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda x: 
            texts if x == "text" else labels if x == "label" else None
        )
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            # Should not raise any exception
            trainer.train(dataset=mock_dataset)
            mocks['trainer_instance'].train.assert_called_once()

    def test_auto_split_creates_eval_dataset(self):
        """train() with auto_split=True and no eval_dataset should create 80/20 split."""
        mocks = self._create_trainer_with_mocks()
        
        texts = [f"text{i}" for i in range(16)]
        labels = ["class_a"] * 8 + ["class_b"] * 8
        
        mock_train_split = MagicMock()
        mock_eval_split = MagicMock()
        mock_splits = {"train": mock_train_split, "test": mock_eval_split}
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=16)
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda x: 
            texts if x == "text" else labels if x == "label" else None
        )
        mock_dataset.train_test_split = MagicMock(return_value=mock_splits)
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            trainer.train(dataset=mock_dataset, auto_split=True)
            
            mock_dataset.train_test_split.assert_called_once_with(test_size=0.2, seed=42)
            call_kwargs = mocks['trainer_cls'].call_args[1]
            assert call_kwargs['train_dataset'] == mock_train_split
            assert call_kwargs['eval_dataset'] == mock_eval_split

    def test_auto_split_disabled(self):
        """train() with auto_split=False should not split even without eval_dataset."""
        mocks = self._create_trainer_with_mocks()
        
        texts = [f"text{i}" for i in range(16)]
        labels = ["class_a"] * 8 + ["class_b"] * 8
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=16)
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda x: 
            texts if x == "text" else labels if x == "label" else None
        )
        mock_dataset.train_test_split = MagicMock()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            trainer.train(dataset=mock_dataset, auto_split=False)
            
            mock_dataset.train_test_split.assert_not_called()
            call_kwargs = mocks['trainer_cls'].call_args[1]
            assert call_kwargs['train_dataset'] == mock_dataset
            assert call_kwargs['eval_dataset'] is None

    def test_auto_split_not_applied_when_eval_dataset_provided(self):
        """train() with explicit eval_dataset should not auto-split regardless of auto_split flag."""
        mocks = self._create_trainer_with_mocks()
        
        texts = [f"text{i}" for i in range(16)]
        labels = ["class_a"] * 8 + ["class_b"] * 8
        
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=16)
        mock_dataset.__getitem__ = MagicMock(side_effect=lambda x: 
            texts if x == "text" else labels if x == "label" else None
        )
        mock_dataset.train_test_split = MagicMock()
        
        mock_eval_dataset = MagicMock()
        
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mocks['setfit_model_cls'],
            'Trainer': mocks['trainer_cls'],
            'TrainingArguments': mocks['training_args_cls'],
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer()
            
            trainer.train(dataset=mock_dataset, eval_dataset=mock_eval_dataset, auto_split=True)
            
            mock_dataset.train_test_split.assert_not_called()
            call_kwargs = mocks['trainer_cls'].call_args[1]
            assert call_kwargs['train_dataset'] == mock_dataset
            assert call_kwargs['eval_dataset'] == mock_eval_dataset


# ============================================================================
# SetFitTrainer Integration Tests
# ============================================================================

class TestSetFitTrainerIntegration:
    """Integration tests for the full SetFit training workflow."""

    def test_trainer_to_backend_round_trip(self):
        """Test full workflow: train → save → load with from_finetuned."""
        import numpy as np
        
        # 1. Set up mocks for SetFitTrainer
        mock_setfit_model = MagicMock()
        mock_setfit_model_cls = MagicMock()
        mock_setfit_model_cls.from_pretrained.return_value = mock_setfit_model
        
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        
        mock_training_args_cls = MagicMock()
        
        # 2. Create and train SetFitTrainer (mocked)
        with patch.dict('taxonomy_framework.training.__dict__', {
            'SetFitModel': mock_setfit_model_cls,
            'Trainer': mock_trainer_cls,
            'TrainingArguments': mock_training_args_cls,
        }):
            from taxonomy_framework.training import SetFitTrainer
            trainer = SetFitTrainer(output_dir="/fake/model/path")
            
            # Train with mock dataset
            mock_dataset = _create_valid_mock_dataset()
            trainer.train(mock_dataset, auto_split=False)
            mock_trainer_instance.train.assert_called_once()
            
            # 3. Save the model (mocked)
            trainer.save()
            mock_setfit_model.save_pretrained.assert_called_once_with("/fake/model/path")
        
        # 4. Load with SetFitBackend.from_finetuned (mocked path validation)
        mock_loaded_model = MagicMock()
        mock_loaded_model.model_body.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_backend_setfit_cls = MagicMock()
        mock_backend_setfit_cls.from_pretrained.return_value = mock_loaded_model
        
        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': mock_backend_setfit_cls}):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.isfile', return_value=True):
                    from taxonomy_framework.embeddings import SetFitBackend
                    backend = SetFitBackend.from_finetuned("/fake/model/path")
                    
                    # 5. Verify the loaded backend can embed text
                    result = backend.embed_text("test text")
                    assert isinstance(result, np.ndarray)
                    mock_loaded_model.model_body.encode.assert_called_with("test text", normalize_embeddings=True)

    def test_trainer_integrates_with_embedder(self):
        """Test loaded backend works in EnsembleEmbedder."""
        import numpy as np
        from taxonomy_framework.embeddings import EnsembleEmbedder
        
        # Set up mock for loaded SetFit model
        mock_loaded_model = MagicMock()
        # Return different embeddings for different texts
        mock_loaded_model.model_body.encode.side_effect = lambda x, normalize_embeddings=False: (
            np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]]) if isinstance(x, list) 
            else np.array([0.8, 0.2])
        )
        
        mock_setfit_cls = MagicMock()
        mock_setfit_cls.from_pretrained.return_value = mock_loaded_model
        
        with patch.dict('taxonomy_framework.embeddings.__dict__', {'SetFitModel': mock_setfit_cls}):
            with patch('os.path.exists', return_value=True):
                with patch('os.path.isfile', return_value=True):
                    from taxonomy_framework.embeddings import SetFitBackend
                    backend = SetFitBackend.from_finetuned("/fake/model/path")
                    
                    # Create EnsembleEmbedder with the loaded backend
                    ensemble = EnsembleEmbedder([backend])
                    
                    # Test retrieve_candidates
                    candidates = ["similar text", "different text", "neutral text"]
                    result = ensemble.retrieve_candidates("query text", candidates, top_k=2)
                    
                    # Verify it returns indices
                    assert len(result) == 2
                    assert all(isinstance(idx, (int, np.integer)) for idx in result)

    def test_setfit_trainer_exported_from_package(self):
        """SetFitTrainer should be importable from taxonomy_framework."""
        from taxonomy_framework import SetFitTrainer
        assert SetFitTrainer is not None
        
        # Verify it has the expected interface
        assert hasattr(SetFitTrainer, 'train')
        assert hasattr(SetFitTrainer, 'save')
