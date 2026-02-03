"""SetFit training module for taxonomy classification fine-tuning."""

from collections import Counter
from typing import Optional, Any, List
from .utils import logger

try:
    from setfit import SetFitModel, Trainer, TrainingArguments
except ImportError:
    SetFitModel = None
    Trainer = None
    TrainingArguments = None

try:
    from datasets import Dataset
except ImportError:
    Dataset = None


class SetFitTrainer:
    """
    Wrapper for SetFit's Trainer to fine-tune classification models.
    
    SetFit uses efficient few-shot learning with contrastive training
    followed by classification head training.
    
    Example:
        trainer = SetFitTrainer(base_model="sentence-transformers/paraphrase-mpnet-base-v2")
        trainer.train(dataset, text_column="text", label_column="label", num_epochs=4)
        trainer.save()
    """
    
    def __init__(
        self,
        base_model: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        output_dir: str = "./models/setfit-model"
    ):
        """
        Initialize SetFitTrainer.
        
        Args:
            base_model: Pre-trained sentence transformer model name or path.
            output_dir: Directory to save the fine-tuned model.
            
        Raises:
            ImportError: If setfit library is not installed.
        """
        if SetFitModel is None or Trainer is None:
            raise ImportError("setfit not installed. Run: pip install setfit")
        
        self.base_model = base_model
        self.output_dir = output_dir
        
        logger.info(f"Loading base model: {base_model}")
        self.model = SetFitModel.from_pretrained(base_model)
        logger.info("Base model loaded successfully")
    
    def _validate_dataset(
        self,
        dataset: Any,
        text_column: str,
        label_column: str
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty")
        
        texts = dataset[text_column]
        labels = dataset[label_column]
        
        for idx, text in enumerate(texts):
            if text is None or (isinstance(text, str) and not text.strip()):
                raise ValueError(f"Found null/empty text at index {idx}")
        
        label_counts = Counter(labels)
        
        if len(label_counts) < 2:
            raise ValueError("Need at least 2 classes for classification")
        
        for label, count in label_counts.items():
            if count < 8:
                raise ValueError(f"Class '{label}' has only {count} samples. Minimum 8 required.")
        
        logger.info("Class distribution:")
        for label, count in sorted(label_counts.items()):
            logger.info(f"  {label}: {count}")
    
    def _prepare_dataset(
        self,
        dataset: Optional[Any],
        texts: Optional[List[str]],
        labels: Optional[List[str]]
    ) -> Any:
        if texts is not None or labels is not None:
            if texts is None or labels is None:
                raise ValueError("Both 'texts' and 'labels' must be provided together")
            if Dataset is None:
                raise ImportError("datasets not installed. Run: pip install datasets")
            return Dataset.from_dict({"text": texts, "label": labels})
        
        if dataset is None:
            raise ValueError("Either 'dataset' or both 'texts' and 'labels' must be provided")
        
        return dataset
    
    def train(
        self,
        dataset: Optional[Any] = None,
        texts: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        text_column: str = "text",
        label_column: str = "label",
        eval_dataset: Optional[Any] = None,
        auto_split: bool = True,
        **training_args
    ) -> None:
        """
        Train the SetFit model on the provided dataset.
        
        Args:
            dataset: Training dataset (HuggingFace Dataset format).
            texts: List of text samples (alternative to dataset).
            labels: List of labels corresponding to texts.
            text_column: Name of the column containing text data.
            label_column: Name of the column containing labels.
            eval_dataset: Optional evaluation dataset.
            auto_split: If True and eval_dataset is None, auto-split 80/20.
            **training_args: Additional arguments passed to TrainingArguments
                (e.g., batch_size=16, num_epochs=4).
        """
        prepared_dataset = self._prepare_dataset(dataset, texts, labels)
        
        self._validate_dataset(prepared_dataset, text_column, label_column)
        
        train_dataset = prepared_dataset
        if auto_split and eval_dataset is None:
            logger.info("Auto-splitting dataset 80/20 for train/eval")
            splits = prepared_dataset.train_test_split(test_size=0.2, seed=42)
            train_dataset = splits["train"]
            eval_dataset = splits["test"]
        
        logger.info("Preparing training arguments")
        args = TrainingArguments(**training_args)
        
        logger.info("Creating SetFit Trainer")
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            column_mapping={text_column: "text", label_column: "label"}
        )
        
        logger.info("Starting training")
        trainer.train()
        logger.info("Training completed")
    
    def save(self) -> None:
        """
        Save the fine-tuned model to output_dir.
        
        Uses SetFitModel.save_pretrained() for full model serialization.
        """
        logger.info(f"Saving model to: {self.output_dir}")
        self.model.save_pretrained(self.output_dir)
        logger.info("Model saved successfully")
