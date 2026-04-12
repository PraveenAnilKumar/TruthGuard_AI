"""
Transformer-based Fake News Detection Training Script - CPU Optimized
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
import os
import sys
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FakeNewsDataset(Dataset):
    """Custom Dataset for transformer fine-tuning"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_dataset(filepath):
    """Load and prepare dataset"""
    logger.info(f"Loading dataset from {filepath}")
    
    df = pd.read_csv(filepath)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Try to identify text and label columns
    text_col = None
    label_col = None
    
    # Common column names
    text_names = ['text', 'content', 'article', 'news', 'statement', 'body', 'title']
    label_names = ['label', 'class', 'target', 'fake', 'real', 'category']
    
    for col in df.columns:
        col_lower = col.lower()
        if not text_col and any(name in col_lower for name in text_names):
            text_col = col
        if not label_col and any(name in col_lower for name in label_names):
            label_col = col
    
    if not text_col:
        text_col = df.columns[0]
        logger.warning(f"No text column identified, using first column: {text_col}")
    
    if not label_col:
        label_col = df.columns[-1]
        logger.warning(f"No label column identified, using last column: {label_col}")
    
    logger.info(f"Using text column: {text_col}")
    logger.info(f"Using label column: {label_col}")
    
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].tolist()
    
    # Convert labels to binary (0=real, 1=fake)
    unique_labels = sorted(set(labels))
    logger.info(f"Unique labels: {unique_labels}")
    
    if len(unique_labels) == 2:
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        labels = [label_map[l] for l in labels]
    else:
        # Try to infer based on common fake/real indicators
        label_map = {}
        for label in unique_labels:
            label_str = str(label).lower()
            if any(word in label_str for word in ['fake', 'false', 'spam', '1']):
                label_map[label] = 1
            else:
                label_map[label] = 0
        labels = [label_map[l] for l in labels]
    
    # Check class distribution
    class_dist = pd.Series(labels).value_counts().to_dict()
    logger.info(f"Class distribution (0=real, 1=fake): {class_dist}")
    
    return texts, labels

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary', zero_division=0)
    recall = recall_score(labels, predictions, average='binary', zero_division=0)
    f1 = f1_score(labels, predictions, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history.get('train_loss', []), label='Train Loss')
    if 'eval_loss' in history:
        axes[0].plot(history['eval_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    if 'eval_accuracy' in history:
        axes[1].plot(history['eval_accuracy'], label='Val Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train transformer-based fake news detection')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV file')
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased',
                       choices=['distilbert-base-uncased', 'bert-base-uncased', 'roberta-base'],
                       help='Transformer model to use')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_short = args.model_name.split('/')[-1]
    output_dir = f"models/fake_news/transformer_{model_name_short}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    texts, labels = load_dataset(args.dataset)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels if len(set(labels)) > 1 else None
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2  # Binary classification
    )
    
    # CPU training
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer, args.max_length)
    test_dataset = FakeNewsDataset(X_test, y_test, tokenizer, args.max_length)
    
    # Calculate training steps
    train_steps_per_epoch = len(train_dataset) // args.batch_size
    total_train_steps = train_steps_per_epoch * args.epochs
    
    # Training arguments - FIXED for latest transformers version
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_steps=int(0.1 * total_train_steps),
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,  # Log more frequently for CPU
        eval_strategy="epoch",  # FIXED: was 'evaluation_strategy'
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=False,  # CPU doesn't support fp16
        dataloader_num_workers=2,
        learning_rate=args.learning_rate,
        report_to="none",
        disable_tqdm=False,  # Show progress bar
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  # Reduced patience for CPU
    )
    
    # Train the model
    logger.info("Starting training on CPU (this will take a while)...")
    logger.info(f"Estimated time: {args.epochs * 60} minutes")
    train_result = trainer.train()
    
    # Evaluate
    logger.info("Evaluating on test set...")
    eval_results = trainer.evaluate()
    
    # Get predictions for test set
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    # Calculate detailed metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    logger.info(f"\n=== Final Evaluation Results ===")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    
    # Print classification report
    logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=['Real', 'Fake'])}")
    
    # Save the model
    model_save_path = os.path.join(output_dir, 'final_model')
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    # Save training history
    history = {
        'train_loss': [log['loss'] for log in trainer.state.log_history if 'loss' in log],
        'eval_accuracy': [log['eval_accuracy'] for log in trainer.state.log_history if 'eval_accuracy' in log],
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'model_name': args.model_name,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'eval_results': eval_results,
        'history': history,
        'timestamp': timestamp
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, ['Real', 'Fake'], 
                         os.path.join(output_dir, 'confusion_matrix.png'))
    
    logger.info(f"All results saved to {output_dir}")
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()