"""
Train Fake News Detection Models
Enhanced version with better model saving, multiple architectures, and comprehensive evaluation
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score, roc_curve)
import os
import sys
import logging
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fake_news_detector import FakeNewsDetector

# Try importing transformers for enhanced capabilities
try:
    from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FakeNewsDataset(Dataset):
    """Custom Dataset for transformer fine-tuning"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
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
    """Load and prepare dataset with enhanced column detection"""
    logger.info(f"Loading dataset from {filepath}")
    
    # Check file extension
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    elif filepath.endswith('.json'):
        df = pd.read_json(filepath)
    else:
        logger.error(f"Unsupported file format: {filepath}")
        return None, None, None
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Try to identify text and label columns
    text_col = None
    label_col = None
    title_col = None
    
    # Common column names
    text_names = ['text', 'content', 'article', 'news', 'statement', 'body', 'description']
    title_names = ['title', 'headline', 'heading', 'subject']
    label_names = ['label', 'class', 'target', 'fake', 'real', 'veracity', 'truth', 'category']
    
    for col in df.columns:
        col_lower = col.lower()
        if not text_col and any(name in col_lower for name in text_names):
            text_col = col
        if not title_col and any(name in col_lower for name in title_names):
            title_col = col
        if not label_col and any(name in col_lower for name in label_names):
            label_col = col
    
    if not text_col and title_col:
        # If no text column but title exists, combine title and any other text
        logger.info("No text column found, using title as text")
        text_col = title_col
    
    if not text_col:
        # Assume first column is text
        text_col = df.columns[0]
        logger.warning(f"No text column identified, using first column: {text_col}")
    
    if not label_col:
        # Assume last column is label
        label_col = df.columns[-1]
        logger.warning(f"No label column identified, using last column: {label_col}")
    
    logger.info(f"Using text column: {text_col}")
    logger.info(f"Using label column: {label_col}")
    if title_col:
        logger.info(f"Using title column: {title_col}")
    
    # Combine title and text if both exist
    if title_col and text_col != title_col:
        texts = []
        for _, row in df.iterrows():
            title = str(row[title_col]) if pd.notna(row[title_col]) else ""
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            combined = f"{title} {text}".strip()
            texts.append(combined if combined else "empty")
    else:
        texts = df[text_col].astype(str).tolist()
    
    labels = df[label_col].tolist()
    
    # Handle missing values
    texts = [t if t and t != 'nan' else 'empty' for t in texts]
    labels = [l for l in labels if pd.notna(l)]
    
    # Convert labels to binary (0=real, 1=fake)
    unique_labels = sorted(set(labels))
    logger.info(f"Unique labels in dataset: {unique_labels}")
    
    if len(unique_labels) == 2:
        # Binary classification
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        labels = [label_map[l] for l in labels]
        logger.info(f"Binary classification: {unique_labels[0]} -> 0 (real), {unique_labels[1]} -> 1 (fake)")
    else:
        # Multi-class or need mapping
        logger.warning(f"More than 2 labels found: {unique_labels}")
        # Create mapping based on common fake/real indicators
        label_map = {}
        for label in unique_labels:
            label_lower = str(label).lower()
            if any(word in label_lower for word in ['fake', 'false', '1', 'true']):
                label_map[label] = 1
            else:
                label_map[label] = 0
        labels = [label_map[l] for l in labels]
        logger.info(f"Mapped labels: {label_map}")
    
    # Check class distribution
    class_dist = pd.Series(labels).value_counts().to_dict()
    logger.info(f"Class distribution: {class_dist}")
    
    return texts, labels, (text_col, label_col, title_col)

def plot_training_history(history, save_path):
    """Plot training history for transformer models"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
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

def plot_confusion_matrix(cm, labels, save_path):
    """Plot confusion matrix"""
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

def plot_roc_curve(y_true, y_pred_proba, save_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"ROC curve saved to {save_path}")

def train_transformer_model(X_train, y_train, X_test, y_test, args):
    """Train a transformer-based model using Hugging Face"""
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers not available. Cannot train transformer model.")
        return None, None, None, None
    
    logger.info("Training transformer model...")
    
    # Model name based on selection
    model_name = args.transformer_model if hasattr(args, 'transformer_model') else 'distilbert-base-uncased'
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2  # Binary classification
    )
    
    # Create datasets
    train_dataset = FakeNewsDataset(X_train, y_train, tokenizer)
    test_dataset = FakeNewsDataset(X_test, y_test, tokenizer)
    
    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/fake_news/transformer_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs if hasattr(args, 'epochs') else 3,
        per_device_train_batch_size=args.batch_size if hasattr(args, 'batch_size') else 16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda eval_pred: {
            'accuracy': accuracy_score(eval_pred.label_ids, np.argmax(eval_pred.predictions, axis=1))
        },
    )
    
    # Train
    logger.info("Starting transformer training...")
    train_result = trainer.train()
    
    # Evaluate
    logger.info("Evaluating transformer model...")
    eval_results = trainer.evaluate()
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_pred_proba = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions), dim=-1)[:, 1].numpy()
    
    # Save the model
    model_save_path = os.path.join(output_dir, 'final_model')
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Transformer model saved to {model_save_path}")
    
    # Save training history
    history = {
        'train_loss': train_result.training_loss,
        'eval_loss': eval_results.get('eval_loss', 0),
        'eval_accuracy': eval_results.get('eval_accuracy', 0),
        'epochs': args.epochs if hasattr(args, 'epochs') else 3,
    }
    
    return y_pred, y_pred_proba, history, output_dir

def evaluate_model(model, X_test, y_test, model_name="model", save_results=True):
    """Evaluate model performance with comprehensive metrics"""
    predictions = []
    confidences = []
    
    for text in X_test:
        try:
            label, conf, click, meta = model.predict(text)
            # Convert to binary
            pred = 1 if label == 'FAKE' else 0
            predictions.append(pred)
            confidences.append(conf if pred == 1 else 1 - conf)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            predictions.append(0)  # Default to real
            confidences.append(0.5)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='binary', zero_division=0)
    recall = recall_score(y_test, predictions, average='binary', zero_division=0)
    f1 = f1_score(y_test, predictions, average='binary', zero_division=0)
    cm = confusion_matrix(y_test, predictions)
    
    # Calculate ROC AUC if we have confidence scores
    try:
        auc = roc_auc_score(y_test, confidences)
    except:
        auc = 0.0
    
    logger.info(f"\n=== {model_name} Evaluation ===")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"AUC-ROC:   {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Classification report
    report = classification_report(y_test, predictions, 
                                  target_names=['Real', 'Fake'], 
                                  output_dict=True)
    logger.info(f"\nClassification Report:\n{classification_report(y_test, predictions, target_names=['Real', 'Fake'])}")
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"models/fake_news/evaluation_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist(),
            'report': report,
            'timestamp': timestamp,
            'model_name': model_name
        }
        
        with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, ['Real', 'Fake'], 
                            os.path.join(results_dir, 'confusion_matrix.png'))
        
        # Plot ROC curve
        plot_roc_curve(y_test, confidences, 
                       os.path.join(results_dir, 'roc_curve.png'))
        
        logger.info(f"Evaluation results saved to {results_dir}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'predictions': predictions,
        'confidences': confidences
    }

def main():
    parser = argparse.ArgumentParser(description='Train fake news detection models')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV file')
    parser.add_argument('--model-type', type=str, default='logistic', 
                       choices=['logistic', 'random_forest', 'transformer'],
                       help='Type of model to train')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--save-model', action='store_true', default=True, help='Save trained model')
    parser.add_argument('--model-name', type=str, default=None, help='Custom name for saved model')
    
    # Transformer-specific arguments
    parser.add_argument('--transformer-model', type=str, default='distilbert-base-uncased',
                       help='Transformer model name or path')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs for transformer')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for transformer training')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate for transformer')
    
    # Advanced options
    parser.add_argument('--use-cv', action='store_true', help='Use cross-validation')
    parser.add_argument('--grid-search', action='store_true', help='Perform grid search for hyperparameters')
    
    args = parser.parse_args()
    
    # Load dataset
    texts, labels, columns = load_dataset(args.dataset)
    if texts is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, 
        random_state=args.random_state, 
        stratify=labels if len(set(labels)) > 1 else None
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train model based on type
    if args.model_type == 'transformer' and TRANSFORMERS_AVAILABLE:
        # Train transformer model
        y_pred, y_pred_proba, history, model_dir = train_transformer_model(
            X_train, y_train, X_test, y_test, args
        )
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"\n=== Transformer Model Evaluation ===")
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Plot training history
        if history:
            plot_path = os.path.join(model_dir, 'training_history.png')
            plot_training_history(history, plot_path)
        
        # Save metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'history': history,
            'model_type': 'transformer',
            'model_name': args.transformer_model
        }
        
        with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
    else:
        # Initialize and train traditional model
        model = FakeNewsDetector(use_transformer=False)
        
        logger.info(f"Training {args.model_type} model...")
        
        if args.use_cv:
            # Perform cross-validation
            logger.info(f"Performing {args.cv_folds}-fold cross-validation...")
            skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_train_fold = [X_train[i] for i in train_idx]
                y_train_fold = [y_train[i] for i in train_idx]
                X_val_fold = [X_train[i] for i in val_idx]
                y_val_fold = [y_train[i] for i in val_idx]
                
                # Train on fold
                fold_model = FakeNewsDetector(use_transformer=False)
                fold_model.train(X_train_fold, y_train_fold, model_type=args.model_type)
                
                # Evaluate on validation
                val_metrics = evaluate_model(fold_model, X_val_fold, y_val_fold, 
                                           save_results=False)
                cv_scores.append(val_metrics['accuracy'])
                logger.info(f"Fold {fold+1} accuracy: {val_metrics['accuracy']:.4f}")
            
            logger.info(f"Cross-validation accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        # Train final model on full training set
        model.train(X_train, y_train, model_type=args.model_type)
        
        # Evaluate
        logger.info("Evaluating model on test set...")
        metrics = evaluate_model(model, X_test, y_test, 
                               model_name=args.model_type, 
                               save_results=True)
        
        # Save model
        if args.save_model:
            model_path = 'models/fake_news/'
            os.makedirs(model_path, exist_ok=True)
            
            if args.model_name:
                model_filename = os.path.join(model_path, f"{args.model_name}.pkl")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = os.path.join(model_path, f"{args.model_type}_{timestamp}.pkl")
            
            # Save model
            joblib.dump(model, model_filename)
            logger.info(f"Model saved to {model_filename}")
            
            # Also save using model's internal save method
            model._save_model()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()