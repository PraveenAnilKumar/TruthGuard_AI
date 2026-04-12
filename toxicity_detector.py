"""
Toxicity Detection Module for TruthGuard AI
Detects toxic content including hate speech, harassment, obscenity, and threats

OPTIMIZED: Module-level singleton uses a lazy proxy — the heavy __init__
(disk scan + pickle load) only runs on first attribute access, not at import.
"""

import warnings
import numpy as np
import pandas as pd
import re
import pickle
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import logging
from collections import Counter
from safe_transformers import run_isolated_text_classification
from translator_utils import content_translator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Check transformers/torch availability WITHOUT importing at module level.
# Importing transformers at module level hard-crashes Python 3.9 on some
# Windows setups (DLL conflict with torch/CUDA). Real imports happen inside methods.
import importlib.util as _ilu
TRANSFORMERS_AVAILABLE = (
    _ilu.find_spec("transformers") is not None and
    _ilu.find_spec("torch") is not None
)
del _ilu
if not TRANSFORMERS_AVAILABLE:
    logger.warning("Transformers/torch not available. Transformer models will be skipped.")

# For downloading datasets
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class ToxicityDetector:
    """
    Advanced Toxicity Detection System with multiple model support.
    """

    def __init__(self, use_ensemble=True, model_dir='models/toxicity/'):
        """
        Initialize Toxicity Detector.

        Args:
            use_ensemble: Whether to use ensemble of models.
            model_dir: Directory to store trained models.
        """
        self.use_ensemble = use_ensemble
        self.model_dir = model_dir
        self.models = {}
        self.vectorizer = None
        self.is_trained = False
        self.model_type = 'ensemble' if use_ensemble else 'single'
        self.threshold = 0.65

        # Toxicity categories
        self.categories = [
            'toxicity',
            'severe_toxicity',
            'obscene',
            'threat',
            'insult',
            'identity_attack',
        ]
        self.category_keyword_map = {
            'toxicity': ['hate', 'stupid', 'idiot', 'dumb', 'moron', 'loser', 'jerk', 'fool', 'pathetic', 'worthless', 'disgusting', 'garbage', 'trash'],
            'severe_toxicity': ['kill', 'die', 'hurt', 'destroy', 'attack', 'violent', 'terror', 'murder', 'assassinate', 'suicide', 'dead'],
            'obscene': ['fuck', 'shit', 'ass', 'damn', 'hell', 'piss', 'crap', 'bitch', 'whore', 'slut', 'dick', 'pussy', 'bastard'],
            'threat': ['kill', 'die', 'hurt', 'destroy', 'attack', 'threat', 'danger', 'warning', 'harm', 'assault', 'violence'],
            'insult': ['idiot', 'stupid', 'dumb', 'moron', 'loser', 'jerk', 'fool', 'pathetic', 'worthless', 'ignorant', 'ugly', 'useless', 'garbage'],
            'identity_attack': ['racist', 'sexist', 'homophobic', 'bigot', 'nazi', 'fascist', 'chauvinist', 'supremacist', 'transphobic', 'biased'],
        }
        self.protected_groups = [
            'women', 'men', 'girls', 'boys', 'gays', 'lesbians', 'muslims', 'hindus',
            'christians', 'jews', 'immigrants', 'refugees', 'asians', 'blacks',
            'whites', 'trans people', 'transgender people',
        ]
        self.benign_conversation_patterns = [
            r'\b(hi|hello|hey|good morning|good afternoon|good evening|good day)\b',
            r'\b(how are you|how are you doing|how r you|how is it going|how have you been)\b',
            r'\b(thank you|thanks|nice to meet you|pleased to meet you|hope you are well|hope you\'re well)\b',
        ]

        # Create model directory
        os.makedirs(model_dir, exist_ok=True)

        # Initialize transformer model if available
        self.transformer_model = None
        self.transformer_tokenizer = None
        self.transformer_model_ref = None
        self.transformer_tokenizer_ref = None
        self.use_transformer = False

        # Try to load existing model
        self._load_latest_model()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_latest_model(self):
        """Load the most recent trained model."""
        try:
            # Check for sklearn model
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
            if model_files:
                latest_model = max(
                    model_files,
                    key=lambda x: os.path.getctime(os.path.join(self.model_dir, x)),
                )
                model_path = os.path.join(self.model_dir, latest_model)

                with open(model_path, 'rb') as f:
                    with warnings.catch_warnings():
                        try:
                            from sklearn.exceptions import InconsistentVersionWarning
                            warnings.simplefilter("ignore", InconsistentVersionWarning)
                        except ImportError:
                            warnings.filterwarnings("ignore", module="sklearn")
                        data = pickle.load(f)

                if isinstance(data, dict):
                    self.models = data['model'] if 'model' in data else {'ensemble': data.get('ensemble', data)}
                    self.vectorizer = data.get('vectorizer')
                    self.is_trained = True
                    logger.info(f"Loaded model from {model_path}")
                    logger.info(f"   Available models: {list(self.models.keys())}")
                else:
                    self.models = {'ensemble': data}
                    self.is_trained = True
                    logger.info(f"Loaded legacy model from {model_path}")
                return

            transformer_path = os.path.join(self.model_dir, 'transformer_model')
            if os.path.exists(transformer_path) and TRANSFORMERS_AVAILABLE:
                self.transformer_model_ref = transformer_path
                self.transformer_tokenizer_ref = transformer_path
                self.use_transformer = True
                self.is_trained = True
                logger.info(f"Configured transformer model for isolated inference from {transformer_path}")
        except Exception as e:
            logger.error(f"Could not load existing model: {e}")

    # ── Text preprocessing ────────────────────────────────────────────────────
    def _preprocess_text(self, text):
        """Preprocess text for toxicity detection."""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s!?.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ── Pretrained transformer loader ─────────────────────────────────────────
    def load_pretrained_transformer(self, model_name="unitary/toxic-bert"):
        """Load a pretrained transformer model for toxicity detection."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available.")
            return False
        self.transformer_model_ref = model_name
        self.transformer_tokenizer_ref = model_name
        self.transformer_model = None
        self.transformer_tokenizer = None
        self.use_transformer = True
        self.is_trained = True
        logger.info(f"Configured pretrained transformer for isolated inference: {model_name}")
        return True

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, texts, labels, categories=None, test_size=0.2, epochs=3,
              use_transformer=False, save_model=True):
        """
        Train toxicity detection models.

        Args:
            texts: List of text samples.
            labels: List of labels (0=non-toxic, 1=toxic).
            categories: Unused (kept for API compatibility).
            test_size: Validation split size.
            epochs: Training epochs for transformer.
            use_transformer: Whether to use transformer model.
            save_model: Whether to save trained model.

        Returns:
            Training history dict.
        """
        if use_transformer and TRANSFORMERS_AVAILABLE:
            return self._train_transformer(texts, labels, epochs, save_model)
        return self._train_sklearn(texts, labels, test_size, save_model)

    def _train_sklearn(self, texts, labels, test_size=0.2, save_model=True):
        """Train sklearn-based models with adaptive handling for dataset size."""
        processed_texts = [self._preprocess_text(t) for t in texts]

        if len(processed_texts) < 10:
            logger.warning("Dataset too small (< 10 samples). Using fallback keyword detection.")
            self.is_trained = False
            return {"error": "Dataset too small. Need at least 10 samples."}

        n_samples = len(processed_texts)
        if n_samples < 50:
            min_df = 1
            max_features = min(500, n_samples * 10)
            logger.info(f"Small dataset ({n_samples} samples). Using adjusted parameters.")
        else:
            min_df = 2
            max_features = 5000

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, labels,
                test_size=min(test_size, 0.3),
                random_state=42,
                stratify=labels,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, labels,
                test_size=min(test_size, 0.3),
                random_state=42,
            )

        # Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=min_df,
            max_df=0.9,
        )
        try:
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            if X_train_vec.shape[1] == 0:
                raise ValueError("No features extracted.")
            logger.info(f"Vectorizer created {X_train_vec.shape[1]} features")
        except Exception as e:
            logger.warning(f"Vectorization error: {e} — falling back to simple vectorizer")
            self.vectorizer = TfidfVectorizer(
                max_features=100, ngram_range=(1, 1),
                stop_words='english', min_df=1, max_df=1.0,
            )
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            logger.info(f"Fallback vectorizer created {X_train_vec.shape[1]} features")

        trained_models = {}
        histories = {}

        # 1. Logistic Regression
        logger.info("Training logistic...")
        try:
            logistic = LogisticRegression(
                max_iter=1000, random_state=42, class_weight='balanced', C=1.0
            )
            logistic.fit(X_train_vec, y_train)
            trained_models['logistic'] = logistic
            y_pred = logistic.predict(X_test_vec)
            histories['logistic'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
            }
            logger.info(f"   logistic — Acc: {histories['logistic']['accuracy']:.4f}, F1: {histories['logistic']['f1']:.4f}")
        except Exception as e:
            logger.warning(f"   Error training logistic: {e}")

        # 2. Random Forest
        if n_samples > 30:
            logger.info("Training random_forest...")
            try:
                rf = RandomForestClassifier(
                    n_estimators=min(100, n_samples), random_state=42,
                    class_weight='balanced', max_depth=10,
                )
                rf.fit(X_train_vec, y_train)
                trained_models['random_forest'] = rf
                y_pred = rf.predict(X_test_vec)
                histories['random_forest'] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
                }
                logger.info(f"   random_forest — Acc: {histories['random_forest']['accuracy']:.4f}, F1: {histories['random_forest']['f1']:.4f}")
            except Exception as e:
                logger.warning(f"   Error training random_forest: {e}")

        # 3. SVM with calibration
        if n_samples > 30:
            logger.info("Training svm...")
            try:
                svm = LinearSVC(random_state=42, class_weight='balanced', dual=True, C=1.0, max_iter=2000)
                calibrated_svm = CalibratedClassifierCV(svm, cv=3)
                calibrated_svm.fit(X_train_vec, y_train)
                trained_models['svm'] = calibrated_svm
                y_pred = calibrated_svm.predict(X_test_vec)
                histories['svm'] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
                }
                logger.info(f"   svm — Acc: {histories['svm']['accuracy']:.4f}, F1: {histories['svm']['f1']:.4f}")
            except Exception as e:
                logger.warning(f"   Error training svm: {e}")

        # 4. Naive Bayes
        logger.info("Training naive_bayes...")
        try:
            nb = MultinomialNB()
            nb.fit(X_train_vec, y_train)
            trained_models['naive_bayes'] = nb
            y_pred = nb.predict(X_test_vec)
            histories['naive_bayes'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
            }
            logger.info(f"   naive_bayes — Acc: {histories['naive_bayes']['accuracy']:.4f}, F1: {histories['naive_bayes']['f1']:.4f}")
        except Exception as e:
            logger.warning(f"   Error training naive_bayes: {e}")

        # 5. Soft-voting ensemble
        if len(trained_models) >= 2:
            logger.info("   Creating ensemble...")
            try:
                ensemble = VotingClassifier(
                    estimators=list(trained_models.items()), voting='soft'
                )
                ensemble.fit(X_train_vec, y_train)
                trained_models['ensemble'] = ensemble
                y_pred_ensemble = ensemble.predict(X_test_vec)
                histories['ensemble'] = {
                    'accuracy': accuracy_score(y_test, y_pred_ensemble),
                    'precision': precision_score(y_test, y_pred_ensemble, average='binary', zero_division=0),
                    'recall': recall_score(y_test, y_pred_ensemble, average='binary', zero_division=0),
                    'f1': f1_score(y_test, y_pred_ensemble, average='binary', zero_division=0),
                }
                logger.info(f"   Ensemble — Acc: {histories['ensemble']['accuracy']:.4f}, F1: {histories['ensemble']['f1']:.4f}")
            except Exception as e:
                logger.warning(f"   Ensemble creation failed: {e}")
                if histories:
                    best = max(histories.items(), key=lambda x: x[1].get('f1', 0))[0]
                    trained_models['ensemble'] = trained_models[best]
                    histories['ensemble'] = histories[best]
        elif trained_models:
            trained_models['ensemble'] = list(trained_models.values())[0]
            histories['ensemble'] = list(histories.values())[0]

        self.models = trained_models
        self.is_trained = True

        if save_model:
            self._save_model(histories)

        return histories

    def _train_transformer(self, texts, labels, epochs=3, save_model=True):
        """Train transformer-based model."""
        if not TRANSFORMERS_AVAILABLE:
            return {"error": "Transformers not available"}
        try:
            import torch
            from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
            from torch.utils.data import Dataset

            class ToxicityDataset(Dataset):
                def __init__(self, texts, labels, tokenizer, max_length=128):
                    self.encodings = tokenizer(
                        texts, truncation=True, padding=True,
                        max_length=max_length, return_tensors='pt',
                    )
                    self.labels = torch.tensor(labels, dtype=torch.long)

                def __getitem__(self, idx):
                    item = {key: val[idx] for key, val in self.encodings.items()}
                    item['labels'] = self.labels[idx]
                    return item

                def __len__(self):
                    return len(self.labels)

            processed_texts = [self._preprocess_text(t) for t in texts]
            model_name = "bert-base-uncased"
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )

            split_idx = int(0.8 * len(processed_texts))
            train_dataset = ToxicityDataset(
                processed_texts[:split_idx], labels[:split_idx], self.transformer_tokenizer
            )
            val_dataset = ToxicityDataset(
                processed_texts[split_idx:], labels[split_idx:], self.transformer_tokenizer
            )

            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=epochs,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=64,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
            )

            trainer = Trainer(
                model=self.transformer_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self._compute_metrics,
            )
            trainer.train()
            eval_results = trainer.evaluate()

            self.use_transformer = True
            self.is_trained = True

            if save_model:
                save_path = os.path.join(self.model_dir, 'transformer_model')
                self.transformer_model.save_pretrained(save_path)
                self.transformer_tokenizer.save_pretrained(save_path)
                metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': 'transformer',
                    'model_name': model_name,
                    'epochs': epochs,
                    'eval_results': eval_results,
                }
                with open(os.path.join(self.model_dir, 'transformer_metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)

            return eval_results
        except Exception as e:
            return {"error": str(e)}

    def _compute_metrics(self, eval_pred):
        """Compute metrics for transformer evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='binary', zero_division=0),
            'recall': recall_score(labels, predictions, average='binary', zero_division=0),
            'f1': f1_score(labels, predictions, average='binary', zero_division=0),
        }

    def _save_model(self, histories):
        """Save trained model to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_data = {
            'model': self.models,
            'vectorizer': self.vectorizer,
            'histories': histories,
            'timestamp': timestamp,
            'model_type': self.model_type,
        }
        model_path = os.path.join(self.model_dir, f'toxicity_model_{timestamp}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        metadata = {
            'timestamp': timestamp,
            'model_type': self.model_type,
            'use_ensemble': self.use_ensemble,
            'histories': histories,
            'model_path': model_path,
        }
        metadata_path = os.path.join(self.model_dir, f'metadata_{timestamp}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Model saved to {model_path}")

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, text, return_proba=True):
        """
        Predict toxicity of text.

        Args:
            text: Input text.
            return_proba: Whether to return probability scores.

        Returns:
            Tuple of (is_toxic, confidence, category_scores, explanation, metadata).
        """
        translated_text, original_lang, was_translated = content_translator.translate_to_english(text)
        if isinstance(translated_text, (list, tuple, set)):
            translated_text = " ".join(str(part) for part in translated_text if part is not None)
        elif translated_text is None:
            translated_text = ""
        elif not isinstance(translated_text, str):
            translated_text = str(translated_text)
        meta = {
            'original_language': original_lang,
            'was_translated': was_translated,
            'processed_text': translated_text if was_translated else None,
        }

        if not self.is_trained:
            logger.warning("Toxicity model not trained/loaded. Falling back to keyword analysis.")
            is_toxic, toxic_prob, category_scores = self._fallback_predict(translated_text)
            processed = self._preprocess_text(translated_text)
            is_toxic, toxic_prob, category_scores, matched_words, context = self._apply_contextual_calibration(
                translated_text,
                processed,
                toxic_prob,
                category_scores,
            )
            explanation = self.get_explanation(
                translated_text,
                is_toxic,
                toxic_prob,
                category_scores,
                matched_words=matched_words,
                context=context,
            )
            meta.update({
                'analysis_context': context,
                'severity': explanation.get('severity', 'low'),
            })
            return is_toxic, toxic_prob, category_scores, explanation, meta

        if self.use_transformer and self.transformer_model is not None:
            is_toxic, toxic_prob, category_scores = self._predict_transformer(translated_text, return_proba)
        else:
            is_toxic, toxic_prob, category_scores = self._predict_sklearn(translated_text, return_proba)

        processed = self._preprocess_text(translated_text)
        is_toxic, toxic_prob, category_scores, matched_words, context = self._apply_contextual_calibration(
            translated_text,
            processed,
            toxic_prob,
            category_scores,
        )
        explanation = self.get_explanation(
            translated_text,
            is_toxic,
            toxic_prob,
            category_scores,
            matched_words=matched_words,
            context=context,
        )
        meta.update({
            'analysis_context': context,
            'severity': explanation.get('severity', 'low'),
        })
        return is_toxic, toxic_prob, category_scores, explanation, meta

    def _predict_sklearn(self, text, return_proba=True):
        """Predict using sklearn models."""
        processed = self._preprocess_text(text)

        if self.vectorizer is None:
            logger.warning("Vectorizer not found, using fallback")
            return self._fallback_predict(text)

        try:
            X = self.vectorizer.transform([processed])
        except Exception as e:
            logger.warning(f"Vectorization error: {e}")
            return self._fallback_predict(text)

        if 'ensemble' in self.models:
            model = self.models['ensemble']
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    toxic_prob = proba[1] if len(proba) > 1 else proba[0]
                else:
                    pred = model.predict(X)[0]
                    toxic_prob = 0.9 if pred == 1 else 0.1
            except Exception as e:
                logger.warning(f"Ensemble prediction error: {e}")
                toxic_probs, preds = [], []
                for name, m in self.models.items():
                    if name != 'ensemble':
                        try:
                            if hasattr(m, 'predict_proba'):
                                p = m.predict_proba(X)[0]
                                tp = p[1] if len(p) > 1 else p[0]
                                toxic_probs.append(tp)
                                preds.append(1 if tp > 0.5 else 0)
                            else:
                                pred = m.predict(X)[0]
                                preds.append(pred)
                                toxic_probs.append(0.9 if pred == 1 else 0.1)
                        except Exception:
                            pass
                if toxic_probs:
                    toxic_prob = float(np.mean(toxic_probs))
                else:
                    return self._fallback_predict(text)
        else:
            if not self.models:
                return self._fallback_predict(text)
            model_name = list(self.models.keys())[0]
            model = self.models[model_name]
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    toxic_prob = proba[1] if len(proba) > 1 else proba[0]
                else:
                    pred = model.predict(X)[0]
                    toxic_prob = 0.9 if pred == 1 else 0.1
            except Exception as e:
                logger.warning(f"Model prediction error: {e}")
                return self._fallback_predict(text)

        is_toxic = toxic_prob > self.threshold
        category_scores, _ = self._generate_category_scores(processed, toxic_prob, is_toxic)
        return is_toxic, toxic_prob, category_scores

    def _predict_transformer(self, text, return_proba=True):
        """Predict using transformer model."""
        processed = self._preprocess_text(text)
        try:
            if self.transformer_model_ref:
                result = run_isolated_text_classification(
                    self.transformer_model_ref,
                    processed,
                    tokenizer_ref=self.transformer_tokenizer_ref or self.transformer_model_ref,
                    local_files_only=os.path.isdir(self.transformer_model_ref),
                    max_length=128,
                )
                if not result.get("ok"):
                    raise RuntimeError(result.get("error", "Unknown isolated inference failure"))
                payload = result["result"]
                scores = payload.get("scores") or []
                if len(scores) >= 2:
                    toxic_prob = float(scores[1])
                elif scores:
                    raw_label = str(payload.get("label", "")).upper()
                    toxic_prob = float(scores[0])
                    if "LABEL_0" in raw_label or "NON" in raw_label:
                        toxic_prob = 1.0 - toxic_prob
                else:
                    toxic_prob = float(payload.get("score", 0.5))
            else:
                import torch
                inputs = self.transformer_tokenizer(
                    processed, return_tensors="pt", truncation=True, padding=True, max_length=128
                )
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    toxic_prob = probabilities[0][1].item()
        except Exception as e:
            logger.warning(f"Transformer prediction error: {e}")
            return self._fallback_predict(text)

        is_toxic = toxic_prob > self.threshold
        category_scores, _ = self._generate_category_scores(processed, toxic_prob, is_toxic)
        return is_toxic, toxic_prob, category_scores

    def _fallback_predict(self, text):
        """Simple keyword-based fallback prediction."""
        processed = self._preprocess_text(text)
        toxic_count = 0
        category_scores = {cat: 0.0 for cat in self.categories}
        
        for category, keywords in self.category_keyword_map.items():
            matches = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', processed)]
            if matches:
                category_scores[category] = min(0.4 + 0.1 * len(matches), 0.9)
                toxic_count += len(matches)
        
        toxic_prob = min(0.05 + 0.15 * toxic_count, 0.95)
        is_toxic = toxic_prob > self.threshold
        return is_toxic, toxic_prob, category_scores

    def _extract_context_features(self, original_text, processed_text):
        """Capture whether risky language is direct, quoted, de-escalating, or severe."""
        original_text = original_text or ""
        processed_text = processed_text or ""
        lowered_original = original_text.lower()
        token_count = len(re.findall(r"[a-z']+", processed_text))

        targeted = bool(re.search(r'\b(you|your|yourself|u|ur)\b', processed_text))
        deescalation = bool(re.search(
            r"\b(please|respectful|calm down|be civil|stay civil|avoid|do not|don't call|shouldn't|not okay|report|moderate|keep it respectful)\b",
            processed_text,
        ))
        quoted_context = bool(re.search(r"['\"][^'\"]{0,40}['\"]", lowered_original)) or bool(
            re.search(r'\b(the word|the phrase|quote|quoted|quoting|example|saying|called)\b', processed_text)
        )
        threat_pattern = bool(re.search(
            r"\b(i('| wi)?ll|i am going to|gonna|go and)\s+(kill|hurt|destroy|attack|harm|beat)\b",
            processed_text,
        )) or bool(re.search(r'\b(go kill yourself|you deserve to die)\b', processed_text))
        identity_targeted = 'you people' in processed_text
        if not identity_targeted:
            for group in self.protected_groups:
                if re.search(r'\b(all|those|these)\s+' + re.escape(group) + r'\b', processed_text):
                    identity_targeted = True
                    break

        obscene_count = len(re.findall(
            r'\b(fuck|shit|bitch|bastard|whore|slut|dick|pussy|asshole)\b',
            processed_text,
        ))
        benign_conversational = token_count <= 12 and any(
            re.search(pattern, processed_text) for pattern in self.benign_conversation_patterns
        )
        polite_conversational = bool(re.search(r"\b(please|thanks|thank you|kindly)\b", processed_text))

        attenuation = 1.0
        if quoted_context and not targeted and not threat_pattern and not identity_targeted:
            attenuation *= 0.55
        if deescalation and not threat_pattern and not identity_targeted:
            attenuation *= 0.75

        return {
            'targeted': targeted,
            'deescalation': deescalation,
            'quoted_context': quoted_context,
            'threat_pattern': threat_pattern,
            'identity_targeted': identity_targeted,
            'obscene_count': obscene_count,
            'token_count': token_count,
            'benign_conversational': benign_conversational,
            'polite_conversational': polite_conversational,
            'attenuation': float(np.clip(attenuation, 0.35, 1.0)),
        }

    def _generate_category_scores(self, processed_text, toxic_prob, is_toxic, context=None):
        """Generate per-category scores using keywords plus contextual calibration."""
        context = context or self._extract_context_features(processed_text, processed_text)
        category_results = {}
        matched_words_all = {}

        for category, keywords in self.category_keyword_map.items():
            matches = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', processed_text)]
            matched_words_all[category] = matches

            if matches:
                lexical_score = min(0.18 * len(matches), 0.72)
            else:
                lexical_score = 0.0

            if category == 'threat' and context['threat_pattern']:
                lexical_score = max(lexical_score, 0.84)
            if category == 'severe_toxicity' and context['threat_pattern']:
                lexical_score = max(lexical_score, 0.76)
            if category == 'identity_attack' and context['identity_targeted']:
                lexical_score = max(lexical_score, 0.74)
            if category == 'insult' and context['targeted'] and matches:
                lexical_score += 0.16
            if category == 'toxicity' and context['targeted'] and matches:
                lexical_score += 0.10
            if category == 'obscene' and context['obscene_count']:
                lexical_score += min(context['obscene_count'] * 0.06, 0.18)

            if matches:
                base_model_component = toxic_prob * (0.42 if is_toxic else 0.18)
                score = lexical_score + base_model_component
            else:
                score = toxic_prob * (0.22 if is_toxic else 0.08)

            if context['attenuation'] < 1.0 and category not in {'threat', 'severe_toxicity'}:
                score *= context['attenuation']

            category_results[category] = float(np.clip(score, 0.0, 1.0))

        return category_results, matched_words_all

    def _apply_contextual_calibration(self, original_text, processed_text, toxic_prob, category_scores):
        """Adjust model output based on directionality, quoting, and severe intent."""
        context = self._extract_context_features(original_text, processed_text)
        lexical_peak = max(category_scores.values()) if category_scores else 0.0
        calibrated_prob = max(float(toxic_prob), lexical_peak * 0.92)
        low_signal = lexical_peak < 0.12

        if context['threat_pattern']:
            calibrated_prob = max(calibrated_prob, 0.88)
        if context['identity_targeted']:
            calibrated_prob = max(calibrated_prob, 0.76)
        if context['targeted'] and (
            category_scores.get('insult', 0) > 0.24 or category_scores.get('toxicity', 0) > 0.24
        ):
            calibrated_prob = max(
                calibrated_prob,
                min(0.56 + (category_scores.get('insult', 0) * 0.35), 0.86),
            )
        if context['deescalation'] and not context['targeted'] and not context['threat_pattern'] and not context['identity_targeted']:
            calibrated_prob = min(calibrated_prob * 0.58, max(self.threshold * 0.82, 0.38))
        if low_signal and context['benign_conversational'] and not context['threat_pattern'] and not context['identity_targeted']:
            calibrated_prob = min(calibrated_prob * 0.34, 0.24)
        elif low_signal and context['polite_conversational'] and context['token_count'] <= 12:
            calibrated_prob = min(calibrated_prob * 0.62, max(self.threshold * 0.7, 0.32))

        if context['attenuation'] < 1.0 and not context['threat_pattern'] and not context['identity_targeted']:
            calibrated_prob *= context['attenuation']

        calibrated_prob = float(np.clip(calibrated_prob, 0.02, 0.99))
        is_toxic = calibrated_prob > self.threshold
        calibrated_categories, matched_words = self._generate_category_scores(
            processed_text,
            calibrated_prob,
            is_toxic,
            context=context,
        )
        return is_toxic, calibrated_prob, calibrated_categories, matched_words, context

    def get_explanation(self, text, is_toxic, confidence, categories, matched_words=None, context=None):
        """Generate a natural language explanation and word importance."""
        processed = self._preprocess_text(text)
        context = context or self._extract_context_features(text, processed)
        if matched_words is None:
            _, matched_words = self._generate_category_scores(processed, confidence, is_toxic, context=context)

        word_impact = {}
        all_triggered_words = set()

        for cat, words in matched_words.items():
            for word in words:
                all_triggered_words.add(word)
                if word not in word_impact:
                    word_impact[word] = {'categories': [cat], 'score': categories.get(cat, 0)}
                else:
                    word_impact[word]['categories'].append(cat)
                    word_impact[word]['score'] = max(word_impact[word]['score'], categories.get(cat, 0))

        reasons = []
        if is_toxic:
            if categories.get('threat', 0) > 0.5 or context.get('threat_pattern'):
                reasons.append("The message contains language that appears to be threatening or violent.")
            if categories.get('identity_attack', 0) > 0.5:
                reasons.append("The message contains potential attacks on identity (race, religion, gender, etc.).")
            if categories.get('obscene', 0) > 0.5:
                reasons.append("The message uses offensive or obscene language.")
            if context.get('targeted') and (
                categories.get('insult', 0) > 0.35 or categories.get('toxicity', 0) > 0.5
            ):
                reasons.append("The message directly targets a person with abusive or derogatory language.")
            if not reasons:
                reasons.append("The message has a high probability of being toxic based on its overall tone.")
        else:
            if context.get('benign_conversational'):
                reasons.append("The message looks like a simple greeting or conversational check-in, so it was treated as low risk.")
            elif context.get('quoted_context') and all_triggered_words:
                reasons.append("Potentially harmful terms were mentioned in a quoted or cautionary context, so the score was reduced.")
            elif context.get('deescalation') and max(categories.values()) > 0.2:
                reasons.append("The message contains risky language, but the overall wording is corrective or de-escalating.")
            else:
                reasons.append("The message appears to be safe and non-toxic.")

        peak = max(categories.values()) if categories else 0.0
        if confidence >= 0.85 or categories.get('threat', 0) > 0.75:
            severity = 'critical'
        elif confidence >= 0.65 or peak > 0.6:
            severity = 'high'
        elif confidence >= 0.4 or peak > 0.3:
            severity = 'moderate'
        else:
            severity = 'low'

        explanation = {
            'is_toxic': is_toxic,
            'confidence': confidence,
            'severity': severity,
            'reasons': reasons,
            'word_impact': word_impact,
            'categories': categories,
        }
        return explanation

    def batch_predict(self, texts):
        """Perform batch prediction for a list of texts."""
        results = []
        for text in texts:
            is_toxic, confidence, categories, explanation, meta = self.predict(text)
            results.append({
                'text': text,
                'is_toxic': is_toxic,
                'confidence_score': confidence,
                'severity': explanation['severity'],
                'top_category': max(categories, key=categories.get) if categories else None,
                'top_score': max(categories.values()) if categories else 0.0,
            })
        return pd.DataFrame(results)

    def download_dataset(self, dataset_name='jigsaw-toxic-comment'):
        """Download or create a toxicity dataset."""
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests not available.")
            return None
        try:
            if dataset_name == 'jigsaw-toxic-comment':
                logger.info("Creating sample dataset...")
                return self._create_sample_dataset()
            logger.warning(f"Dataset {dataset_name} not supported")
            return None
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return None

    def _create_sample_dataset(self, n_samples=1000):
        """Create a sample dataset for testing."""
        np.random.seed(42)
        non_toxic = [
            "I really enjoyed this movie, it was great!", "The weather today is beautiful.",
            "Thank you for your help!", "This product works perfectly.",
            "I agree with your point of view.", "Have a wonderful day!",
            "The service was excellent.", "This is a very interesting article.",
            "I appreciate your feedback.", "Looking forward to the event.",
        ]
        toxic = [
            "You are such an idiot!", "I hate you and everyone like you.",
            "Go kill yourself, you're worthless.", "This is complete garbage, you suck!",
            "Shut up you stupid moron.", "You're a pathetic loser.",
            "I'll destroy you and your family.", "This is so bad, it's disgusting.",
            "You people are all the same, go away.", "What a waste of space, you're useless.",
        ]
        texts, labels = [], []
        for _ in range(n_samples // 2):
            t = np.random.choice(non_toxic)
            if np.random.random() > 0.3:
                t += np.random.choice(["", "!", ".", "?"])
            texts.append(t); labels.append(0)
        for _ in range(n_samples // 2):
            texts.append(np.random.choice(toxic)); labels.append(1)
        indices = np.random.permutation(len(texts))
        return pd.DataFrame({'text': [texts[i] for i in indices], 'toxic': [labels[i] for i in indices]})

    def get_model_info(self):
        """Get information about loaded models."""
        info = {
            'is_trained': self.is_trained,
            'model_type': 'transformer' if self.use_transformer else 'sklearn',
            'use_ensemble': self.use_ensemble,
            'threshold': self.threshold,
            'num_models': len(self.models) if not self.use_transformer else 1,
            'categories': self.categories,
        }
        if self.use_transformer and (self.transformer_model is not None or self.transformer_model_ref is not None):
            info['transformer_name'] = (
                getattr(getattr(self.transformer_model, "config", None), "_name_or_path", None)
                or self.transformer_model_ref
            )
        elif self.models:
            info['model_names'] = list(self.models.keys())
        return info

    def visualize_toxicity(self, text):
        """Create visualization for toxicity analysis."""
        is_toxic, confidence, categories, explanation, _meta = self.predict(text)
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            title={'text': "Toxicity Score"},
            domain={'x': [0, 0.5], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if is_toxic else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 65], 'color': "yellow"},
                    {'range': [65, 100], 'color': "salmon"},
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': self.threshold * 100},
            },
        ))
        categories_list = list(categories.keys())
        scores = list(categories.values())
        fig.add_trace(go.Bar(
            x=scores, y=categories_list, orientation='h',
            marker=dict(color=scores, colorscale='RdYlGn_r', cmin=0, cmax=1),
            text=[f"{s:.1%}" for s in scores], textposition='auto',
            domain={'x': [0.6, 1], 'y': [0, 1]},
        ))
        fig.update_layout(
            height=400, title="Toxicity Analysis", showlegend=False,
            xaxis2={'title': 'Score', 'range': [0, 1]},
            grid={'rows': 1, 'columns': 2, 'pattern': 'independent'},
        )
        return fig

    def create_distribution_chart(self, results_df):
        """Create distribution chart for batch results."""
        toxic_count = len(results_df[results_df['is_toxic'] == True])
        non_toxic_count = len(results_df[results_df['is_toxic'] == False])
        fig = go.Figure(data=[go.Pie(
            labels=['Toxic', 'Non-Toxic'], values=[toxic_count, non_toxic_count],
            marker=dict(colors=['red', 'green']), hole=0.3, domain={'x': [0, 0.5]},
        )])
        confidence_scores = results_df[results_df['confidence_score'].notna()]['confidence_score']
        if len(confidence_scores) > 0:
            fig.add_trace(go.Histogram(
                x=confidence_scores, nbinsx=20, marker_color='blue', opacity=0.7,
                name='Confidence Distribution', domain={'x': [0.6, 1], 'y': [0, 1]},
            ))
        fig.update_layout(
            title="Toxicity Distribution",
            xaxis2={'title': 'Confidence Score', 'range': [0, 1]},
            yaxis2={'title': 'Count'}, height=400, showlegend=False,
        )
        return fig


# ── Lazy proxy singleton ──────────────────────────────────────────────────────
# Importing this module is now instant. ToxicityDetector.__init__ (which
# scans the filesystem and loads pickle files) only runs on first attribute
# access — i.e., when the detector is first used after login.
class _LazyToxicityDetector:
    """Proxy that defers ToxicityDetector construction until first use."""
    _instance = None

    def _get(self):
        if self._instance is None:
            object.__setattr__(self, '_instance', ToxicityDetector(use_ensemble=True))
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def __setattr__(self, name, value):
        if name == '_instance':
            object.__setattr__(self, name, value)
        else:
            setattr(self._get(), name, value)


toxicity_detector = _LazyToxicityDetector()
