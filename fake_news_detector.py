"""
Fake News Detection Module
Supports both traditional ML and transformer-based approaches

OPTIMIZED: NLTK downloads are deferred into __init__ (not at module level).
The module-level singleton uses a lazy proxy so the heavy __init__ (filesystem
scan + model loading) only runs when the detector is first accessed, not when
the module is imported.
"""

import numpy as np
import os
import logging
import glob
import json
import re
import string
from typing import Union, Tuple, List, Dict, Optional, Any
from datetime import datetime
import threading
import gc

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from safe_transformers import run_isolated_text_classification
from translator_utils import content_translator
from ocr_utils import extract_text_from_image, get_image_reader_status

# Optional extensions
REALTIME_READY = False

# Check transformers/torch availability WITHOUT importing at module level.
# Importing transformers at module level hard-crashes Python 3.9 on some
# Windows setups (DLL conflict). The real imports happen inside methods.
import importlib.util as _ilu
TRANSFORMERS_AVAILABLE = (
    _ilu.find_spec("transformers") is not None and
    _ilu.find_spec("torch") is not None
)
del _ilu

# Configure logging
logger = logging.getLogger(__name__)
_NLTK_SETUP_ATTEMPTED = False


def _coerce_text(value: Union[str, List[str], Tuple[str, ...], Any]) -> str:
    """Convert translator / UI payloads into a safe text string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(_coerce_text(item) for item in value if item is not None)
    return str(value)


def _download_nltk_data():
    """Download missing NLTK resources quietly. Safe to call multiple times."""
    global _NLTK_SETUP_ATTEMPTED
    if _NLTK_SETUP_ATTEMPTED:
        return
    _NLTK_SETUP_ATTEMPTED = True

    resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
    }
    for path, resource in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:
                logger.warning(f"NLTK resource {resource} unavailable: {exc}")


class FakeNewsDetector:
    """
    Fake News Detector with multiple model support.
    Can load both traditional ML and transformer models.
    """

    def __init__(self, use_transformer: bool = False, model_path: str = 'models/fake_news/'):
        """
        Initialize the fake news detector.

        Args:
            use_transformer: Whether to prefer transformer model.
            model_path: Path to saved models.
        """
        # Download NLTK data here (deferred from module level)
        _download_nltk_data()

        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self.transformer_model = None
        self.transformer_pipeline = None
        self.transformer_backend = None
        self._invert_model_labels = False
        self.is_trained = False
        self.model_metadata = {}
        self.available_models = []
        
        # Ensemble support
        self.ensemble_models = []
        self.model_names = []
        self.model_paths = []
        self._lock = threading.RLock()
        self._hf_models_loaded = False
        self._loaded_hf_pipelines = {}  # {name: pipeline}
        
        self.available_hf_models = {
            "HF_DistilRoBERTa": "mrm8488/distilroberta-finetuned-fake-news",
            # "HF_RoBERTa_Fake": "hamzab/roberta-fake-news-classification", # Disabled for 8GB RAM stability
            # "HF_BERT_Fake": "diptamath/bert_fake_news",                  # Disabled for 8GB RAM stability
        }

        # Text preprocessing components
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords are unavailable. Continuing with an empty stopword set.")
            self.stop_words = set()

        # Create model directory
        os.makedirs(self.model_path, exist_ok=True)

        # Only scan for models here. The actual model is loaded on demand.
        self.get_available_models()

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, texts: List[str], labels: List[int], model_type: str = 'random_forest'):
        """
        Train a traditional ML model.

        Args:
            texts: List of input texts.
            labels: List of binary labels (0=REAL, 1=FAKE).
            model_type: 'logistic' or 'random_forest'.
        """
        logger.info(f"Training {model_type} model on {len(texts)} samples")

        processed_texts = [self.preprocess_text(t) for t in texts]
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

        classifier = (LogisticRegression(max_iter=1000)
                      if model_type == 'logistic'
                      else RandomForestClassifier(n_estimators=100))

        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', classifier),
        ])
        self.model.fit(processed_texts, labels)
        self.is_trained = True
        self.use_transformer = False
        self.model_metadata = {
            'type': model_type,
            'timestamp': datetime.now().isoformat(),
            'samples': len(texts),
        }
        logger.info(f"✅ Training completed for {model_type}")
        return True

    def _save_model(self, model_filename: str = None):
        """Save the currently trained traditional model."""
        if not self.is_trained or self.model is None or self.use_transformer:
            logger.warning("No traditional model to save")
            return False
        try:
            if model_filename is None:
                model_type = self.model_metadata.get('type', 'model')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = os.path.join(self.model_path, f"{model_type}_{timestamp}.pkl")
            os.makedirs(os.path.dirname(model_filename), exist_ok=True)
            joblib.dump(self.model, model_filename)
            meta_path = model_filename.replace('.pkl', '_metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
            logger.info(f"✅ Model saved to {model_filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    # ── Text preprocessing ────────────────────────────────────────────────────
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        text = _coerce_text(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        return text

    # ── Model loading ─────────────────────────────────────────────────────────
    def load_best_model(self):
        """Automatically load the best available model."""
        logger.info(f"Scanning for models in {self.model_path}")
        self.get_available_models()

        if not self.available_models:
            logger.warning("No models found in directory")
            return

        # On 8GB systems, prefer Random Forest for stability and speed
        rf_models = [m for m in self.available_models if m['type'] == 'random_forest']
        if rf_models:
            latest = rf_models[0]
            logger.info(f"Prioritizing Random Forest for stability: {latest['name']}")
            if self.load_traditional_model(latest['path']):
                self.use_transformer = False
                self.is_trained = True
                logger.info("✅ Loaded Random Forest model")
                return

        # Fall back to transformer models if RF is not available
        transformer_models = [m for m in self.available_models if m['type'] == 'transformer']
        if transformer_models and TRANSFORMERS_AVAILABLE:
            latest = transformer_models[0]
            logger.info(f"Loading transformer model: {latest['name']}")
            if self.load_transformer_model(latest['path']):
                self.use_transformer = True
                self.is_trained = True
                logger.info("✅ Loaded transformer model")
                return

        logger.warning("No models could be loaded. Using fallback predictions.")
        self.is_trained = False

    def _ensure_hf_models_ready(self, requested_names: Optional[List[str]] = None):
        """Register HF models for isolated inference and unload those not requested."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available - skipping HF models.")
            return

        with self._lock:
            available_names = list(self.available_hf_models.keys())
            target_names = [n for n in (requested_names or available_names) if n in available_names]
            
            # --- UNLOAD UNUSED MODELS ---
            to_unload = [name for name in self._loaded_hf_pipelines if name not in target_names]
            for name in to_unload:
                logger.info(f"Unloading HF model to save RAM: {name}")
                del self._loaded_hf_pipelines[name]
            
            if to_unload:
                gc.collect()

            # --- LOAD REQUESTED MODELS ---
            to_load = [name for name in target_names if name not in self._loaded_hf_pipelines]
            if not to_load:
                return

            for name in to_load:
                model_id = self.available_hf_models.get(name)
                if not model_id:
                    continue
                logger.info(f"Registering HF model for isolated inference: {name} ({model_id})")
                self._loaded_hf_pipelines[name] = {
                    "model": model_id,
                    "tokenizer": model_id,
                    "local_files_only": False,
                }

            self._hf_models_loaded = bool(self._loaded_hf_pipelines)
            if self._loaded_hf_pipelines:
                self.is_trained = True
            gc.collect()

    def get_huggingface_models(self) -> List[str]:
        """Return names of available open-source models."""
        return list(self.available_hf_models.keys())

    def ensure_model_loaded(self, model_path: Optional[str] = None, use_ensemble: bool = False, requested_models: Optional[List[str]] = None) -> bool:
        """Load requested model(s) on demand."""
        if use_ensemble or requested_models:
            # If specifically requested models are provided, filter HF loading
            hf_requested = None
            if requested_models:
                hf_requested = [m for m in requested_models if m in self.available_hf_models]
                
                # If local models are NOT in requested_models, we might want to unload them
                # But for now, we'll just ensure the best local one is available if requested
                local_requested = [m for m in requested_models if not m.startswith("HF_")]
                if local_requested:
                    if not self.is_trained:
                        self.load_best_model()
                else:
                    # If No local model requested but one is loaded, unload it if it's heavy (transformer)
                    if self.use_transformer:
                        logger.info("Unloading local transformer to save RAM as it was not requested.")
                        self.unload_current_model()
            else:
                # Default ensemble behavior
                if not self.is_trained:
                    self.load_best_model()
            
            # Trigger lazy load/unload of HF models
            self._ensure_hf_models_ready(requested_names=hf_requested)
            return bool(
                self.is_trained
                or self.model is not None
                or self.transformer_backend is not None
                or self._loaded_hf_pipelines
            )

        self.get_available_models()

        if model_path:
            normalized_requested = os.path.normcase(os.path.abspath(model_path))
            match = next(
                (
                    m for m in self.available_models
                    if os.path.normcase(os.path.abspath(m['path'])) == normalized_requested
                ),
                None,
            )
            if not match:
                logger.warning(f"Requested model path not found: {model_path}")
                return False
        else:
            if not self.available_models:
                logger.warning("No models available for on-demand loading.")
                return False
            preferred = [m for m in self.available_models if m['type'] == 'transformer']
            match = preferred[0] if preferred and TRANSFORMERS_AVAILABLE else self.available_models[0]

        loaded_path = self.model_metadata.get('loaded_path')
        if loaded_path and os.path.normcase(os.path.abspath(loaded_path)) == os.path.normcase(os.path.abspath(match['path'])):
            return True

        self.unload_current_model()

        if match['type'] == 'transformer':
            loaded = self.load_transformer_model(match['path'])
            if not loaded:
                logger.warning("Transformer load failed. Falling back to a traditional model if available.")
                fallback = next((m for m in self.available_models if m['type'] != 'transformer'), None)
                if fallback is not None:
                    loaded = self.load_traditional_model(fallback['path'])
                    if loaded:
                        match = fallback
        else:
            loaded = self.load_traditional_model(match['path'])
            if not loaded:
                logger.warning("Traditional model load failed. Falling back to a transformer model if available.")
                fallback = next((m for m in self.available_models if m['type'] == 'transformer'), None)
                if fallback is not None:
                    loaded = self.load_transformer_model(fallback['path'])
                    if loaded:
                        match = fallback

        if loaded:
            self.model_metadata['loaded_path'] = match['path']
        return loaded

    def unload_current_model(self):
        """Release loaded model objects to reduce memory pressure."""
        with self._lock:
            self.model = None
            self.vectorizer = None
            self.tokenizer = None
            self.transformer_model = None
            self.transformer_pipeline = None
            self.transformer_backend = None
            self.is_trained = False
            self.use_transformer = False
            self.model_metadata.pop('loaded_path', None)
            
            # Explicitly clear HF models if needed
            for name in list(self._loaded_hf_pipelines.keys()):
                logger.info(f"Purging model from RAM: {name}")
                del self._loaded_hf_pipelines[name]
        
        self.clear_cache()

    def clear_cache(self):
        """Perform aggressive garbage collection to free RAM."""
        gc.collect()
        logger.info("FakeNewsDetector: System cache cleared and GC triggered.")

    def get_realtime_verifier(self):
        """Import the realtime verifier only when requested."""
        try:
            from realtime_verifier import realtime_verifier
            return realtime_verifier
        except Exception as e:
            logger.error(f"Realtime verifier unavailable: {e}")
            return None

    def _apply_label_orientation(self, label: str, confidence: float) -> Tuple[str, float]:
        if label not in {"FAKE", "REAL"}:
            return label, confidence
        if self._invert_model_labels:
            return ("REAL", confidence) if label == "FAKE" else ("FAKE", confidence)
        return label, confidence

    def _calibrate_label_orientation(self, predictor) -> None:
        probes = [
            ("REAL", "Government report confirms the annual budget was released on schedule."),
            ("REAL", "According to university research, the study was officially published this year."),
            ("FAKE", "Scientists confirmed the moon is made entirely of cheese."),
            ("FAKE", "Secret miracle cure shocks doctors and they do not want you to know."),
        ]
        self._invert_model_labels = False
        mismatches = 0
        try:
            for expected, sample_text in probes:
                predicted_label, _ = predictor(sample_text)
                if predicted_label != expected:
                    mismatches += 1
            self._invert_model_labels = mismatches > (len(probes) // 2)
            if self._invert_model_labels:
                logger.warning("Loaded fake-news model appears to have inverted label semantics. Auto-correcting outputs.")
        except Exception as exc:
            logger.warning(f"Could not calibrate label orientation: {exc}")
            self._invert_model_labels = False

    def load_traditional_model(self, model_path: str) -> bool:
        """Load a trained traditional ML model."""
        try:
            logger.info(f"Loading traditional model from {model_path}")
            loaded_obj = joblib.load(model_path)

            # Some legacy exports contain a whole FakeNewsDetector instance
            # instead of just the sklearn pipeline. Unwrap those safely.
            if hasattr(loaded_obj, "model") and getattr(loaded_obj, "model", None) is not None and not hasattr(loaded_obj, "named_steps"):
                logger.info("Detected wrapped FakeNewsDetector artifact. Extracting inner sklearn pipeline.")
                self.model = loaded_obj.model
                if getattr(loaded_obj, "vectorizer", None) is not None:
                    self.vectorizer = loaded_obj.vectorizer
                if getattr(loaded_obj, "model_metadata", None):
                    self.model_metadata = dict(getattr(loaded_obj, "model_metadata", {}) or {})
            else:
                self.model = loaded_obj

            base = os.path.basename(model_path).replace('.pkl', '')
            patterns = [
                os.path.join(self.model_path, f'vectorizer_{base}.pkl'),
                os.path.join(self.model_path, 'vectorizer.pkl'),
            ]
            for vec_path in patterns:
                if os.path.exists(vec_path):
                    self.vectorizer = joblib.load(vec_path)
                    logger.info(f"Loaded vectorizer from {vec_path}")
                    break

            if not self.vectorizer:
                all_vecs = glob.glob(os.path.join(self.model_path, "vectorizer*.pkl"))
                if all_vecs:
                    self.vectorizer = joblib.load(all_vecs[0])
                    logger.info(f"Using fallback vectorizer from {all_vecs[0]}")

            if hasattr(self.model, 'named_steps') and 'tfidf' in self.model.named_steps:
                self.vectorizer = self.model.named_steps['tfidf']

            if not hasattr(self.model, 'predict'):
                logger.error("Loaded traditional artifact does not expose a predict method.")
                self.model = None
                self.vectorizer = None
                return False

            try:
                smoke_text = ["official report confirms scheduled update"]
                if hasattr(self.model, 'predict_proba'):
                    self.model.predict_proba(smoke_text)
                else:
                    self.model.predict(smoke_text)
            except Exception as validation_exc:
                logger.error(f"Traditional model validation failed: {validation_exc}")
                self.model = None
                self.vectorizer = None
                self.is_trained = False
                return False

            self.is_trained = True
            self.use_transformer = False
            self._calibrate_label_orientation(self._predict_traditional)

            meta_path = model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    self.model_metadata = json.load(f)

            logger.info("✅ Traditional model loaded")
            return True
        except Exception as e:
            logger.error(f"Error loading traditional model: {e}")
            return False

    def load_transformer_model(self, model_path: str) -> bool:
        """Register a trained transformer model for isolated inference."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available")
            return False
        try:
            logger.info(f"Loading transformer model from {model_path}")
            model_dir = model_path
            if os.path.isdir(model_path):
                final_dir = os.path.join(model_path, 'final_model')
                if os.path.exists(final_dir):
                    model_dir = final_dir

            config_path = os.path.join(model_dir, "config.json")
            if not os.path.exists(config_path):
                logger.error(f"Transformer model config not found at {config_path}")
                return False

            self.transformer_backend = {
                "model": model_dir,
                "tokenizer": model_dir,
                "local_files_only": True,
            }
            self.transformer_pipeline = None
            self.tokenizer = None
            self.transformer_model = None
            logger.info("✅ Transformer model registered for isolated inference")

            self.use_transformer = True
            self.is_trained = True
            self._calibrate_label_orientation(self._predict_transformer)
            gc.collect()

            meta_path = os.path.join(model_path, 'metrics.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    self.model_metadata = json.load(f)

            return True
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            return False

    def get_available_models(self, refresh: bool = False) -> List[Dict]:
        """Get list of all available trained models."""
        if self.available_models and not refresh:
            return self.available_models

        models = []
        logger.info(f"Scanning directory: {self.model_path}")

        trans_dirs = glob.glob(os.path.join(self.model_path, "transformer_*"))
        for d in trans_dirs:
            info = {
                'name': os.path.basename(d),
                'type': 'transformer',
                'path': d,
                'timestamp': os.path.getctime(d),
            }
            meta_path = os.path.join(d, 'metrics.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        info['accuracy'] = json.load(f).get('accuracy', 'N/A')
                except Exception:
                    pass
            models.append(info)

        rf_files = glob.glob(os.path.join(self.model_path, "random_forest_*.pkl"))
        for f in rf_files:
            info = {
                'name': os.path.basename(f),
                'type': 'random_forest',
                'path': f,
                'timestamp': os.path.getctime(f),
            }
            meta_path = f.replace('.pkl', '_metadata.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as jf:
                        info['accuracy'] = json.load(jf).get('accuracy', 'N/A')
                except Exception:
                    pass
            models.append(info)

        models.sort(key=lambda x: x['timestamp'], reverse=True)
        self.available_models = models
        logger.info(f"Total models found: {len(models)}")
        return models

    # ── Scoring helpers ────────────────────────────────────────────────────────
    def _calculate_clickbait_score(self, text: Union[str, List[str]]) -> float:
        """Calculate a clickbait score based on linguistic triggers."""
        text = _coerce_text(text)

        score = 0.0
        text_lower = text.lower()
        triggers = [
            'shocking', 'unbelievable', 'miracle', 'revealed', 'finally',
            "you won't believe", 'worst', 'best', 'secret', 'hidden',
            'illegal', 'banned', 'exposed', 'exclusive', 'breaking',
        ]
        score += min(sum(1 for t in triggers if t in text_lower) * 0.15, 0.45)
        if '!!!' in text: score += 0.2
        if '???' in text: score += 0.15
        words = text.split()
        if len(words) > 5:
            caps = sum(1 for w in words if w.isupper() and len(w) > 1)
            if caps / len(words) > 0.3:
                score += 0.2
        if text.strip().endswith('?'):
            score += 0.1
        return min(score, 1.0)

    def get_image_reader_status(self) -> Dict[str, Any]:
        """Expose image-reader availability for screenshot-based verification."""
        try:
            return get_image_reader_status()
        except Exception as e:
            logger.error(f"Image reader status check failed: {e}")
            return {
                'available': False,
                'backend': None,
                'error': str(e),
            }

    def predict_from_image(
        self,
        image_source: Any,
        check_realtime: bool = True,
        use_ensemble: bool = False,
        requested_models: Optional[List[str]] = None,
        image_name: Optional[str] = None,
    ) -> Tuple[str, float, float, Dict]:
        """
        Extract article text from an image and run the standard fake-news check.

        Args:
            image_source: PIL image, bytes, file-like object, or path.
            check_realtime: Whether to verify against live news feeds.
            use_ensemble: Whether to use all available models.
            requested_models: Specific list of models to use.
            image_name: Optional friendly filename for metadata.

        Returns:
            Tuple of (label, confidence, clickbait_score, metadata).
        """
        ocr_payload = extract_text_from_image(image_source, image_name=image_name)
        extracted_text = _coerce_text(ocr_payload.get("text"))
        normalized_text = re.sub(r"\s+", " ", extracted_text).strip()

        if len(normalized_text) < 25:
            raise ValueError(
                "The image reader could not extract enough article text. "
                "Please upload a clearer article screenshot or paste the article text directly."
            )

        label, conf, clickbait_score, meta = self.predict(
            extracted_text,
            check_realtime=check_realtime,
            use_ensemble=use_ensemble,
            requested_models=requested_models,
        )

        meta = dict(meta)
        meta["source_type"] = "image"
        meta["ocr"] = {
            "backend": ocr_payload.get("backend", "windows_ocr"),
            "image_name": ocr_payload.get("image_name", image_name),
            "line_count": ocr_payload.get("line_count", 0),
            "word_count": ocr_payload.get("word_count", 0),
            "language": ocr_payload.get("language", ""),
            "width": ocr_payload.get("width"),
            "height": ocr_payload.get("height"),
            "preprocessed_size": ocr_payload.get("preprocessed_size"),
            "extracted_text": extracted_text,
        }
        return (label, conf, clickbait_score, meta)

    def _apply_realtime_adjustment(self, label: str, conf: float, meta: Dict[str, Any], text: str) -> Tuple[str, float]:
        rt = meta.get('realtime_result')
        if not rt or rt.get('status') != 'SUCCESS':
            return label, conf

        consensus = float(rt.get('consensus_score', 0.5) or 0.0)
        contradiction_score = float(rt.get('contradiction_score', 0.0) or 0.0)
        verdict_code = str(rt.get('verdict_code', 'UNVERIFIED') or 'UNVERIFIED')

        if verdict_code == 'CONTRADICTED_BY_SOURCES' or contradiction_score >= 0.2:
            if label == 'REAL':
                label = 'FAKE'
                conf = max(conf, min(0.95, 0.72 + contradiction_score * 0.4))
                meta['realtime_impact'] = "Reversed REAL label because matched live sources contradicted the claim."
            else:
                conf = min(0.99, max(conf, 0.76 + contradiction_score * 0.25))
                meta['realtime_impact'] = "Suspicion strengthened because matched live sources contradicted the claim."
            logger.info(f"Label adjusted by contradiction signals for: {text[:50]}...")
            return label, conf

        if verdict_code == 'VERIFIED_ONLINE' and consensus > 0.7:
            if label == 'FAKE':
                if consensus > 0.85:
                    label = 'REAL'
                    conf = consensus
                    meta['realtime_impact'] = "Reversed FAKE label due to strong factual consensus."
                else:
                    conf = max(0.5, conf * (1 - consensus))
                    meta['realtime_impact'] = "Confidence lowered due to conflicting factual reports."
            else:
                conf = min(0.99, conf + (consensus * 0.2))
                meta['realtime_impact'] = "Credibility boosted by strong factual consensus."
            return label, conf

        if consensus < 0.25:
            if label == 'REAL':
                if consensus < 0.1:
                    label = 'FAKE'
                    conf = 0.9
                    meta['realtime_impact'] = "Reversed REAL label: No factual reports found."
                else:
                    conf = max(0.5, conf * (consensus + 0.5))
                    meta['realtime_impact'] = "Confidence lowered due to lack of factual reporting."
            else:
                conf = min(0.99, conf + 0.1)
                meta['realtime_impact'] = "Suspicion confirmed by lack of legitimate reporting."

            logger.info(f"Label adjusted by weak consensus for: {text[:50]}...")

        return label, conf

    def _looks_like_short_claim(self, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", _coerce_text(text)).strip()
        words = normalized.split()
        return bool(normalized) and len(words) <= 18 and len(normalized) <= 180

    def _apply_claim_guard(self, label: str, conf: float, meta: Dict[str, Any], text: str) -> Tuple[str, float]:
        if label != 'REAL' or not self._looks_like_short_claim(text):
            return label, conf

        rt = meta.get('realtime_result')
        if not rt or rt.get('status') != 'SUCCESS':
            meta['realtime_impact'] = "Short claim could not be corroborated with live reporting, so it was marked unverified."
            return 'UNVERIFIED', min(conf, 0.6)

        verdict_code = str(rt.get('verdict_code', 'UNVERIFIED') or 'UNVERIFIED')
        consensus = float(rt.get('consensus_score', 0.0) or 0.0)
        if verdict_code != 'VERIFIED_ONLINE' or consensus < 0.72:
            meta['realtime_impact'] = "Short claim was not strongly verified by live reporting, so it was marked unverified."
            return 'UNVERIFIED', min(conf, max(consensus, 0.58))

        return label, conf

    # ── Prediction ─────────────────────────────────────────────────────────────
    def predict(self, text: str, check_realtime: bool = False, use_ensemble: bool = False, requested_models: Optional[List[str]] = None) -> Tuple[str, float, float, Dict]:
        """
        Predict if text is fake news.

        Args:
            text: Input article text.
            check_realtime: Whether to verify against live news feeds.
            use_ensemble: Whether to use all available models (local + HF).
            requested_models: Specific list of model names to use.

        Returns:
            Tuple of (label, confidence, clickbait_score, metadata).
        """
        translated_text, original_lang, was_translated = content_translator.translate_to_english(text)
        translated_text = _coerce_text(translated_text)
        clickbait_score = self._calculate_clickbait_score(translated_text[:200])

        meta = {
            'original_language': original_lang,
            'was_translated': was_translated,
            'processed_text': translated_text if was_translated else None,
            'realtime_result': None,
            'ensemble_mode': use_ensemble,
            'individual_scores': {}
        }

        if check_realtime:
            try:
                realtime_verifier = self.get_realtime_verifier()
                if realtime_verifier is not None:
                    meta['realtime_result'] = realtime_verifier.verify_claim(translated_text[:300])
            except Exception as e:
                logger.error(f"Real-time verification during predict failed: {e}")

        if not self.is_trained or (
            self.model is None
            and self.transformer_pipeline is None
            and self.transformer_model is None
            and self.transformer_backend is None
            and not self._loaded_hf_pipelines
        ):
            logger.warning("No trained model loaded. Using fallback.")
            label, conf = self._fallback_predict(translated_text)
            if check_realtime and meta.get('realtime_result'):
                label, conf = self._apply_realtime_adjustment(label, conf, meta, translated_text)
            label, conf = self._apply_claim_guard(label, conf, meta, translated_text)
            return (label, conf, clickbait_score, meta)

        try:
            if use_ensemble or requested_models:
                label, conf, ensemble_details = self._predict_ensemble(translated_text, requested_models=requested_models)
                meta['individual_scores'] = ensemble_details
            elif self.use_transformer:
                label, conf = self._predict_transformer(translated_text)
            else:
                label, conf = self._predict_traditional(translated_text)

            if check_realtime and meta.get('realtime_result'):
                label, conf = self._apply_realtime_adjustment(label, conf, meta, translated_text)
            label, conf = self._apply_claim_guard(label, conf, meta, translated_text)
            
            # Aggressive cleanup after heavy operation
            if use_ensemble or requested_models:
                gc.collect()

            return (label, conf, clickbait_score, meta)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            label, conf = self._fallback_predict(translated_text)
            if check_realtime and meta.get('realtime_result'):
                label, conf = self._apply_realtime_adjustment(label, conf, meta, translated_text)
            label, conf = self._apply_claim_guard(label, conf, meta, translated_text)
            return (label, conf, clickbait_score, meta)

    def _predict_traditional(self, text: str) -> Tuple[str, float]:
        """Predict using traditional ML model."""
        try:
            if not self.model:
                logger.warning("Traditional model called but not loaded.")
                return self._fallback_predict(text)

            processed = self.preprocess_text(text)
            if not isinstance(processed, str):
                processed = str(processed)

            # Some older/corrupted sklearn models might fail on predict_proba
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba([processed])[0]
                    if len(proba) == 2:
                        fake_prob, real_prob = proba[1], proba[0]
                        raw_label = 'FAKE' if fake_prob > real_prob else 'REAL'
                        raw_conf = float(fake_prob) if fake_prob > real_prob else float(real_prob)
                        return self._apply_label_orientation(raw_label, raw_conf)
                except Exception as e:
                    logger.warning(f"predict_proba failed, falling back to predict: {e}")

            pred = self.model.predict([processed])[0]
            if isinstance(pred, str):
                normalized = pred.strip().upper()
                if normalized in {"FAKE", "REAL"}:
                    return self._apply_label_orientation(normalized, 0.95)
            raw_label = 'FAKE' if pred == 1 else 'REAL'
            return self._apply_label_orientation(raw_label, 0.95)
        except Exception as e:
            logger.error(f"Traditional prediction error: {e}")
            # If it's the 'list' object has no attribute 'strip', it's likely a vectorizer-level exception
            return self._fallback_predict(text)

    def _predict_transformer(self, text: str) -> Tuple[str, float]:
        """Predict using transformer model."""
        try:
            if self.transformer_backend is not None:
                result = run_isolated_text_classification(
                    self.transformer_backend["model"],
                    _coerce_text(text)[:512],
                    tokenizer_ref=self.transformer_backend.get("tokenizer"),
                    local_files_only=bool(self.transformer_backend.get("local_files_only", False)),
                )
                if not result.get("ok"):
                    raise RuntimeError(result.get("error", "Unknown isolated inference failure"))
                result = result["result"]
                label = result['label']
                score = result['score']
                if 'LABEL_1' in label or 'FAKE' in label.upper():
                    return self._apply_label_orientation('FAKE', score)
                elif 'LABEL_0' in label or 'REAL' in label.upper():
                    return self._apply_label_orientation('REAL', score)
                raw_label = 'FAKE' if score > 0.5 else 'REAL'
                return self._apply_label_orientation(raw_label, score)

            elif self.tokenizer and self.transformer_model:
                import torch
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                if probs.shape[-1] == 2:
                    fake_prob = probs[0][1].item()
                    real_prob = probs[0][0].item()
                    raw_label = 'FAKE' if fake_prob > real_prob else 'REAL'
                    raw_conf = fake_prob if fake_prob > real_prob else real_prob
                    return self._apply_label_orientation(raw_label, raw_conf)
                pred = torch.argmax(probs, dim=-1).item()
                raw_label = 'FAKE' if pred == 1 else 'REAL'
                return self._apply_label_orientation(raw_label, probs[0][pred].item())
            return self._fallback_predict(text)
        except Exception as e:
            logger.error(f"Transformer prediction error: {e}")
            return self._fallback_predict(text)

    def _predict_ensemble(self, text: str, requested_models: Optional[List[str]] = None) -> Tuple[str, float, Dict[str, Any]]:
        """Run parallel inference across multiple models and aggregate results."""
        results = {}
        
        # Determine which models to actually run
        run_local = True
        run_hf_names = list(self._loaded_hf_pipelines.keys())
        
        if requested_models:
            run_hf_names = [n for n in run_hf_names if n in requested_models]
            # Local models names are usually matched by substring or type
            # We'll check if any non-HF models are requested
            local_req = [m for m in requested_models if not m.startswith("HF_")]
            run_local = len(local_req) > 0

        # Prepare list of inference tasks
        tasks = []
        
        # 1. Local model (if loaded and requested)
        if run_local and (self.model or self.transformer_pipeline or self.transformer_model):
            def local_task():
                try:
                    if self.use_transformer:
                        return "Local_Transformer", self._predict_transformer(text)
                    else:
                        return "Local_ML", self._predict_traditional(text)
                except Exception as e:
                    return "Local", (None, 0)
            tasks.append(local_task)
            
        # 2. HF Ensemble models (only those in run_hf_names)
        for name in run_hf_names:
            model = self._loaded_hf_pipelines.get(name)
            if not model:
                continue
            
            def hf_task(m=model, n=name):
                try:
                    res = run_isolated_text_classification(
                        m["model"],
                        _coerce_text(text)[:512],
                        tokenizer_ref=m.get("tokenizer"),
                        local_files_only=bool(m.get("local_files_only", False)),
                    )
                    if not res.get("ok"):
                        raise RuntimeError(res.get("error", "Unknown isolated inference failure"))
                    res = res["result"]
                    lbl = res['label']
                    scr = res['score']
                    if 'LABEL_1' in lbl or 'FAKE' in lbl.upper():
                        return n, ('FAKE', scr)
                    return n, ('REAL', scr)
                except Exception as e:
                    return n, (None, 0)
            tasks.append(hf_task)
            
        if not tasks:
            l, c = self._fallback_predict(text)
            return l, c, {}

        # Execute sequentially (for 8GB RAM stability)
        for t in tasks:
            try:
                name, (lbl, conf) = t()
                if lbl:
                    results[name] = {'label': lbl, 'confidence': conf}
            except Exception as e:
                logger.error(f"Ensemble task error: {e}")
                    
        if not results:
            l, c = self._fallback_predict(text)
            return l, c, {}
            
        # Aggregate scores (Weighted: HF models get slightly more weight if they exist)
        total_fake_score = 0
        total_real_score = 0
        weight_sum = 0
        
        for name, res in results.items():
            weight = 1.2 if name.startswith("HF_") else 1.0
            weight_sum += weight
            
            if res['label'] == 'FAKE':
                total_fake_score += res['confidence'] * weight
                total_real_score += (1 - res['confidence']) * weight
            else:
                total_real_score += res['confidence'] * weight
                total_fake_score += (1 - res['confidence']) * weight
                
        avg_fake = total_fake_score / weight_sum
        avg_real = total_real_score / weight_sum
        
        # Cleanup after accumulation
        gc.collect()

        if avg_fake > avg_real:
            return 'FAKE', avg_fake, results
        else:
            return 'REAL', avg_real, results

    def _fallback_predict(self, text: str) -> Tuple[str, float]:
        """Fallback prediction using simple heuristics."""
        text = _coerce_text(text)
        text_lower = text.lower()
        fake_indicators = [
            'breaking', 'shocking', "you won't believe", 'viral',
            "they don't want you to know", 'secret', 'conspiracy',
            'miracle', 'cure', 'hidden truth', 'what happened next',
        ]
        real_indicators = [
            'according to', 'source', 'report', 'study', 'research',
            'official', 'government', 'university', 'published',
        ]
        fake_score = sum(1 for w in fake_indicators if w in text_lower)
        real_score = sum(1 for w in real_indicators if w in text_lower)
        total = fake_score + real_score
        if total == 0:
            return ('REAL', 0.6)
        fake_ratio = fake_score / total
        if fake_ratio > 0.6:
            return ('FAKE', fake_ratio)
        elif fake_ratio < 0.3:
            return ('REAL', 1 - fake_ratio)
        return ('REAL', 0.55)

    def get_model_info(self) -> Dict:
        """Get information about the currently loaded model."""
        info = {
            'is_trained': self.is_trained,
            'use_transformer': self.use_transformer,
            'model_path': self.model_path,
            'available_models': len(self.available_models) if self.available_models else 0,
            'ensemble_count': len(self._loaded_hf_pipelines),
            'metadata': self.model_metadata,
        }
        if self.is_trained:
            if self._loaded_hf_pipelines:
                info['model_type'] = f'Ensemble ({len(self._loaded_hf_pipelines)} OS models + Local)'
            elif self.use_transformer:
                info['model_type'] = 'transformer (isolated)'
            elif self.model:
                if hasattr(self.model, 'named_steps'):
                    classifier = self.model.named_steps.get('classifier')
                    info['model_type'] = type(classifier).__name__ if classifier else type(self.model).__name__
                else:
                    info['model_type'] = type(self.model).__name__
        else:
            info['model_type'] = 'fallback'
        return info


# ── Lazy proxy singleton ──────────────────────────────────────────────────────
# Importing this module is now instant. The real FakeNewsDetector.__init__
# (which scans the filesystem and loads models) only runs on first attribute
# access — i.e., when the detector is actually used after login.
class _LazyFakeNewsDetector:
    """Proxy that defers FakeNewsDetector construction until first use."""
    _instance: Optional[FakeNewsDetector] = None

    def _get(self) -> FakeNewsDetector:
        if self._instance is None:
            object.__setattr__(self, '_instance', FakeNewsDetector())
        return self._instance  # type: ignore[return-value]

    def __getattr__(self, name: str):
        return getattr(self._get(), name)

    def __setattr__(self, name: str, value):
        if name == '_instance':
            object.__setattr__(self, name, value)
        else:
            setattr(self._get(), name, value)


fake_news_detector = _LazyFakeNewsDetector()
