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
    """Prepare optional NLTK resources without blocking offline startup."""
    global _NLTK_SETUP_ATTEMPTED
    if _NLTK_SETUP_ATTEMPTED:
        return
    _NLTK_SETUP_ATTEMPTED = True

    allow_downloads = os.getenv("TRUTHGUARD_ALLOW_NLTK_DOWNLOADS", "0").lower() in {"1", "true", "yes"}
    resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
    }
    for path, resource in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            if not allow_downloads:
                logger.info(
                    "NLTK resource %s is missing; continuing with lightweight fallbacks. "
                    "Set TRUTHGUARD_ALLOW_NLTK_DOWNLOADS=1 to download it.",
                    resource,
                )
                continue
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
            requested_pool = available_names if requested_names is None else requested_names
            target_names = [n for n in requested_pool if n in available_names]
            
            # --- UNLOAD UNUSED MODELS ---
            to_unload = [name for name in self._loaded_hf_pipelines if name not in target_names]
            for name in to_unload:
                logger.info(f"Unloading HF model to save RAM: {name}")
                del self._loaded_hf_pipelines[name]
            
            if to_unload:
                self._hf_models_loaded = bool(self._loaded_hf_pipelines)
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
            'cover-up', 'hoax', 'doctors hate', 'miracle cure',
            'what happened next', 'before it is deleted', 'media blackout',
        ]
        score += min(sum(1 for t in triggers if t in text_lower) * 0.15, 0.45)
        if '!!!' in text: score += 0.2
        if '???' in text: score += 0.15
        if re.search(r'\b(share|forward)\b.{0,30}\b(now|before|everyone)\b', text_lower):
            score += 0.15
        if re.search(r'\b(secret|miracle|forbidden|suppressed)\b.{0,45}\b(cure|truth|proof|remedy)\b', text_lower):
            score += 0.15
        words = text.split()
        if len(words) > 5:
            caps = sum(1 for w in words if w.isupper() and len(w) > 1)
            if caps / len(words) > 0.3:
                score += 0.2
        if text.strip().endswith('?'):
            score += 0.1
        return min(score, 1.0)

    def _compute_credibility_report(self, text: str) -> Dict[str, Any]:
        """
        Aggressive multi-signal credibility analysis.
        Returns a structured report with per-dimension scores and an overall
        credibility_score (0.0 = completely fraudulent, 1.0 = fully credible).
        Safe to call on any text; never raises.
        """
        try:
            text = _coerce_text(text)
            tl = text.lower()
            words = text.split()
            word_count = max(len(words), 1)
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            sent_count = max(len(sentences), 1)
            url_count = len(re.findall(r'https?://\S+|www\.\S+', text))
            direct_quote_count = len(re.findall(r'"[^"]{12,}"|\'[^\']{12,}\'', text))

            # ── 1. Sourcing & Attribution signals ────────────────────────
            sourcing_positive = [
                'according to', 'sources say', 'officials said', 'spokesperson',
                'study shows', 'research published', 'peer-reviewed', 'data shows',
                'confirmed by', 'verified by', 'cited', 'referenced', 'report by',
                'journalists', 'investigation found', 'court documents', 'records show',
                'press release', 'statement from', 'ministry of', 'department of',
                'news agency', 'filing shows', 'regulator said', 'police said',
                'hospital said', 'researchers found', 'published in', 'interview with',
            ]
            sourcing_negative = [
                'anonymous', 'some people say', 'many believe', 'everyone knows',
                'they say', 'rumor has it', 'insider claims', 'unnamed source',
                'word on the street', 'i heard', 'apparently', 'supposedly',
                'sources close to', 'unconfirmed', 'allegedly', 'speculated',
                'people are saying', 'they will not tell you', 'they are hiding',
                'mainstream media refuses', 'mainstream media will not report',
            ]
            src_pos = sum(1 for p in sourcing_positive if p in tl)
            src_neg = sum(1 for n in sourcing_negative if n in tl)
            sourcing_score = float(np.clip(
                (src_pos * 0.15 + url_count * 0.08 + direct_quote_count * 0.04 - src_neg * 0.12 + 0.3),
                0.0,
                1.0,
            ))

            # ── 2. Sensationalism & Emotional manipulation ────────────────
            sensational_triggers = [
                'shocking', 'explosive', 'bombshell', 'outrage', 'scandal',
                "you won't believe", 'unbelievable', 'jaw-dropping', 'terrifying',
                'disgusting', 'heartbreaking', 'mind-blowing', 'enraging',
                'must read', 'must see', 'share this', 'spread the word',
                'wake up', 'sheeple', 'blind', 'brainwashed', 'censored',
                'they hide', 'deep state', 'shadow government', 'cover-up',
                'they don\'t want you', 'hidden truth', 'forbidden knowledge',
                'media blackout', 'before it is deleted', 'this will be removed',
                'share before', 'forward this', 'the truth finally exposed',
            ]
            sens_count = sum(1 for t in sensational_triggers if t in tl)
            # Exclamation & caps abuse
            excl_ratio = text.count('!') / word_count
            caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
            caps_ratio = caps_words / word_count
            sensationalism_score = float(np.clip(
                sens_count * 0.1 + excl_ratio * 0.5 + caps_ratio * 0.4, 0.0, 1.0
            ))

            # ── 3. Writing quality & Structure ───────────────────────────
            avg_sent_len = word_count / sent_count
            # Very short sentences → tabloid-style. Very long → academic.
            if avg_sent_len < 8:
                structure_score = 0.35
            elif avg_sent_len > 30:
                structure_score = 0.75
            else:
                structure_score = float(np.clip(0.35 + (avg_sent_len - 8) / 22 * 0.50, 0.0, 1.0))

            # Paragraph-like structure (presence of periods)
            period_density = text.count('.') / word_count
            if period_density > 0.04:
                structure_score = min(structure_score + 0.1, 1.0)

            # ── 4. Claim density (specific numbers & dates increase credibility) ─
            numbers = re.findall(r'\b\d{4}\b|\d+\.\d+%?|\$\d+|\d+\s*(?:million|billion|thousand)', text)
            date_patterns = re.findall(
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}\b',
                text,
            )
            claim_density_raw = (len(numbers) * 0.12 + len(date_patterns) * 0.15)
            claim_density_score = float(np.clip(claim_density_raw, 0.0, 0.6))

            # ── 5. Persuasion & Absolute language ────────────────────────
            absolute_terms = [
                'always', 'never', 'everyone', 'nobody', 'all',
                'proven fact', 'undeniable', 'irrefutable', 'the truth is',
                'fact:', 'fact -', '100%', 'definitely', 'certainly', 'obviously',
            ]
            abs_count = sum(1 for t in absolute_terms if t in tl)
            absolutism_penalty = float(np.clip(abs_count * 0.08, 0.0, 0.35))

            # ── 6. Credible institution mentions ─────────────────────────
            institutions = [
                'reuters', 'ap news', 'associated press', 'bbc', 'new york times',
                'washington post', 'guardian', 'nytimes', 'bloomberg', 'npr',
                'who ', 'world health', 'cdc', 'fda', 'nasa', 'un ', 'united nations',
                'harvard', 'oxford', 'mit ', 'stanford', 'university', 'institute',
                'journal of', 'nature ', 'science ', 'lancet', 'plos',
            ]
            inst_hits = sum(1 for i in institutions if i in tl)
            institution_score = float(np.clip(inst_hits * 0.12, 0.0, 0.45))

            # ── 7. Conspiracy / Pseudoscience signals ────────────────────
            conspiracy_terms = [
                'deep state', 'new world order', 'illuminati', 'globalist',
                'chemtrail', 'microchip', 'mind control', 'reptilian', 'lizard people',
                'flat earth', 'moon landing fake', 'crisis actor', 'false flag',
                'agenda 21', 'great reset', 'depopulation', 'poison', '5g',
                'vaccine chip', 'bill gates', 'soros', 'rothschild', 'cabal',
                'plandemic', 'climate hoax', 'fake pandemic', 'secret cabal',
            ]
            conspiracy_hits = sum(1 for c in conspiracy_terms if c in tl)
            conspiracy_penalty = float(np.clip(conspiracy_hits * 0.2, 0.0, 0.6))

            # ── 8. Propaganda & Polarizing language ──────────────────────
            propaganda_terms = [
                'radical left', 'far left', 'far-left', 'radical right', 'far right',
                'far-right', 'globalist agenda', 'mainstream media lies', 'fake news media',
                'lamestream', 'libtard', 'snowflake', 'communist', 'marxist',
                'satanic', 'evil elites', 'puppet', 'regime', 'tyranny',
                'enemy of the people', 'traitors', 'evil cabal', 'the regime media',
            ]
            prop_hits = sum(1 for p in propaganda_terms if p in tl)
            propaganda_penalty = float(np.clip(prop_hits * 0.15, 0.0, 0.5))

            # -- 9. Fraud, implausibility, and share-pressure signals --
            health_fraud_terms = [
                'miracle cure', 'secret cure', 'suppressed cure', 'doctors hate',
                'big pharma', 'cures cancer', 'cure cancer', 'cure diabetes',
                'instant cure', 'natural remedy cures', 'no side effects',
                'vaccine kills', 'deadly vaccine', 'detox', 'one weird trick',
            ]
            fabricated_authority_terms = [
                'experts are speechless', 'scientists baffled', 'scientists shocked',
                'doctors shocked', 'officials are hiding', 'leaked document proves',
                'secret document proves', 'whistleblower reveals', 'insider reveals',
            ]
            urgency_terms = [
                'share now', 'forward this', 'before it is deleted',
                'before they delete', 'before it gets removed', 'do not let them censor',
                'send this to everyone', 'act now', 'wake up before it is too late',
            ]
            implausible_terms = [
                'moon is made of cheese', 'earth is flat', 'lizard people',
                'reptilian', 'time traveler', 'time travel', 'immortality pill',
                'aliens built', 'dead returned to life', 'teleportation device',
                'mind control vaccine', '5g mind control', 'microchip vaccine',
            ]
            fraud_hits = sum(1 for p in health_fraud_terms if p in tl)
            fabricated_authority_hits = sum(1 for p in fabricated_authority_terms if p in tl)
            urgency_hits = sum(1 for p in urgency_terms if p in tl)
            implausible_hits = sum(1 for p in implausible_terms if p in tl)
            fraud_pattern_penalty = float(np.clip(
                fraud_hits * 0.22 + fabricated_authority_hits * 0.14,
                0.0,
                0.7,
            ))
            urgency_penalty = float(np.clip(urgency_hits * 0.16, 0.0, 0.45))
            implausibility_penalty = float(np.clip(implausible_hits * 0.35, 0.0, 0.8))

            evidence_gap_penalty = 0.0
            if word_count >= 45 and src_pos == 0 and inst_hits == 0 and url_count == 0:
                evidence_gap_penalty = 0.30
            elif word_count >= 20 and src_pos == 0 and inst_hits == 0 and url_count == 0:
                evidence_gap_penalty = 0.18

            severe_pattern_bonus = 0.0
            if fraud_hits >= 2 or fabricated_authority_hits >= 2:
                severe_pattern_bonus += 0.22
            if implausible_hits > 0:
                severe_pattern_bonus += 0.30
            if conspiracy_hits >= 2:
                severe_pattern_bonus += 0.18
            if urgency_hits > 0 and (fraud_hits > 0 or sens_count > 0):
                severe_pattern_bonus += 0.12
            severe_pattern_bonus = float(np.clip(severe_pattern_bonus, 0.0, 0.42))

            # ── Aggregate credibility score ───────────────────────────────
            raw = (
                sourcing_score * 0.28
                + structure_score * 0.15
                + claim_density_score * 0.12
                + institution_score * 0.20
                - sensationalism_score * 0.20
                - absolutism_penalty * 0.08
                - conspiracy_penalty * 0.25
                - propaganda_penalty * 0.15
                - fraud_pattern_penalty * 0.28
                - urgency_penalty * 0.16
                - implausibility_penalty * 0.30
                - evidence_gap_penalty * 0.18
            )
            credibility_score = float(np.clip(raw + 0.30, 0.0, 1.0))  # +0.30 neutral baseline
            risk_score = float(np.clip(
                sensationalism_score * 0.18
                + absolutism_penalty * 0.10
                + conspiracy_penalty * 0.22
                + propaganda_penalty * 0.14
                + fraud_pattern_penalty * 0.28
                + urgency_penalty * 0.14
                + implausibility_penalty * 0.26
                + evidence_gap_penalty * 0.18
                + severe_pattern_bonus
                + max(0.0, 0.35 - sourcing_score) * 0.18
                - institution_score * 0.12
                - claim_density_score * 0.05,
                0.0,
                1.0,
            ))

            # ── Verdict tier ──────────────────────────────────────────────
            if credibility_score >= 0.72:
                tier = 'CREDIBLE'
                tier_label = 'Credible'
            elif credibility_score >= 0.50:
                tier = 'UNCERTAIN'
                tier_label = 'Uncertain'
            elif credibility_score >= 0.30:
                tier = 'SUSPICIOUS'
                tier_label = 'Suspicious'
            else:
                tier = 'FRAUDULENT'
                tier_label = 'Likely Fraudulent'

            if risk_score >= 0.62 and credibility_score < 0.55:
                tier = 'FRAUDULENT'
                tier_label = 'Likely Fraudulent'
            elif risk_score >= 0.44 and tier == 'CREDIBLE':
                tier = 'UNCERTAIN'
                tier_label = 'Uncertain'

            # ── Flags list ────────────────────────────────────────────────
            flags: List[str] = []
            if sensationalism_score > 0.35:
                flags.append('High sensationalism / emotional manipulation')
            if sourcing_score < 0.25:
                flags.append('Weak or absent sourcing')
            if conspiracy_hits > 0:
                flags.append(f'{conspiracy_hits} conspiracy term(s) detected')
            if prop_hits > 0:
                flags.append(f'{prop_hits} polarizing language term(s) detected')
            if fraud_hits > 0:
                flags.append(f'{fraud_hits} health/scam misinformation pattern(s) detected')
            if fabricated_authority_hits > 0:
                flags.append(f'{fabricated_authority_hits} fabricated-authority phrase(s) detected')
            if urgency_hits > 0:
                flags.append('Urgent share-pressure language detected')
            if implausible_hits > 0:
                flags.append(f'{implausible_hits} implausible claim pattern(s) detected')
            if evidence_gap_penalty >= 0.18:
                flags.append('Article makes claims without clear sourcing, links, or institution references')
            if caps_ratio > 0.25:
                flags.append('Excessive ALLCAPS usage')
            if abs_count >= 3:
                flags.append('Heavy use of absolute / certain language')
            if src_neg >= 2:
                flags.append('Multiple unverified / anonymous sourcing signals')

            positives: List[str] = []
            if inst_hits > 0:
                positives.append(f'{inst_hits} credible institution(s) mentioned')
            if src_pos >= 2:
                positives.append(f'{src_pos} sourcing / attribution signal(s)')
            if len(date_patterns) > 0:
                positives.append(f'{len(date_patterns)} specific date(s) referenced')
            if len(numbers) > 1:
                positives.append(f'{len(numbers)} specific figure(s) cited')
            if url_count > 0:
                positives.append(f'{url_count} verifiable link(s) included')
            if direct_quote_count > 0:
                positives.append(f'{direct_quote_count} direct quote(s) included')

            return {
                'credibility_score': round(credibility_score, 4),
                'risk_score': round(risk_score, 4),
                'tier': tier,
                'tier_label': tier_label,
                'dimensions': {
                    'sourcing': round(sourcing_score, 3),
                    'institution_mentions': round(institution_score, 3),
                    'claim_density': round(claim_density_score, 3),
                    'structure': round(structure_score, 3),
                    'sensationalism': round(sensationalism_score, 3),
                    'absolutism_penalty': round(absolutism_penalty, 3),
                    'conspiracy_penalty': round(conspiracy_penalty, 3),
                    'propaganda_penalty': round(propaganda_penalty, 3),
                    'fraud_pattern_penalty': round(fraud_pattern_penalty, 3),
                    'urgency_penalty': round(urgency_penalty, 3),
                    'implausibility_penalty': round(implausibility_penalty, 3),
                    'evidence_gap_penalty': round(evidence_gap_penalty, 3),
                    'severe_pattern_bonus': round(severe_pattern_bonus, 3),
                    'risk_score': round(risk_score, 3),
                },
                'flags': flags,
                'positives': positives,
                'word_count': word_count,
                'sentence_count': sent_count,
            }
        except Exception as exc:
            logger.warning(f'Credibility report failed: {exc}')
            return {
                'credibility_score': 0.5,
                'risk_score': 0.5,
                'tier': 'UNCERTAIN',
                'tier_label': 'Uncertain',
                'dimensions': {},
                'flags': [],
                'positives': [],
                'word_count': 0,
                'sentence_count': 0,
            }

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
        if not self._looks_like_short_claim(text):
            return label, conf

        report = meta.get('credibility_report') or {}
        dims = report.get('dimensions') or {}
        risk = float(report.get('risk_score', 0.0) or 0.0)
        severe_signal = (
            float(dims.get('fraud_pattern_penalty', 0.0) or 0.0) >= 0.36
            or float(dims.get('implausibility_penalty', 0.0) or 0.0) >= 0.35
            or float(dims.get('conspiracy_penalty', 0.0) or 0.0) >= 0.40
            or risk >= 0.72
        )

        rt = meta.get('realtime_result')
        if not rt or rt.get('status') != 'SUCCESS':
            if label == 'FAKE' and severe_signal:
                return label, conf
            meta['realtime_impact'] = "Short claim could not be corroborated with live reporting, so it was marked unverified."
            return 'UNVERIFIED', min(conf, 0.6)

        verdict_code = str(rt.get('verdict_code', 'UNVERIFIED') or 'UNVERIFIED')
        consensus = float(rt.get('consensus_score', 0.0) or 0.0)
        if verdict_code != 'VERIFIED_ONLINE' or consensus < 0.72:
            if label == 'FAKE' and (severe_signal or verdict_code == 'CONTRADICTED_BY_SOURCES'):
                return label, conf
            meta['realtime_impact'] = "Short claim was not strongly verified by live reporting, so it was marked unverified."
            return 'UNVERIFIED', min(conf, max(consensus, 0.58))

        return label, conf

    def _apply_credibility_adjustment(self, label: str, conf: float, meta: Dict[str, Any]) -> Tuple[str, float]:
        """
        Secondary credibility nudge using the heuristic credibility report.
        Does NOT override strong ML or realtime decisions — only applies a
        mild confidence shift when the report strongly contradicts the current
        label.  Safe to call when no report is present.
        """
        report = meta.get('credibility_report')
        if not report:
            return label, conf

        tier = report.get('tier', 'UNCERTAIN')
        cred = float(report.get('credibility_score', 0.5))

        # Conspiracy / extremely fraudulent content detected — escalate
        dims = report.get('dimensions', {})
        conspiracy_pen = float(dims.get('conspiracy_penalty', 0.0))
        prop_pen = float(dims.get('propaganda_penalty', 0.0))
        sens = float(dims.get('sensationalism', 0.0))

        if conspiracy_pen >= 0.4:
            if label == 'REAL':
                label = 'FAKE'
                conf = max(conf, 0.82)
                meta['credibility_impact'] = 'Overridden to FAKE: extreme conspiracy signals detected.'
                return label, conf
            else:
                conf = min(0.99, conf + 0.08)
                meta['credibility_impact'] = 'Confidence raised: conspiracy signals reinforce FAKE verdict.'
                return label, conf

        if tier == 'FRAUDULENT' and label == 'REAL':
            # Credibility report says blatantly fraudulent but ML said REAL —
            # soften confidence rather than flip outright
            conf = max(0.45, conf * 0.65)
            meta['credibility_impact'] = 'Confidence softened: heuristic credibility analysis flagged content as likely fraudulent.'
        elif tier == 'CREDIBLE' and label == 'FAKE' and cred >= 0.78:
            # Report says clearly credible but model says FAKE — soften
            conf = max(0.45, conf * 0.70)
            meta['credibility_impact'] = 'Confidence softened: content signals are consistent with credible journalism.'
        elif tier == 'SUSPICIOUS' and label == 'REAL' and sens > 0.5:
            conf = max(0.48, conf * 0.75)
            meta['credibility_impact'] = 'Confidence softened: high sensationalism detected in otherwise credible-labelled content.'
        elif not meta.get('credibility_impact'):
            meta['credibility_impact'] = f'Content credibility tier: {report.get("tier_label", tier)}.'

        return label, conf

    # ── Prediction ─────────────────────────────────────────────────────────────
    def _realtime_context(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        rt = meta.get('realtime_result') or {}
        status = str(rt.get('status', '') or '')
        verdict_code = str(rt.get('verdict_code', 'UNVERIFIED') or 'UNVERIFIED')
        consensus = float(rt.get('consensus_score', 0.0) or 0.0)
        contradiction = float(rt.get('contradiction_score', 0.0) or 0.0)
        has_result = bool(rt)
        return {
            'has_result': has_result,
            'status': status,
            'verdict_code': verdict_code,
            'consensus': consensus,
            'contradiction': contradiction,
            'verified': (
                status == 'SUCCESS'
                and verdict_code == 'VERIFIED_ONLINE'
                and consensus >= 0.72
                and contradiction < 0.18
            ),
            'contradicted': (
                status == 'SUCCESS'
                and (verdict_code == 'CONTRADICTED_BY_SOURCES' or contradiction >= 0.20)
            ),
            'weak': (
                status == 'NO_RESULTS'
                or (
                    status == 'SUCCESS'
                    and (consensus < 0.42 or verdict_code in {'UNVERIFIED', 'PARTIALLY_SUPPORTED'})
                )
            ),
        }

    def _apply_evidence_fusion(
        self,
        label: str,
        conf: float,
        clickbait_score: float,
        meta: Dict[str, Any],
        text: str,
    ) -> Tuple[str, float]:
        """
        Fuse model output, live verification, and heuristic risk signals.
        This is deliberately stronger than a simple confidence nudge: weak
        REAL calls can be downgraded or flipped when multiple fraud signals
        align and live evidence does not verify the claim.
        """
        report = meta.get('credibility_report') or {}
        dims = report.get('dimensions') or {}
        tier = str(report.get('tier', 'UNCERTAIN') or 'UNCERTAIN')
        cred = float(report.get('credibility_score', 0.5) or 0.5)
        risk = float(report.get('risk_score', max(0.0, 1.0 - cred)) or 0.0)
        rt_ctx = self._realtime_context(meta)

        fraud = float(dims.get('fraud_pattern_penalty', 0.0) or 0.0)
        implausible = float(dims.get('implausibility_penalty', 0.0) or 0.0)
        conspiracy = float(dims.get('conspiracy_penalty', 0.0) or 0.0)
        urgency = float(dims.get('urgency_penalty', 0.0) or 0.0)
        sensationalism = float(dims.get('sensationalism', 0.0) or 0.0)
        evidence_gap = float(dims.get('evidence_gap_penalty', 0.0) or 0.0)

        severe_signal = (
            fraud >= 0.36
            or implausible >= 0.35
            or conspiracy >= 0.40
            or (fraud + urgency + sensationalism) >= 0.72
            or (risk >= 0.66 and evidence_gap >= 0.18)
        )
        high_risk = (
            risk >= 0.48
            or tier == 'FRAUDULENT'
            or (tier == 'SUSPICIOUS' and (clickbait_score >= 0.45 or sensationalism >= 0.32))
        )

        if rt_ctx['verified'] and not severe_signal:
            if label == 'FAKE':
                label = 'REAL'
                conf = max(conf * 0.65, rt_ctx['consensus'])
            else:
                conf = min(0.98, max(conf, 0.72 + rt_ctx['consensus'] * 0.20))
            meta['evidence_fusion'] = 'Live source consensus outweighed weaker misinformation signals.'
            return label, conf

        if rt_ctx['contradicted']:
            label = 'FAKE'
            conf = max(conf, min(0.97, 0.76 + rt_ctx['contradiction'] * 0.35 + risk * 0.10))
            meta['evidence_fusion'] = 'Live source contradiction was treated as decisive evidence.'
            return label, conf

        if severe_signal and not rt_ctx['verified']:
            label = 'FAKE'
            conf = max(conf, min(0.96, 0.78 + risk * 0.18 + clickbait_score * 0.08))
            meta['evidence_fusion'] = 'Severe fraud or implausibility signals overrode weaker model evidence.'
            return label, conf

        if high_risk and not rt_ctx['verified']:
            if label == 'UNVERIFIED' and self._looks_like_short_claim(text) and rt_ctx['weak'] and not severe_signal:
                meta.setdefault('evidence_fusion', 'Weak live corroboration kept this short claim in the unverified category.')
                return label, conf
            if label in {'REAL', 'UNVERIFIED'}:
                label = 'FAKE'
                conf = max(conf if label == 'FAKE' else 0.0, min(0.92, 0.70 + risk * 0.25 + clickbait_score * 0.08))
                meta['evidence_fusion'] = 'Multiple risk signals outweighed the original weak credible reading.'
            else:
                conf = min(0.96, max(conf, 0.68 + risk * 0.24))
                meta['evidence_fusion'] = 'Risk signals reinforced the fake-news verdict.'
            return label, conf

        if label == 'REAL' and rt_ctx['weak'] and (risk >= 0.34 or cred < 0.50):
            label = 'UNVERIFIED'
            conf = min(conf, 0.64)
            meta['evidence_fusion'] = 'Weak live corroboration and mixed credibility signals prevented a REAL verdict.'
            return label, conf

        meta.setdefault('evidence_fusion', 'Model, heuristic, and live-evidence signals were consistent enough to keep the verdict.')
        return label, conf

    def _build_user_reply(
        self,
        label: str,
        conf: float,
        clickbait_score: float,
        meta: Dict[str, Any],
        text: str,
    ) -> Dict[str, Any]:
        report = meta.get('credibility_report') or {}
        flags = list(report.get('flags') or [])
        positives = list(report.get('positives') or [])
        tier_label = report.get('tier_label', report.get('tier', 'Uncertain'))
        risk = float(report.get('risk_score', 0.5) or 0.5)
        cred = float(report.get('credibility_score', 0.5) or 0.5)
        rt_ctx = self._realtime_context(meta)

        if label == 'FAKE':
            summary = (
                f"I would not trust this as written. TruthGuard rates it likely fake or misleading "
                f"with {conf:.0%} confidence."
            )
        elif label == 'UNVERIFIED':
            summary = (
                "I cannot verify this strongly enough to call it real. Treat it as unverified "
                "until a reliable source confirms the key claim."
            )
        elif conf < 0.65:
            summary = (
                f"This leans credible, but confidence is modest at {conf:.0%}. I would treat it as plausible, "
                "not proven, until the original source and key details are checked."
            )
        else:
            summary = (
                f"This looks credible overall, with {conf:.0%} confidence, but it should still be checked "
                "against the original source for high-stakes decisions."
            )

        reasons: List[str] = []
        if rt_ctx['contradicted']:
            reasons.append('Live reporting matched the topic but contradicted important details in the claim.')
        elif rt_ctx['verified']:
            reasons.append('Live reporting from credible sources strongly corroborated the claim.')
        elif rt_ctx['has_result'] and rt_ctx['weak']:
            reasons.append('Live news search found weak or incomplete corroboration.')

        if meta.get('evidence_fusion'):
            reasons.append(str(meta['evidence_fusion']))
        if meta.get('credibility_impact'):
            reasons.append(str(meta['credibility_impact']))

        if flags:
            reasons.extend(flags[:3])
        elif positives:
            reasons.extend(positives[:3])

        if clickbait_score >= 0.55:
            reasons.append(f'Clickbait pressure is high ({clickbait_score:.2f}/1.00), which often appears in misinformation.')

        if not reasons:
            reasons.append(f'Credibility tier is {tier_label} with {cred:.0%} credibility and {risk:.0%} risk.')

        if label == 'FAKE':
            next_steps = [
                'Do not share it until the original source, date, and primary evidence are verified.',
                'Look for the same claim in reputable outlets or official records, not reposts or screenshots.',
            ]
        elif label == 'UNVERIFIED':
            next_steps = [
                'Ask for a primary source or a reputable outlet carrying the same specific claim.',
                'Check whether the headline changed key details such as dates, numbers, names, or locations.',
            ]
        else:
            next_steps = [
                'Open the cited source and confirm that the headline matches the body of the article.',
                'For medical, legal, financial, or election claims, verify against an official source too.',
            ]

        return {
            'summary': summary,
            'reasons': reasons[:6],
            'next_steps': next_steps,
            'credibility_score': round(cred, 4),
            'risk_score': round(risk, 4),
            'tier': tier_label,
        }

    def _finalize_prediction(
        self,
        label: str,
        conf: float,
        clickbait_score: float,
        meta: Dict[str, Any],
        translated_text: str,
        check_realtime: bool,
    ) -> Tuple[str, float, Dict[str, Any]]:
        meta['model_verdict'] = {
            'label': label,
            'confidence': round(float(conf), 4),
        }
        if check_realtime and meta.get('realtime_result'):
            label, conf = self._apply_realtime_adjustment(label, conf, meta, translated_text)
        label, conf = self._apply_claim_guard(label, conf, meta, translated_text)
        label, conf = self._apply_credibility_adjustment(label, conf, meta)
        label, conf = self._apply_evidence_fusion(label, conf, clickbait_score, meta, translated_text)
        conf = float(np.clip(conf, 0.0, 0.99))
        meta['final_verdict'] = {
            'label': label,
            'confidence': round(conf, 4),
        }
        meta['user_reply'] = self._build_user_reply(label, conf, clickbait_score, meta, translated_text)
        return label, conf, meta

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

        # Aggressive credibility analysis — always computed regardless of model
        credibility_report = self._compute_credibility_report(translated_text)

        meta = {
            'original_language': original_lang,
            'was_translated': was_translated,
            'processed_text': translated_text if was_translated else None,
            'realtime_result': None,
            'ensemble_mode': use_ensemble,
            'individual_scores': {},
            'credibility_report': credibility_report,
        }

        if check_realtime:
            try:
                realtime_verifier = self.get_realtime_verifier()
                if realtime_verifier is not None:
                    meta['realtime_result'] = realtime_verifier.verify_claim(translated_text[:6000])
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
            label, conf, meta = self._finalize_prediction(
                label,
                conf,
                clickbait_score,
                meta,
                translated_text,
                check_realtime,
            )
            return (label, conf, clickbait_score, meta)

        try:
            if use_ensemble or requested_models:
                label, conf, ensemble_details = self._predict_ensemble(translated_text, requested_models=requested_models)
                meta['individual_scores'] = ensemble_details
            elif self.use_transformer:
                label, conf = self._predict_transformer(translated_text)
            else:
                label, conf = self._predict_traditional(translated_text)

            label, conf, meta = self._finalize_prediction(
                label,
                conf,
                clickbait_score,
                meta,
                translated_text,
                check_realtime,
            )

            # Aggressive cleanup after heavy operation
            if use_ensemble or requested_models:
                gc.collect()

            return (label, conf, clickbait_score, meta)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            label, conf = self._fallback_predict(translated_text)
            label, conf, meta = self._finalize_prediction(
                label,
                conf,
                clickbait_score,
                meta,
                translated_text,
                check_realtime,
            )
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
        """
        Fallback prediction using the credibility report when no ML model is loaded.
        More aggressive and multi-signal than the old keyword scan.
        """
        report = self._compute_credibility_report(_coerce_text(text))
        score = float(report.get('credibility_score', 0.5))
        risk = float(report.get('risk_score', max(0.0, 1.0 - score)) or 0.0)
        tier = report.get('tier', 'UNCERTAIN')
        if tier == 'FRAUDULENT':
            return ('FAKE', float(np.clip(0.66 + risk * 0.30 + (1.0 - score) * 0.12, 0.68, 0.97)))
        if tier == 'SUSPICIOUS':
            return ('FAKE', float(np.clip(0.58 + risk * 0.28 + (0.55 - score) * 0.12, 0.55, 0.86)))
        if tier == 'CREDIBLE':
            return ('REAL', float(np.clip(score, 0.60, 0.95)))
        # UNCERTAIN — mild lean based on score
        if risk >= 0.48:
            return ('FAKE', float(np.clip(0.58 + risk * 0.25, 0.58, 0.78)))
        if score >= 0.55:
            return ('REAL', float(np.clip(score * 0.9, 0.52, 0.72)))
        return ('FAKE', float(np.clip(0.72 - score, 0.52, 0.72)))

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
