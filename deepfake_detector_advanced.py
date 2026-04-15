"""
Advanced Deepfake Detector with Ensemble Learning
Supports loading existing .h5 model files and model switching

OPTIMIZED: All heavy imports (TensorFlow, transformers, torch) are deferred
inside functions / methods and never run at module import time.
The module-level singleton uses a lazy proxy identical to the other detectors.
"""

import numpy as np
import cv2
from PIL import Image
import os
import glob
import logging
import tempfile
import threading
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heavy frameworks are imported LAZILY inside the class to avoid blocking
# the Streamlit startup process. These module-level booleans track whether
# each framework is actually available so we can fail gracefully.
# ---------------------------------------------------------------------------
_TF_AVAILABLE = None      # None = not yet checked
_TORCH_AVAILABLE = None
_TRANSFORMERS_AVAILABLE = None
_FACENET_PYTORCH_AVAILABLE = None
_MEDIAPIPE_AVAILABLE = None
DEFAULT_DEEPFAKE_MODEL_NAME = "HF_Transformer_V1"


def _check_tf():
    global _TF_AVAILABLE
    if _TF_AVAILABLE is None:
        try:
            import tensorflow  # noqa: F401
            _TF_AVAILABLE = True
        except Exception as e:
            _TF_AVAILABLE = False
            logger.warning("TensorFlow not available. Keras models won't load: %s", e)
    return _TF_AVAILABLE


def _check_torch():
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            _TORCH_AVAILABLE = True
        except Exception as e:
            _TORCH_AVAILABLE = False
            logger.warning("Torch not available. Optional face detectors won't load: %s", e)
    return _TORCH_AVAILABLE


def _check_transformers():
    global _TRANSFORMERS_AVAILABLE
    if _TRANSFORMERS_AVAILABLE is None:
        try:
            import transformers  # noqa: F401
            _TRANSFORMERS_AVAILABLE = True
        except Exception as e:
            _TRANSFORMERS_AVAILABLE = False
            logger.warning("Transformers not available. HF deepfake model disabled: %s", e)
    return _TRANSFORMERS_AVAILABLE


def _check_facenet_pytorch():
    global _FACENET_PYTORCH_AVAILABLE
    if _FACENET_PYTORCH_AVAILABLE is None:
        try:
            import facenet_pytorch  # noqa: F401
            _FACENET_PYTORCH_AVAILABLE = True
        except Exception as e:
            _FACENET_PYTORCH_AVAILABLE = False
            logger.warning("facenet-pytorch not available. MTCNN face detector disabled: %s", e)
    return _FACENET_PYTORCH_AVAILABLE


def _check_mediapipe():
    global _MEDIAPIPE_AVAILABLE
    if _MEDIAPIPE_AVAILABLE is None:
        try:
            import mediapipe  # noqa: F401
            _MEDIAPIPE_AVAILABLE = True
        except Exception as e:
            _MEDIAPIPE_AVAILABLE = False
            logger.warning("MediaPipe not available. Optional face detector disabled: %s", e)
    return _MEDIAPIPE_AVAILABLE


class DeepfakeDetectorAdvanced:
    """
    Advanced Deepfake Detector using ensemble of pre-trained models.
    Supports loading existing .h5 model files and model switching.
    """

    def __init__(self, threshold: float = 0.5, models_dir: str = "models"):
        self.threshold = threshold
        self.models_dir = models_dir
        self.ensemble_models = []
        self.model_names = []
        self.model_paths = []
        self.available_model_specs = []
        self.available_model_names = []
        self.available_model_paths = []
        self.input_size = (224, 224)
        self.model_weights = {}
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.face_cascades = self._load_face_cascades()
        self.best_model_name = None
        auto_threshold_mode = os.getenv("TRUTHGUARD_DF_AUTO_THRESHOLD", "1").strip().lower()
        self._auto_threshold_requested = (
            abs(float(threshold) - 0.5) < 1e-9
            and auto_threshold_mode not in {"0", "false", "off", "disabled"}
        )
        try:
            configured_ensemble_size = int(os.getenv("TRUTHGUARD_DF_ENSEMBLE_SIZE", "2"))
        except ValueError:
            configured_ensemble_size = 2
        self.max_ensemble_local_models = max(1, min(configured_ensemble_size, 3))
        self.enable_hf_model = os.getenv("TRUTHGUARD_ENABLE_HF_DEEPFAKE_MODELS", "0").strip() == "1"
        self.available_hf_models = {
            "HF_Transformer_V1": "prithivMLmods/deepfake-detector-model-v1",
            "HF_Transformer_V2": "prithivMLmods/Deep-Fake-Detector-v2-Model",
            "SDXL_Detector": "Organika/sdxl-detector",
            "AI_Image_Detector": "umm-maybe/AI-image-detector",
        }
        self.enable_pretrained_fallback = os.getenv("TRUTHGUARD_ENABLE_PRETRAINED_FALLBACK", "0").strip() == "1"
        advanced_face_detector_mode = os.getenv(
            "TRUTHGUARD_ENABLE_ADVANCED_FACE_DETECTORS",
            "auto",
        ).strip().lower()
        self.enable_advanced_face_detectors = advanced_face_detector_mode not in {
            "0",
            "false",
            "off",
            "disabled",
        }
        self._hf_models_loaded = False
        self._pretrained_models_loaded = False
        self._existing_models_loaded = False
        self._model_scan_complete = False
        self._failed_model_names = set()
        self.min_face_size = 40
        try:
            configured_max_face_regions = int(os.getenv("TRUTHGUARD_MAX_FACE_REGIONS", "6"))
        except ValueError:
            configured_max_face_regions = 6
        self.max_face_regions = max(1, min(configured_max_face_regions, 12))
        self._mtcnn_face_detector = None
        self._mtcnn_face_detector_loaded = False
        self._mp_face_detector = None
        self._mp_face_detector_loaded = False
        self._last_face_detection_mode = "uninitialized"
        self._last_detected_face_count = 0
        self._last_analyzed_region_count = 0
        self._runtime_lock = threading.RLock()
        self._evaluation_accuracy_by_token = None
        self._evaluation_summary_by_token = None

        os.makedirs(models_dir, exist_ok=True)

        # Load existing .h5 models — TF is imported inside _load_existing_models
        self._scan_available_model_files()
        if self._auto_threshold_requested:
            self.threshold = self.get_recommended_threshold()
        # HuggingFace + pretrained ImageNet are lazy-loaded on first predict call

    # ── .h5 model loading ────────────────────────────────────────────────────
    def _discover_model_files(self) -> List[str]:
        h5_files = []
        for ext in ["*.h5", "*.weights.h5"]:
            h5_files.extend(glob.glob(os.path.join(self.models_dir, ext)))
            h5_files.extend(glob.glob(os.path.join(self.models_dir, "deepfake", ext)))
        return [
            path for path in sorted(set(h5_files))
            if os.path.exists(path) and os.path.getsize(path) > 10240
        ]

    def _scan_available_model_files(self, refresh: bool = False) -> List[Dict[str, str]]:
        """Scan the filesystem for candidate models without importing TensorFlow."""
        if self._model_scan_complete and not refresh:
            return self.available_model_specs

        specs = []
        for model_path in self._discover_model_files():
            model_name = os.path.basename(model_path).replace('.weights.h5', '').replace('.h5', '')
            specs.append({
                'name': model_name,
                'path': model_path,
                'kind': 'weights' if model_path.endswith('.weights.h5') else 'full_model',
            })
        specs = self._rank_model_specs(specs)

        self.available_model_specs = specs
        self.available_model_names = [spec['name'] for spec in specs]
        self.available_model_paths = [spec['path'] for spec in specs]
        self._model_scan_complete = True

        if self.available_model_names:
            self._initialize_smart_weights(self.available_model_names)
            logger.info(f"Discovered {len(self.available_model_names)} deepfake model file(s) for lazy loading.")
        else:
            logger.info("No valid .h5 files found in models directory.")

        return self.available_model_specs

    def _extract_timestamp_token(self, value: str) -> Optional[str]:
        match = re.search(r'(\d{8}_\d{6})', value or "")
        return match.group(1) if match else None

    def _parse_timestamp_token(self, value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y%m%d_%H%M%S")
        except ValueError:
            return None

    def _load_evaluation_summary_map(self) -> Dict[str, Dict[str, float]]:
        if self._evaluation_summary_by_token is not None:
            return self._evaluation_summary_by_token

        evaluation_summaries: Dict[str, Dict[str, float]] = {}
        for result_path in glob.glob(os.path.join(self.models_dir, "evaluation_*.json")):
            token = self._extract_timestamp_token(os.path.basename(result_path))
            if not token:
                continue
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                summary: Dict[str, float] = {}
                accuracy = payload.get("accuracy")
                if accuracy is not None:
                    summary["accuracy"] = float(accuracy)
                recommended_threshold = payload.get("recommended_threshold")
                if recommended_threshold is not None:
                    summary["recommended_threshold"] = float(recommended_threshold)
                threshold_used = payload.get("threshold_used")
                if threshold_used is not None:
                    summary["threshold_used"] = float(threshold_used)
                if summary:
                    evaluation_summaries[token] = summary
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue

        self._evaluation_summary_by_token = evaluation_summaries
        return evaluation_summaries

    def _load_evaluation_accuracy_map(self) -> Dict[str, float]:
        if self._evaluation_accuracy_by_token is not None:
            return self._evaluation_accuracy_by_token

        evaluation_scores = {
            token: float(summary["accuracy"])
            for token, summary in self._load_evaluation_summary_map().items()
            if "accuracy" in summary
        }
        self._evaluation_accuracy_by_token = evaluation_scores
        return evaluation_scores

    def _get_model_accuracy_hint(self, spec: Dict[str, str]) -> float:
        model_token = self._extract_timestamp_token(spec.get('name', ''))
        model_time = self._parse_timestamp_token(model_token) if model_token else None
        if model_time is None:
            return -1.0

        best_accuracy = -1.0
        smallest_positive_delta = None
        for eval_token, accuracy in self._load_evaluation_accuracy_map().items():
            eval_time = self._parse_timestamp_token(eval_token)
            if eval_time is None:
                continue

            delta_seconds = (eval_time - model_time).total_seconds()
            if delta_seconds < 0 or delta_seconds > 18 * 3600:
                continue

            if smallest_positive_delta is None or delta_seconds < smallest_positive_delta:
                smallest_positive_delta = delta_seconds
                best_accuracy = accuracy

        return best_accuracy

    def _get_model_threshold_hint(self, spec: Optional[Dict[str, str]]) -> Optional[float]:
        if not spec:
            return None

        model_token = self._extract_timestamp_token(spec.get('name', ''))
        model_time = self._parse_timestamp_token(model_token) if model_token else None
        if model_time is None:
            return None

        best_threshold = None
        smallest_positive_delta = None
        for eval_token, summary in self._load_evaluation_summary_map().items():
            eval_time = self._parse_timestamp_token(eval_token)
            if eval_time is None or "recommended_threshold" not in summary:
                continue

            delta_seconds = (eval_time - model_time).total_seconds()
            if delta_seconds < 0 or delta_seconds > 18 * 3600:
                continue

            if smallest_positive_delta is None or delta_seconds < smallest_positive_delta:
                smallest_positive_delta = delta_seconds
                best_threshold = float(summary["recommended_threshold"])

        return best_threshold

    def _model_spec_priority(self, spec: Dict[str, str]) -> Tuple[float, int, float, str]:
        """Prefer the strongest local checkpoints when choosing default models."""
        name = spec.get('name', '').lower()
        priority = 0
        if 'best' in name:
            priority += 100
        if 'artifact' in name:
            priority += 40
        if 'mobilenet' in name:
            priority += 24
        if 'efficientnet' in name:
            priority += 20
        if 'cnn' in name:
            priority += 16
        if spec.get('kind') == 'full_model':
            priority += 8
        try:
            modified = os.path.getmtime(spec['path'])
        except OSError:
            modified = 0.0
        accuracy_hint = self._get_model_accuracy_hint(spec)
        return (accuracy_hint, priority, modified, spec.get('name', ''))

    def _rank_model_specs(self, specs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return sorted(specs, key=self._model_spec_priority, reverse=True)

    def _select_preferred_model_spec(self, specs: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        ranked = self._rank_model_specs(specs)
        return ranked[0] if ranked else None

    def _select_default_ensemble_specs(self, specs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Keep the explicit ensemble path small enough for low-memory environments."""
        ranked = self._rank_model_specs(specs)
        return ranked[:self.max_ensemble_local_models]

    def get_preferred_model_name(self) -> Optional[str]:
        if DEFAULT_DEEPFAKE_MODEL_NAME in self.available_hf_models:
            return DEFAULT_DEEPFAKE_MODEL_NAME
        specs = self._scan_available_model_files()
        preferred = self._select_preferred_model_spec(specs)
        if preferred:
            return preferred['name']
        if self.best_model_name:
            return self.best_model_name
        return self.model_names[0] if self.model_names else None

    def get_recommended_threshold(self) -> float:
        if self.get_preferred_model_name() in self.available_hf_models:
            return 0.5
        specs = self._scan_available_model_files()
        preferred = self._select_preferred_model_spec(specs)
        hinted_threshold = self._get_model_threshold_hint(preferred)
        if hinted_threshold is None:
            return float(self.threshold)
        return float(np.clip(hinted_threshold, 0.1, 0.9))

    def _get_loaded_model_subset(self, target_names: Optional[List[str]] = None) -> Tuple[List[str], List[Any]]:
        if not target_names:
            return list(self.model_names), list(self.ensemble_models)

        selected_names: List[str] = []
        selected_models: List[Any] = []
        for name in target_names:
            if name not in self.model_names:
                continue
            model_idx = self.model_names.index(name)
            selected_names.append(name)
            selected_models.append(self.ensemble_models[model_idx])
        return selected_names, selected_models

    def _load_existing_models(self, requested_names: Optional[List[str]] = None):
        """Load existing .h5 / .weights.h5 model files from disk."""
        if not _check_tf():
            logger.warning("TensorFlow unavailable — skipping .h5 model loading.")
            return

        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models import load_model

        try:
            specs = self._scan_available_model_files()
            if requested_names:
                requested_order = {name: idx for idx, name in enumerate(requested_names)}
                requested_set = set(requested_names)
                specs = [
                    spec for spec in specs
                    if spec['name'] in requested_set and spec['name'] not in self._failed_model_names
                ]
                specs = sorted(
                    specs,
                    key=lambda spec: requested_order.get(spec['name'], len(requested_order))
                )
            else:
                specs = [spec for spec in specs if spec['name'] not in self.model_names and spec['name'] not in self._failed_model_names]
                specs = self._select_default_ensemble_specs(specs)

            if not specs:
                if requested_names is None:
                    self._existing_models_loaded = True
                return

            import gc
            loaded_count = 0
            for spec in specs:
                model_name = spec['name']
                model_path = spec['path']

                if model_name in self.model_names or model_name in self._failed_model_names:
                    continue

                try:
                    logger.info(f"Loading model: {model_name}")
                    
                    # Pre-emptive cleanup for 8GB RAM
                    gc.collect()

                    if spec['kind'] == 'weights':
                        loaded = False
                        for arch in ['efficientnet_v2', 'efficientnet_b0']:
                            try:
                                model = self._build_model_for_weights(model_name, arch)
                                model.load_weights(model_path)
                                loaded = True
                                logger.info(f"Loaded weights into {arch}")
                                break
                            except Exception as e:
                                logger.warning(f"Failed {arch}: {e}")
                        if not loaded:
                            logger.error(f"Could not load weights for {model_name}")
                            self._failed_model_names.add(model_name)
                            continue
                    else:
                        model = load_model(model_path, compile=False)

                    self.ensemble_models.append(model)
                    self.model_names.append(model_name)
                    self.model_paths.append(model_path)
                    loaded_count += 1
                    logger.info(f"Registered model: {model_name}")
                    gc.collect()

                except Exception as e:
                    self._failed_model_names.add(model_name)
                    logger.error(f"Error loading {model_path}: {e}")

            if requested_names is None:
                self._existing_models_loaded = True

            if loaded_count:
                logger.info(f"Loaded {loaded_count} model(s) successfully.")
                self._initialize_smart_weights(self.model_names)

        except Exception as e:
            logger.error(f"Error scanning models directory: {e}")

    def _load_hf_models(self, requested_names: Optional[List[str]] = None):
        """Load HuggingFace Transformer models for deepfake detection (lazy)."""
        if not self.enable_hf_model:
            logger.info("HF deepfake models are disabled for stability in this environment.")
            return
        if not _check_transformers() or not _check_torch():
            logger.warning("Transformers/Torch not available — skipping HF models.")
            return

        import gc
        import torch
        from transformers import pipeline as hf_pipeline

        target_names = requested_names or [DEFAULT_DEEPFAKE_MODEL_NAME]
        device = 0 if torch.cuda.is_available() else -1

        for model_key in target_names:
            if model_key in self.model_names or model_key not in self.available_hf_models:
                continue

            model_id = self.available_hf_models[model_key]
            try:
                logger.info(f"Loading HuggingFace model {model_key} ({model_id}) on {'GPU' if device == 0 else 'CPU'}...")
                
                # Pre-emptive cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                hf_pipe = hf_pipeline(
                    "image-classification",
                    model=model_id,
                    device=device,
                )
                self.ensemble_models.append(hf_pipe)
                self.model_names.append(model_key)
                self.model_paths.append(f"huggingface://{model_id}")
                logger.info(f"HuggingFace model {model_key} added to ensemble.")
                gc.collect()
            except Exception as e:
                logger.error(f"Failed to load HuggingFace model {model_key}: {e}")
                self._failed_model_names.add(model_key)

        if self.model_names:
            self._initialize_smart_weights(self.model_names)

    def _ensure_models_ready(
        self,
        requested_model_name: Optional[str] = None,
        include_hf: bool = True,
        prefer_ensemble: bool = False,
    ):
        """Lazy-load saved models on demand and fall back to pretrained features only if needed."""
        with self._runtime_lock:
            specs = self._scan_available_model_files()
            target_names: List[str] = []

            if requested_model_name:
                target_names = [requested_model_name]
            elif specs:
                if prefer_ensemble:
                    target_names = [spec['name'] for spec in self._select_default_ensemble_specs(specs)]
                else:
                    preferred_spec = self._select_preferred_model_spec(specs)
                    if preferred_spec:
                        target_names = [preferred_spec['name']]

            if target_names:
                missing_names = [
                    name for name in target_names
                    if name not in self.model_names and name not in self._failed_model_names
                ]
                if missing_names:
                    self._load_existing_models(missing_names)
                if prefer_ensemble:
                    self._existing_models_loaded = all(
                        name in self.model_names or name in self._failed_model_names
                        for name in target_names
                    )
            elif prefer_ensemble:
                self._existing_models_loaded = True

            if len(self.ensemble_models) == 0 and not self._pretrained_models_loaded:
                self._pretrained_models_loaded = True
                if self.enable_pretrained_fallback:
                    logger.info("No loadable local models found. Loading pretrained ImageNet models on first use...")
                    self._load_pretrained_models()
                else:
                    logger.info("No loadable local models found. Using forensic-only fallback instead of downloading pretrained models.")

            if include_hf and self.enable_hf_model and not self._hf_models_loaded:
                self._hf_models_loaded = True
                self._load_hf_models()

            # Handle explicit HF request if not loaded
            if (
                self.enable_hf_model
                and requested_model_name in self.available_hf_models
                and requested_model_name not in self.model_names
            ):
                self._load_hf_models([requested_model_name])

    def _ensure_hf_loaded(
        self,
        requested_model_name: Optional[str] = None,
        include_hf: bool = True,
        prefer_ensemble: bool = False,
    ):
        self._ensure_models_ready(
            requested_model_name=requested_model_name,
            include_hf=include_hf,
            prefer_ensemble=prefer_ensemble,
        )

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _initialize_smart_weights(self, names: Optional[List[str]] = None):
        target_names = names or self.model_names
        if not target_names:
            return
        preferred_name = self.get_preferred_model_name()
        weights = {}
        self.best_model_name = None
        for name in target_names:
            nl = name.lower()
            if name == preferred_name or 'best' in nl:
                weights[name] = 1.0
                if self.best_model_name is None or name == preferred_name:
                    self.best_model_name = name
            elif 'transformer' in nl or 'hf_' in nl or 'detector' in nl:
                weights[name] = 1.2
            elif 'efficientnet' in nl:
                weights[name] = 0.8
            elif 'resnet' in nl or 'xception' in nl:
                weights[name] = 0.6
            else:
                weights[name] = 0.5
        if self.best_model_name is None:
            if preferred_name in target_names:
                self.best_model_name = preferred_name
            else:
                self.best_model_name = target_names[0]
        total = sum(weights.values())
        if total > 0:
            self.model_weights = dict(weights)
            logger.info(f"Smart weights: {self.model_weights}")
        self.model_reliability = {name: 1.0 for name in target_names}

    def _build_model_for_weights(self, name: str, arch_type: str = 'efficientnet_v2'):
        """Build the architecture used in train_deepfake.py for weight loading."""
        if not _check_tf():
            raise RuntimeError("TensorFlow not available")
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import EfficientNetB0, EfficientNetV2S

        if arch_type == 'efficientnet_v2':
            base = EfficientNetV2S(weights=None, include_top=False,
                                   input_shape=(224, 224, 3), include_preprocessing=True)
        else:
            base = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))

        x = base.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        return models.Model(inputs=base.input, outputs=output)

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    def get_gradcam_heatmap(self, model, img_array, last_conv_layer_name=None):
        if not _check_tf():
            return None
        import tensorflow as tf
        try:
            if not hasattr(model, 'layers') or not hasattr(model, 'get_layer'):
                return None
            if last_conv_layer_name is None:
                for layer in reversed(model.layers):
                    if hasattr(layer, 'output_shape'):
                        shape = layer.output_shape
                        if isinstance(shape, list): shape = shape[0]
                        if len(shape) == 4 and any(k in layer.name.lower() for k in ('conv', 'res', 'block')):
                            last_conv_layer_name = layer.name
                            break
                if last_conv_layer_name is None:
                    last_conv_layer_name = "top_conv"

            try:
                grad_model = tf.keras.models.Model(
                    [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
                )
            except Exception:
                grad_model = tf.keras.models.Model(
                    [model.inputs], [model.get_layer("top_conv").output, model.output]
                )

            with tf.GradientTape() as tape:
                last_conv_out, preds = grad_model(img_array)
                class_channel = preds[:, 0]

            grads = tape.gradient(class_channel, last_conv_out)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            last_conv_out = last_conv_out[0]
            heatmap = last_conv_out @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            max_val = tf.math.reduce_max(heatmap)
            heatmap = tf.maximum(heatmap, 0) / max_val if max_val > 0 else tf.zeros_like(heatmap)
            return heatmap.numpy()
        except Exception as e:
            logger.error(f"Grad-CAM error: {e}")
            return None

    def apply_heatmap(self, heatmap, original_img, alpha=0.4):
        try:
            heatmap_u8 = np.uint8(255 * heatmap)
            jet = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            jet = cv2.resize(jet, (original_img.shape[1], original_img.shape[0]))
            superimposed = jet.astype(np.float32) * alpha + original_img.astype(np.float32)
            return np.clip(superimposed, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f"apply_heatmap error: {e}")
            return original_img

    # ── ELA & FFT ─────────────────────────────────────────────────────────────
    def perform_ela(self, image_array: np.ndarray, quality: int = 90) -> Tuple[np.ndarray, float]:
        try:
            original = Image.fromarray(image_array)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
            try:
                original.save(temp_path, 'JPEG', quality=quality)
                compressed = Image.open(temp_path)
                compressed_arr = np.array(compressed)
                if original.size != compressed.size:
                    compressed = compressed.resize(original.size)
                    compressed_arr = np.array(compressed)
                diff = np.abs(image_array.astype(np.float32) - compressed_arr.astype(np.float32))
                scale = 255.0 / (np.max(diff) if np.max(diff) > 0 else 1)
                ela_img = (diff * scale).astype(np.uint8)
                ela_score = min(float(np.mean(diff) / 255.0 * 10), 1.0)
                return ela_img, ela_score
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logger.error(f"ELA error: {e}")
            return image_array, 0.0

    def perform_fft_analysis(self, image_array: np.ndarray) -> Tuple[np.ndarray, float]:
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
            magnitude_spectrum = np.nan_to_num(magnitude_spectrum, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            magnitude_spectrum -= float(np.min(magnitude_spectrum))
            max_val = float(np.max(magnitude_spectrum))
            if max_val > 1e-6:
                mag_img = np.clip((magnitude_spectrum / max_val) * 255.0, 0, 255).astype(np.uint8)
            else:
                mag_img = np.zeros_like(gray, dtype=np.uint8)
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols), np.uint8)
            r_inner = min(rows, cols) // 8
            cv2.circle(mask, (ccol, crow), r_inner, 0, -1)
            high_freq = np.abs(dft_shift) * mask
            mean_hf = np.mean(high_freq)
            max_hf = np.max(high_freq)
            fft_score = min((max_hf / (mean_hf + 1e-6)) / 150.0, 1.0)
            return mag_img, float(fft_score)
        except Exception as e:
            logger.error(f"FFT error: {e}")
            return np.zeros_like(image_array[:, :, 0]), 0.0

    def _estimate_blur_score(self, image_array: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Low variance means blur; map into [0,1] where higher is more suspicious.
            return float(np.clip(1.0 - (lap_var / 450.0), 0.0, 1.0))
        except Exception:
            return 0.0

    def _estimate_blockiness_score(self, image_array: np.ndarray, block_size: int = 8) -> float:
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if gray.shape[0] < block_size * 2 or gray.shape[1] < block_size * 2:
                return 0.0
            vertical = np.abs(gray[:, block_size::block_size] - gray[:, block_size - 1:-1:block_size])
            horizontal = np.abs(gray[block_size::block_size, :] - gray[block_size - 1:-1:block_size, :])
            edge_strength = (np.mean(vertical) + np.mean(horizontal)) / 2.0
            local_var = np.var(gray) + 1e-6
            return float(np.clip(edge_strength / (np.sqrt(local_var) * 3.5), 0.0, 1.0))
        except Exception:
            return 0.0

    def _estimate_color_misalignment_score(self, image_array: np.ndarray) -> float:
        try:
            b, g, r = cv2.split(image_array.astype(np.float32))
            rg = np.mean(np.abs(r - g))
            gb = np.mean(np.abs(g - b))
            rb = np.mean(np.abs(r - b))
            return float(np.clip((rg + gb + rb) / (255.0 * 0.9), 0.0, 1.0))
        except Exception:
            return 0.0

    def extract_forensic_features(self, image_array: np.ndarray) -> Dict[str, float]:
        _, ela_score = self.perform_ela(image_array)
        _, fft_score = self.perform_fft_analysis(image_array)
        blur_score = self._estimate_blur_score(image_array)
        blockiness_score = self._estimate_blockiness_score(image_array)
        color_score = self._estimate_color_misalignment_score(image_array)
        forensic_score = float(np.clip(
            0.34 * ela_score +
            0.24 * fft_score +
            0.26 * blockiness_score +
            0.10 * color_score +
            0.06 * blur_score,
            0.0, 1.0
        ))
        return {
            "ela_score": float(ela_score),
            "fft_score": float(fft_score),
            "blur_score": float(blur_score),
            "blockiness_score": float(blockiness_score),
            "color_score": float(color_score),
            "forensic_score": forensic_score,
        }

    # ── Pretrained fallback ───────────────────────────────────────────────────
    def _load_pretrained_models(self):
        if not _check_tf():
            self._load_fallback_models()
            return
        try:
            import gc
            from tensorflow.keras.applications import ResNet50, EfficientNetB0, Xception
            
            gc.collect()
            logger.info("Loading ResNet50...")
            self.ensemble_models.append(ResNet50(weights='imagenet', include_top=False, pooling='avg'))
            self.model_names.append('ResNet50')
            self.model_paths.append('pretrained')
            logger.info("Loading EfficientNetB0...")
            self.ensemble_models.append(EfficientNetB0(weights='imagenet', include_top=False, pooling='avg'))
            self.model_names.append('EfficientNet')
            self.model_paths.append('pretrained')
            logger.info("Loading Xception...")
            self.ensemble_models.append(Xception(weights='imagenet', include_top=False, pooling='avg'))
            self.model_names.append('Xception')
            self.model_paths.append('pretrained')
            logger.info(f"Loaded {len(self.ensemble_models)} pretrained models.")
            self._initialize_smart_weights()
        except Exception as e:
            logger.error(f"Error loading pretrained models: {e}")
            self._load_fallback_models()

    def _load_fallback_models(self):
        if not _check_tf():
            return
        try:
            from tensorflow.keras.applications import MobileNetV2
            self.ensemble_models.append(MobileNetV2(weights='imagenet', include_top=False, pooling='avg'))
            self.model_names.append('MobileNetV2')
            self.model_paths.append('pretrained')
            logger.info("Loaded MobileNetV2 as fallback.")
        except Exception as e:
            logger.error(f"Failed to load fallback models: {e}")

    # ── Image preprocessing ───────────────────────────────────────────────────
    def set_model_weights(self, weights: Dict[str, float]):
        self.model_weights = weights

    def preprocess_for_model(self, image_array: np.ndarray, model: Any = None, model_name: str = "") -> np.ndarray:
        input_shape = self.input_size
        if model and hasattr(model, 'input_shape'):
            shape = model.input_shape
            if isinstance(shape, list): shape = shape[0]
            if shape and len(shape) >= 3:
                input_shape = (shape[1], shape[2])

        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = image_array

        img_resized = cv2.resize(img_rgb, input_shape)
        m_name = model_name.lower()
        if model and hasattr(model, 'name'):
            m_name += "_" + model.name.lower()

        if 'efficientnet' in m_name:
            img_normalized = img_resized.astype(np.float32)
        elif 'mobilenet' in m_name or 'resnet' in m_name:
            img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
        else:
            img_normalized = img_resized.astype(np.float32) / 255.0

        return np.expand_dims(img_normalized, axis=0)

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict_with_model(self, model: Any, image_array: np.ndarray, model_name: str = "") -> Optional[float]:
        try:
            is_hf = (
                any(k in model_name for k in ('HF_', 'Detector', 'Transformer'))
                or 'huggingface' in str(type(model)).lower()
                or 'transformers' in str(type(model)).lower()
                or not hasattr(model, "predict")
            )

            if is_hf:
                rgb_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                # HuggingFace pipelines are callable
                results = model(pil_img)
                score = 0.5
                for res in results:
                    lbl = res['label'].lower()
                    if lbl == 'fake':
                        score = res['score']; break
                    elif lbl == 'real':
                        score = 1.0 - res['score']
                return float(score)

            image_batch = self.preprocess_for_model(image_array, model, model_name)

            # Check if model expects a list of inputs (often true for some Keras architectures)
            # to avoid structure warnings.
            try:
                if hasattr(model, 'predict'):
                    # Some models (like EfficientNet) might trigger structural warnings if not wrapped
                    # in a list depending on how they were exported/loaded.
                    pred = model.predict([image_batch], verbose=0)
                else:
                    logger.error(f"Model {model_name} has no predict method and was not identified as HF.")
                    return None
            except TypeError:
                # Fallback if verbose=0 is not supported or if list input fails
                pred = model.predict(image_batch)
            if isinstance(pred, list): pred = pred[0]
            if len(pred.shape) > 2: pred = pred.flatten()
            if pred.shape[-1] == 2:
                return float(pred[0][1])
            elif len(pred) == 1:
                return float(pred[0])
            elif len(pred.shape) == 1 and pred.shape[0] > 1:
                # Handle possible classification output [prob_real, prob_fake] or similar
                return float(pred[1]) if pred.shape[0] >= 2 else float(pred[0])
            return float(np.mean(pred))
        except Exception as e:
            logger.error(f"Prediction error ({model_name}): {e}")
            return None

    def _load_face_cascades(self) -> List[Tuple[str, Any]]:
        cascades: List[Tuple[str, Any]] = []
        for label, filename in [
            ("frontal_default", "haarcascade_frontalface_default.xml"),
            ("frontal_alt2", "haarcascade_frontalface_alt2.xml"),
            ("profile", "haarcascade_profileface.xml"),
        ]:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
            if cascade is not None and not cascade.empty():
                cascades.append((label, cascade))
        if not cascades and self.face_cascade is not None and not self.face_cascade.empty():
            cascades.append(("frontal_default", self.face_cascade))
        return cascades

    def _detection_variants(self, gray: np.ndarray) -> List[np.ndarray]:
        variants = [gray]
        try:
            variants.append(cv2.equalizeHist(gray))
        except Exception:
            pass
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            variants.append(clahe.apply(gray))
        except Exception:
            pass

        unique: List[np.ndarray] = []
        seen = set()
        for variant in variants:
            signature = (variant.shape, int(np.mean(variant)))
            if signature not in seen:
                seen.add(signature)
                unique.append(variant)
        return unique

    def _normalize_face_box(
        self,
        box: Tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        x, y, w, h = map(int, box)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        if w < 12 or h < 12:
            return None
        aspect = w / max(h, 1)
        if aspect < 0.45 or aspect > 1.8:
            return None
        return (x, y, w, h)

    def _boxes_match_same_face(
        self,
        candidate: Tuple[int, int, int, int],
        existing: Tuple[int, int, int, int],
    ) -> bool:
        x, y, w, h = candidate
        ex, ey, ew, eh = existing
        inter_x1 = max(x, ex)
        inter_y1 = max(y, ey)
        inter_x2 = min(x + w, ex + ew)
        inter_y2 = min(y + h, ey + eh)
        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if inter <= 0:
            return False

        area = max(w * h, 1)
        existing_area = max(ew * eh, 1)
        union = area + existing_area - inter + 1e-6
        iou = inter / union
        coverage = inter / max(min(area, existing_area), 1)

        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)
        existing_center_x = ex + (ew / 2.0)
        existing_center_y = ey + (eh / 2.0)
        center_dx = abs(center_x - existing_center_x) / max((w + ew) / 2.0, 1.0)
        center_dy = abs(center_y - existing_center_y) / max((h + eh) / 2.0, 1.0)
        size_ratio = max(area, existing_area) / max(min(area, existing_area), 1)

        # Cascade detectors often jitter the same face across neighboring scales.
        # Treat moderately overlapping boxes as one face so duplicate detections
        # do not inflate the reported face count.
        return (
            iou >= 0.24
            or coverage >= 0.50
            or (center_dx <= 0.30 and center_dy <= 0.30 and size_ratio <= 3.0)
        )

    def _merge_face_boxes(
        self,
        first: Tuple[int, int, int, int],
        second: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        x1, y1, w1, h1 = first
        x2, y2, w2, h2 = second
        area1 = max(w1 * h1, 1)
        area2 = max(w2 * h2, 1)
        total_area = area1 + area2
        return (
            int(round(((x1 * area1) + (x2 * area2)) / total_area)),
            int(round(((y1 * area1) + (y2 * area2)) / total_area)),
            int(round(((w1 * area1) + (w2 * area2)) / total_area)),
            int(round(((h1 * area1) + (h2 * area2)) / total_area)),
        )

    def _cluster_face_detections(
        self,
        detections: List[Tuple[int, int, int, int]],
    ) -> List[Dict[str, Any]]:
        clusters: List[Dict[str, Any]] = []
        for candidate in sorted(detections, key=lambda box: box[2] * box[3], reverse=True):
            matched_idx = None
            for idx, cluster in enumerate(clusters):
                if self._boxes_match_same_face(candidate, cluster["box"]):
                    matched_idx = idx
                    break

            if matched_idx is None:
                clusters.append({"box": candidate, "hits": 1})
            else:
                clusters[matched_idx]["box"] = self._merge_face_boxes(
                    clusters[matched_idx]["box"],
                    candidate,
                )
                clusters[matched_idx]["hits"] += 1
        return clusters

    def _merge_duplicate_face_boxes(
        self,
        detections: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        return [cluster["box"] for cluster in self._cluster_face_detections(detections)]

    def _build_face_consensus(
        self,
        primary: List[Tuple[int, int, int, int]],
        secondary: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        if not primary or not secondary:
            return []

        consensus: List[Tuple[int, int, int, int]] = []
        used_secondary = set()
        for candidate in primary:
            match_idx = None
            for idx, other in enumerate(secondary):
                if idx in used_secondary:
                    continue
                if self._boxes_match_same_face(candidate, other):
                    match_idx = idx
                    break
            if match_idx is None:
                continue
            used_secondary.add(match_idx)
            consensus.append(self._merge_face_boxes(candidate, secondary[match_idx]))

        return self._merge_duplicate_face_boxes(consensus)

    def _filter_cascade_face_boxes(
        self,
        detections: List[Tuple[int, int, int, int]],
        image_shape: Tuple[int, ...],
    ) -> List[Tuple[int, int, int, int]]:
        if not detections:
            return []

        height, width = image_shape[:2]
        image_area = float(max(height * width, 1))
        filtered: List[Tuple[int, int, int, int]] = []
        clusters = self._cluster_face_detections(detections)

        for cluster in sorted(
            clusters,
            key=lambda item: (item["hits"], item["box"][2] * item["box"][3]),
            reverse=True,
        ):
            box = cluster["box"]
            support = int(cluster["hits"])
            area_ratio = (box[2] * box[3]) / image_area

            if area_ratio < 0.004 or area_ratio > 0.68:
                continue
            if support <= 1 and area_ratio < 0.015:
                continue
            if support <= 2 and area_ratio < 0.008 and len(clusters) >= self.max_face_regions:
                continue

            filtered.append(box)

        return filtered

    def _select_advanced_face_boxes(
        self,
        image_array: np.ndarray,
    ) -> Tuple[str, List[Tuple[int, int, int, int]]]:
        candidates: List[Tuple[str, List[Tuple[int, int, int, int]]]] = []

        mtcnn_boxes = self._detect_faces_mtcnn(image_array)
        if mtcnn_boxes:
            candidates.append(("mtcnn", mtcnn_boxes))

        mediapipe_boxes = self._detect_faces_mediapipe(image_array)
        if mediapipe_boxes:
            candidates.append(("mediapipe", mediapipe_boxes))

        if not candidates:
            return ("", [])

        if mtcnn_boxes and mediapipe_boxes:
            consensus_boxes = self._build_face_consensus(mtcnn_boxes, mediapipe_boxes)
            if consensus_boxes:
                return ("advanced_consensus", consensus_boxes)

        reasonable_limit = max(self.max_face_regions + 2, 8)
        reasonable_candidates = [
            candidate for candidate in candidates
            if len(candidate[1]) <= reasonable_limit
        ]
        if reasonable_candidates:
            candidates = reasonable_candidates

        preferred_mode, preferred_boxes = max(
            candidates,
            key=lambda item: (len(item[1]), 1 if item[0] == "mtcnn" else 0),
        )
        return (preferred_mode, preferred_boxes)

    def _face_count_metadata(self) -> Dict[str, int]:
        return {
            'face_count': int(max(self._last_detected_face_count, 0)),
            'analyzed_region_count': int(max(self._last_analyzed_region_count, 0)),
        }

    def _fallback_face_boxes(self, image_shape: Tuple[int, ...]) -> List[Tuple[int, int, int, int]]:
        height, width = image_shape[:2]
        if height < 24 or width < 24:
            return []

        boxes: List[Tuple[int, int, int, int]] = []
        portrait_size = int(min(width, height) * 0.62)
        portrait_x = max(0, (width - portrait_size) // 2)
        portrait_y = max(0, min(height - portrait_size, int(height * 0.12)))
        boxes.append((portrait_x, portrait_y, portrait_size, portrait_size))

        upper_size = int(min(width, height) * 0.5)
        upper_x = max(0, (width - upper_size) // 2)
        upper_y = max(0, min(height - upper_size, int(height * 0.05)))
        boxes.append((upper_x, upper_y, upper_size, upper_size))

        normalized: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            clean = self._normalize_face_box(box, width, height)
            if clean is not None:
                normalized.append(clean)
        return normalized[:2]

    def _ensure_mtcnn_face_detector(self):
        """Lazy-load facenet-pytorch MTCNN for more reliable multi-face detection."""
        if self._mtcnn_face_detector_loaded:
            return self._mtcnn_face_detector

        self._mtcnn_face_detector_loaded = True
        if not (_check_torch() and _check_facenet_pytorch()):
            return None

        try:
            from facenet_pytorch import MTCNN

            self._mtcnn_face_detector = MTCNN(
                keep_all=True,
                device='cpu',
                min_face_size=max(20, self.min_face_size),
                post_process=False,
            )
        except Exception as e:
            logger.warning(f"MTCNN face detector unavailable: {e}")
            self._mtcnn_face_detector = None
        return self._mtcnn_face_detector

    def _detect_faces_mtcnn(self, image_array: np.ndarray) -> List[Tuple[int, int, int, int]]:
        detector = self._ensure_mtcnn_face_detector()
        if detector is None or image_array is None or image_array.size == 0:
            return []

        try:
            if len(image_array.shape) == 2:
                rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            boxes, probs = detector.detect(Image.fromarray(rgb))
        except Exception as e:
            logger.debug(f"MTCNN face detection failed: {e}")
            return []

        if boxes is None or probs is None:
            return []

        height, width = rgb.shape[:2]
        image_area = float(max(height * width, 1))
        detections: List[Tuple[int, int, int, int]] = []

        for box, prob in zip(boxes, probs):
            if box is None:
                continue
            score = float(prob) if prob is not None else 0.0
            if score < 0.86:
                continue

            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            clean = self._normalize_face_box((x1, y1, x2 - x1, y2 - y1), width, height)
            if clean is None:
                continue

            area_ratio = (clean[2] * clean[3]) / image_area
            if area_ratio < 0.003 or area_ratio > 0.68:
                continue

            detections.append(clean)

        return self._merge_duplicate_face_boxes(detections)

    def _ensure_mediapipe_face_detector(self):
        """Lazy-load MediaPipe face detection so import-time startup stays fast."""
        if self._mp_face_detector_loaded:
            return self._mp_face_detector

        self._mp_face_detector_loaded = True
        if not _check_mediapipe():
            return None

        try:
            import mediapipe as mp

            self._mp_face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.60,
            )
        except Exception as e:
            logger.warning(f"MediaPipe face detector unavailable: {e}")
            self._mp_face_detector = None
        return self._mp_face_detector

    def _detect_faces_mediapipe(self, image_array: np.ndarray) -> List[Tuple[int, int, int, int]]:
        detector = self._ensure_mediapipe_face_detector()
        if detector is None or image_array is None or image_array.size == 0:
            return []

        try:
            if len(image_array.shape) == 2:
                rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
        except Exception as e:
            logger.debug(f"MediaPipe face detection failed: {e}")
            return []

        if results is None or not getattr(results, "detections", None):
            return []

        height, width = rgb.shape[:2]
        detections: List[Tuple[int, int, int, int]] = []
        image_area = float(max(height * width, 1))

        for detection in results.detections:
            score = float(detection.score[0]) if getattr(detection, "score", None) else 0.0
            if score < 0.55:
                continue

            rel_box = detection.location_data.relative_bounding_box
            x = int(round(rel_box.xmin * width))
            y = int(round(rel_box.ymin * height))
            w = int(round(rel_box.width * width))
            h = int(round(rel_box.height * height))

            clean = self._normalize_face_box((x, y, w, h), width, height)
            if clean is None:
                continue

            area_ratio = (clean[2] * clean[3]) / image_area
            if area_ratio < 0.004 or area_ratio > 0.68:
                continue

            detections.append(clean)

        return self._merge_duplicate_face_boxes(detections)

    def _run_cascade_face_search(
        self,
        gray: np.ndarray,
        cascades: List[Tuple[str, Any]],
    ) -> List[Tuple[int, int, int, int]]:
        if gray is None or gray.size == 0 or not cascades:
            return []

        height, width = gray.shape[:2]
        dynamic_min_face = max(24, min(self.min_face_size, max(24, int(min(height, width) * 0.08))))
        detections: List[Tuple[int, int, int, int]] = []
        configs = [
            (1.03, 3),
            (1.06, 4),
            (1.1, 5),
        ]
        scales = [1.0, 1.35, 1.7] if min(height, width) < 720 else [1.0, 1.25]

        for variant in self._detection_variants(gray):
            for cascade_name, cascade in cascades:
                for scale in scales:
                    if scale != 1.0:
                        scaled = cv2.resize(
                            variant,
                            None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_CUBIC,
                        )
                    else:
                        scaled = variant

                    scaled_min_face = max(int(dynamic_min_face * scale), 20)
                    for scale_factor, min_neighbors in configs:
                        try:
                            faces = cascade.detectMultiScale(
                                scaled,
                                scaleFactor=scale_factor,
                                minNeighbors=min_neighbors,
                                minSize=(scaled_min_face, scaled_min_face),
                            )
                        except Exception:
                            faces = []

                        for face in faces:
                            x, y, w, h = map(int, face)
                            if scale != 1.0:
                                x = int(round(x / scale))
                                y = int(round(y / scale))
                                w = int(round(w / scale))
                                h = int(round(h / scale))
                            clean = self._normalize_face_box((x, y, w, h), width, height)
                            if clean is not None:
                                detections.append(clean)

                        if cascade_name == "profile":
                            flipped = cv2.flip(scaled, 1)
                            try:
                                mirrored_faces = cascade.detectMultiScale(
                                    flipped,
                                    scaleFactor=scale_factor,
                                    minNeighbors=min_neighbors,
                                    minSize=(scaled_min_face, scaled_min_face),
                                )
                            except Exception:
                                mirrored_faces = []

                            for face in mirrored_faces:
                                x, y, w, h = map(int, face)
                                mirrored_x = scaled.shape[1] - x - w
                                if scale != 1.0:
                                    mirrored_x = int(round(mirrored_x / scale))
                                    y = int(round(y / scale))
                                    w = int(round(w / scale))
                                    h = int(round(h / scale))
                                clean = self._normalize_face_box((mirrored_x, y, w, h), width, height)
                                if clean is not None:
                                    detections.append(clean)

        return self._filter_cascade_face_boxes(detections, gray.shape)

    def detect_faces(self, image_array: np.ndarray) -> List[Tuple[int, int, int, int]]:
        self._last_detected_face_count = 0
        self._last_analyzed_region_count = 0
        if image_array is None or image_array.size == 0:
            self._last_face_detection_mode = "missing_image"
            return []

        if len(image_array.shape) == 2:
            gray = image_array
        else:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        frontal_cascades = [item for item in self.face_cascades if item[0] != "profile"]
        profile_cascades = [item for item in self.face_cascades if item[0] == "profile"]
        cascade_faces = self._run_cascade_face_search(gray, frontal_cascades or self.face_cascades)
        if not cascade_faces and profile_cascades:
            cascade_faces = self._run_cascade_face_search(gray, profile_cascades)

        selected_faces = cascade_faces
        selected_mode = "cascade" if cascade_faces else ""
        should_retry_with_advanced = self.enable_advanced_face_detectors and (
            not cascade_faces or len(cascade_faces) > 1
        )
        if should_retry_with_advanced:
            advanced_mode, advanced_faces = self._select_advanced_face_boxes(image_array)
            if advanced_faces and (
                not cascade_faces
                or len(cascade_faces) > self.max_face_regions
                or advanced_mode == "advanced_consensus"
                or len(advanced_faces) < len(cascade_faces)
            ):
                selected_faces = advanced_faces
                selected_mode = advanced_mode

        if selected_faces:
            self._last_face_detection_mode = selected_mode or "cascade"
            self._last_detected_face_count = len(selected_faces)
        else:
            selected_faces = self._fallback_face_boxes(image_array.shape)
            self._last_face_detection_mode = "fallback_crop" if selected_faces else "none"

        selected_faces.sort(key=lambda box: box[2] * box[3], reverse=True)
        limited = selected_faces[:self.max_face_regions]
        self._last_analyzed_region_count = len(limited)
        return limited

    def _expand_face_bbox(self, bbox: Tuple[int, int, int, int], image_shape: Tuple[int, ...], margin: float = 0.2) -> Tuple[int, int, int, int]:
        x, y, w, h = bbox
        mw, mh = int(w * margin), int(h * margin)
        y1 = max(0, y - mh)
        y2 = min(image_shape[0], y + h + mh)
        x1 = max(0, x - mw)
        x2 = min(image_shape[1], x + w + mw)
        return x1, y1, x2, y2

    def _aggregate_face_scores(self, face_results: List[Dict]) -> Tuple[float, float]:
        if not face_results:
            return 0.0, 0.0
        scores = np.array([r['score'] for r in face_results], dtype=np.float32)
        areas = np.array([r['bbox'][2] * r['bbox'][3] for r in face_results], dtype=np.float32)
        if np.sum(areas) <= 0:
            weighted_mean = float(np.mean(scores))
        else:
            weighted_mean = float(np.average(scores, weights=areas))
        max_score = float(np.max(scores))
        consensus = 1.0 - min(float(np.std(scores)), 1.0)
        # Blend weighted consensus with strongest suspicious face.
        ensemble_score = float(np.clip(0.7 * weighted_mean + 0.3 * max_score, 0.0, 1.0))
        return ensemble_score, consensus

    def _resolve_video_sample_interval(self, capture: Any, sample_rate: Optional[int] = None) -> int:
        if sample_rate is not None:
            try:
                return max(1, int(sample_rate))
            except (TypeError, ValueError):
                return 1

        fps = 0.0
        if capture is not None and hasattr(capture, 'get'):
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
            except Exception:
                fps = 0.0

        if fps > 1.0:
            # Aim for roughly 3 analyzed frames per second by default.
            return max(1, int(round(fps / 3.0)))
        return 10

    def _forensic_only_result(
        self,
        image_array: np.ndarray,
        faces: Optional[List[Tuple[int, int, int, int]]] = None,
        reason: str = "",
    ) -> Dict:
        """Fallback path when trained models cannot score the image."""
        faces = faces or []
        ela_img, ela_score = self.perform_ela(image_array)
        fft_img, fft_score = self.perform_fft_analysis(image_array)
        full_forensics = self.extract_forensic_features(image_array)

        target_bbox = None
        active_forensics = full_forensics
        if faces:
            target_bbox = max(faces, key=lambda box: box[2] * box[3])
            x1, y1, x2, y2 = self._expand_face_bbox(target_bbox, image_array.shape)
            region = image_array[y1:y2, x1:x2]
            if region.size > 0:
                active_forensics = self.extract_forensic_features(region)

        heuristic_score = float(np.clip(
            0.62 * active_forensics['forensic_score'] +
            0.38 * full_forensics['forensic_score'],
            0.0,
            1.0,
        ))
        decision_threshold = max(self.threshold, 0.58)
        is_deepfake = heuristic_score > decision_threshold
        confidence = float(np.clip(
            55.0 + abs(heuristic_score - decision_threshold) * 90.0,
            55.0,
            78.0,
        ))

        focus = "detected face region" if target_bbox else "full frame"
        detail = reason or "Trained deepfake models were unavailable."
        return {
            'is_deepfake': is_deepfake,
            'confidence': confidence,
            'face_detection_mode': self._last_face_detection_mode,
            'ensemble_score': heuristic_score,
            'consistency': 0.45 if target_bbox else 0.35,
            'model_scores': {},
            'models_used': ['forensic_fallback'],
            'heatmap': None,
            'status': 'FORENSIC_ONLY',
            'message': f"{detail} Using forensic-only analysis on the {focus}.",
            'ela_image': ela_img,
            'ela_score': ela_score,
            'fft_image': fft_img,
            'fft_score': fft_score,
            'blur_score': active_forensics['blur_score'],
            'blockiness_score': active_forensics['blockiness_score'],
            'color_score': active_forensics['color_score'],
            'forensic_score': active_forensics['forensic_score'],
            'target_face_bbox': target_bbox,
            **self._face_count_metadata(),
        }

    def detect_deepfake_ensemble(
        self,
        image_array: np.ndarray,
        return_heatmap: bool = False,
        requested_models: Optional[List[str]] = None,
    ) -> Dict:
        try:
            with self._runtime_lock:
                # If requested_models includes HF models, ensure they are loaded
                if requested_models:
                    hf_targets = [m for m in requested_models if m in self.available_hf_models]
                    if hf_targets:
                        if self.enable_hf_model:
                            self._load_hf_models(hf_targets)
                        else:
                            logger.warning("Ignoring HF deepfake requests because HF deepfake models are disabled.")

                self._ensure_hf_loaded(include_hf=False, prefer_ensemble=True)

                if requested_models:
                    ensemble_target_names = requested_models
                else:
                    ensemble_target_names = [
                        spec['name'] for spec in self._select_default_ensemble_specs(self.available_model_specs)
                    ]

                active_model_names, active_models = self._get_loaded_model_subset(
                    ensemble_target_names or None
                )
                faces = self.detect_faces(image_array)
                face_count = self._last_detected_face_count
                analyzed_region_count = self._last_analyzed_region_count

                if not active_models:
                    return self._forensic_only_result(
                        image_array,
                        faces,
                        "TensorFlow-backed deepfake models are unavailable in this environment.",
                    )

                if not faces:
                    return self._forensic_only_result(
                        image_array,
                        [],
                        "No faces were confidently detected.",
                    )

                ela_img, ela_score = self.perform_ela(image_array)
                fft_img, fft_score = self.perform_fft_analysis(image_array)
                full_forensics = self.extract_forensic_features(image_array)
                face_results = []

                for f_idx, (x, y, w, h) in enumerate(faces):
                    x1, y1, x2, y2 = self._expand_face_bbox((x, y, w, h), image_array.shape)
                    face_crop = image_array[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    model_scores = {}
                    valid_scores = []
                    for i, model in enumerate(active_models):
                        name = active_model_names[i]
                        score = self.predict_with_model(model, face_crop, name)
                        if score is not None:
                            model_scores[name] = float(score)
                            valid_scores.append(score)

                    if not valid_scores:
                        continue

                    adjusted_weights = {}
                    for name, score in model_scores.items():
                        bw = self.model_weights.get(name, 1.0)
                        certainty = abs(score - 0.5) * 2.0
                        adjusted_weights[name] = bw * 0.1 if 0.495 <= score <= 0.505 else bw * (0.5 + certainty)

                    wsum = sum(score * adjusted_weights[n] for n, score in model_scores.items())
                    wtot = sum(adjusted_weights.values())
                    model_score = wsum / wtot if wtot > 0 else float(np.mean(valid_scores))
                    face_forensics = self.extract_forensic_features(face_crop)
                    face_area_ratio = (w * h) / max(image_array.shape[0] * image_array.shape[1], 1)
                    area_weight = float(np.clip(face_area_ratio * 8.0, 0.15, 1.0))
                    face_score = float(np.clip(
                        (0.74 * model_score) + (0.26 * face_forensics['forensic_score'] * area_weight),
                        0.0, 1.0
                    ))

                    face_results.append({'id': f_idx, 'score': face_score,
                                         'model_scores': model_scores, 'bbox': (x, y, w, h),
                                         'face_crop': face_crop,
                                         'forensics': face_forensics,
                                         'area_weight': area_weight})

                if not face_results:
                    return self._forensic_only_result(
                        image_array,
                        faces,
                        "Loaded models could not score the detected face crops.",
                    )

                best = max(face_results, key=lambda x: x['score'])
                ensemble_score, consistency = self._aggregate_face_scores(face_results)
                # Let strong global forensics influence uncertain cases without overwhelming model evidence.
                ensemble_score = float(np.clip(
                    0.82 * ensemble_score + 0.18 * full_forensics['forensic_score'],
                    0.0, 1.0
                ))
                is_deepfake = ensemble_score > self.threshold
                confidence = ensemble_score * 100 if is_deepfake else (1 - ensemble_score) * 100

                heatmap_img = None
                if return_heatmap and active_models:
                    heatmap_model = None
                    heatmap_name = ""
                    best_idx = active_model_names.index(self.best_model_name) if self.best_model_name in active_model_names else 0
                    candidate = active_models[best_idx]
                    if hasattr(candidate, 'layers') and hasattr(candidate, 'get_layer'):
                        heatmap_model = candidate
                        heatmap_name = active_model_names[best_idx]
                    else:
                        for i, m in enumerate(active_models):
                            if hasattr(m, 'layers') and hasattr(m, 'get_layer'):
                                heatmap_model = m
                                heatmap_name = active_model_names[i]
                                break
                    if heatmap_model:
                        try:
                            prep = self.preprocess_for_model(best['face_crop'], heatmap_model, heatmap_name)
                            hmap = self.get_gradcam_heatmap(heatmap_model, prep)
                            if hmap is not None:
                                heatmap_img = self.apply_heatmap(hmap, best['face_crop'])
                        except Exception as e:
                            logger.error(f"Heatmap generation failed: {e}")

                import gc
                gc.collect()

                return {
                    'is_deepfake': is_deepfake, 'confidence': float(confidence),
                    'face_count': face_count, 'ensemble_score': float(ensemble_score),
                    'analyzed_region_count': analyzed_region_count,
                    'face_detection_mode': self._last_face_detection_mode,
                    'consistency': float(consistency),
                    'model_scores': best['model_scores'],
                    'all_face_scores': [float(r['score']) for r in face_results],
                    'face_forensics': best.get('forensics', {}),
                    'models_used': active_model_names,
                    'message': self._get_message(is_deepfake, confidence),
                    'heatmap': heatmap_img, 'ela_image': ela_img, 'ela_score': ela_score,
                    'fft_image': fft_img, 'fft_score': fft_score,
                    'blur_score': full_forensics['blur_score'],
                    'blockiness_score': full_forensics['blockiness_score'],
                    'color_score': full_forensics['color_score'],
                    'forensic_score': full_forensics['forensic_score'],
                    'target_face_bbox': best['bbox'],
                }
        except Exception as e:
            logger.error(f"Ensemble detection error: {e}")
            return {
                'is_deepfake': False, 'confidence': 0.0, 'face_count': 0,
                'ensemble_score': 0.0, 'consistency': 0.0, 'model_scores': {},
                'models_used': self.model_names, 'message': f'Error: {e}',
            }

    def detect_with_single_model(
        self,
        image_array: np.ndarray,
        model_name: str,
        return_heatmap: bool = False,
        include_visual_artifacts: bool = True,
    ) -> Dict:
        try:
            with self._runtime_lock:
                self._ensure_hf_loaded(
                    requested_model_name=model_name,
                    include_hf=(model_name in self.available_hf_models and self.enable_hf_model),
                )
                faces = self.detect_faces(image_array)
                face_count = self._last_detected_face_count
                analyzed_region_count = self._last_analyzed_region_count
                if model_name not in self.model_names:
                    return self._forensic_only_result(
                        image_array,
                        faces,
                        f"Requested model {model_name} could not be loaded.",
                    )

                model_idx = self.model_names.index(model_name)
                model = self.ensemble_models[model_idx]

                if not faces:
                    return self._forensic_only_result(
                        image_array,
                        [],
                        "No faces were confidently detected.",
                    )

                # Evaluate the largest detected face instead of the whole frame.
                x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                x1, y1, x2, y2 = self._expand_face_bbox((x, y, w, h), image_array.shape)
                face_crop = image_array[y1:y2, x1:x2]
                score = self.predict_with_model(model, face_crop, model_name)
                if score is None:
                    return self._forensic_only_result(
                        image_array,
                        faces,
                        f"Model {model_name} could not score the detected face crop.",
                    )
                forensics = self.extract_forensic_features(face_crop)
                ela_img = fft_img = None
                ela_score = float(forensics.get('ela_score', 0.0))
                fft_score = float(forensics.get('fft_score', 0.0))
                if include_visual_artifacts:
                    ela_img, ela_score = self.perform_ela(image_array)
                    fft_img, fft_score = self.perform_fft_analysis(image_array)
                score = float(np.clip(0.8 * score + 0.2 * forensics['forensic_score'], 0.0, 1.0))
                is_deepfake = score > self.threshold
                confidence = score * 100 if is_deepfake else (1 - score) * 100

                heatmap_img = None
                if return_heatmap and hasattr(model, 'layers'):
                    prep = self.preprocess_for_model(face_crop, model, model_name)
                    hmap = self.get_gradcam_heatmap(model, prep)
                    if hmap is not None:
                        heatmap_img = self.apply_heatmap(hmap, face_crop)

                return {
                    'is_deepfake': is_deepfake, 'confidence': float(confidence),
                    'face_count': face_count, 'ensemble_score': float(score),
                    'analyzed_region_count': analyzed_region_count,
                    'face_detection_mode': self._last_face_detection_mode,
                    'consistency': 1.0, 'model_scores': {model_name: float(score)},
                    'model_used': model_name, 'models_used': [model_name],
                    'message': self._get_message(is_deepfake, confidence),
                    'heatmap': heatmap_img,
                    'ela_image': ela_img,
                    'fft_image': fft_img,
                    'target_face_bbox': (x, y, w, h),
                    'forensic_score': forensics['forensic_score'],
                    'ela_score': float(ela_score),
                    'fft_score': float(fft_score),
                    'blur_score': forensics['blur_score'],
                    'blockiness_score': forensics['blockiness_score'],
                    'color_score': forensics['color_score'],
                }
        except Exception as e:
            logger.error(f"Single model detection error: {e}")
            return {'is_deepfake': False, 'confidence': 0.0, 'face_count': 0,
                    'ensemble_score': 0.0, 'consistency': 1.0,
                    'message': f'Error: {e}', 'model_used': model_name,
                    'models_used': [model_name] if model_name else []}

    def detect_deepfake_video_advanced(
        self,
        video_path: str,
        sample_rate: Optional[int] = None,
        requested_models: Optional[List[str]] = None,
        return_heatmap: bool = False,
    ) -> Dict:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Could not open video file'}

            frame_results = []
            frame_count = 0
            worst_frame = None
            max_score = -1
            sample_interval = self._resolve_video_sample_interval(cap, sample_rate)

            while True:
                ret, frame = cap.read()
                if not ret: break
                if frame_count % sample_interval == 0:
                    if requested_models:
                        result = self.detect_deepfake_ensemble(frame, requested_models=requested_models)
                    else:
                        result = self.detect_deepfake_ensemble(frame)
                    score = result['ensemble_score']
                    frame_results.append({'frame': frame_count, 'is_deepfake': result['is_deepfake'],
                                          'confidence': result['confidence'], 'ensemble_score': float(score)})
                    if score > max_score:
                        max_score = score
                        worst_frame = frame.copy()
                frame_count += 1
            cap.release()

            if not frame_results:
                return {'error': 'No frames could be processed'}

            forensic = {}
            if worst_frame is not None:
                if requested_models:
                    forensic = self.detect_deepfake_ensemble(
                        worst_frame,
                        return_heatmap=return_heatmap,
                        requested_models=requested_models,
                    )
                else:
                    forensic = self.detect_deepfake_ensemble(worst_frame, return_heatmap=return_heatmap)

            scores = [r['ensemble_score'] for r in frame_results]
            deepfake_frames = sum(1 for r in frame_results if r['is_deepfake'])
            avg_conf = float(np.mean([r['confidence'] for r in frame_results]))
            avg_score = float(np.mean(scores))
            ratio = deepfake_frames / len(frame_results)
            temporal_peak = float(max(scores))
            temporal_consistency = float(np.mean([1.0 if s > self.threshold else 0.0 for s in scores]))
            video_score = float(np.clip(0.5 * avg_score + 0.35 * ratio + 0.15 * temporal_peak, 0.0, 1.0))
            is_deepfake = video_score > max(self.threshold, 0.42)
            models_used = requested_models if requested_models else forensic.get('models_used', [])

            result = {
                'is_deepfake': is_deepfake, 'confidence': max(avg_conf, video_score * 100 if is_deepfake else (1 - video_score) * 100),
                'deepfake_ratio': float(ratio),
                'total_frames_analyzed': len(frame_results),
                'frame_stride': int(sample_interval),
                'avg_ensemble_score': avg_score,
                'ensemble_score': float(video_score),
                'temporal_scores': scores,
                'flicker_score': float(np.std(scores)),
                'temporal_peak_score': temporal_peak,
                'temporal_consistency': temporal_consistency,
                'message': self._get_video_message(is_deepfake, ratio, max(avg_conf, video_score * 100)),
                'models_used': models_used,
            }
            for key in ['heatmap', 'ela_image', 'ela_score', 'fft_image', 'fft_score',
                        'target_face_bbox', 'face_count', 'model_scores']:
                if key in forensic:
                    result[key] = forensic[key]
            return result
        except Exception as e:
            logger.error(f"Video analysis error: {e}")
            return {'error': str(e)}

    def detect_video_with_single_model(
        self,
        video_path: str,
        model_name: str,
        sample_rate: Optional[int] = None,
        return_heatmap: bool = False,
    ) -> Dict:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Could not open video file'}

            frame_results = []
            frame_count = 0
            worst_frame = None
            max_score = -1.0
            sample_interval = self._resolve_video_sample_interval(cap, sample_rate)
            while True:
                ret, frame = cap.read()
                if not ret: break
                if frame_count % sample_interval == 0:
                    result = self.detect_with_single_model(
                        frame,
                        model_name,
                        return_heatmap=False,
                        include_visual_artifacts=False,
                    )
                    frame_results.append({'frame': frame_count, 'is_deepfake': result['is_deepfake'],
                                          'confidence': result['confidence'], 'ensemble_score': result['ensemble_score']})
                    if result['ensemble_score'] > max_score:
                        max_score = float(result['ensemble_score'])
                        worst_frame = frame.copy()
                frame_count += 1
            cap.release()

            if not frame_results:
                return {'error': 'No frames could be processed'}

            forensic = {}
            if worst_frame is not None:
                forensic = self.detect_with_single_model(
                    worst_frame,
                    model_name,
                    return_heatmap=return_heatmap,
                    include_visual_artifacts=True,
                )

            ratio = sum(1 for r in frame_results if r['is_deepfake']) / len(frame_results)
            avg_conf = float(np.mean([r['confidence'] for r in frame_results]))
            result = {
                'is_deepfake': ratio > 0.3, 'confidence': avg_conf,
                'deepfake_ratio': float(ratio), 'total_frames_analyzed': len(frame_results),
                'frame_stride': int(sample_interval),
                'avg_ensemble_score': float(np.mean([r['ensemble_score'] for r in frame_results])),
                'ensemble_score': float(np.mean([r['ensemble_score'] for r in frame_results])),
                'model_used': model_name, 'models_used': [model_name],
                'message': self._get_video_message(ratio > 0.3, ratio, avg_conf),
            }
            for key in [
                'heatmap', 'ela_image', 'ela_score', 'fft_image', 'fft_score',
                'target_face_bbox', 'face_count', 'model_scores', 'forensic_score',
                'blur_score', 'blockiness_score', 'color_score',
                'face_detection_mode', 'analyzed_region_count',
            ]:
                if key in forensic:
                    result[key] = forensic[key]
            return result
        except Exception as e:
            logger.error(f"Single model video analysis error: {e}")
            return {'error': str(e)}

    def get_model_info(self) -> Dict:
        self._scan_available_model_files()
        local_names = self.available_model_names if self.available_model_names else [n for n in self.model_names if n not in self.available_hf_models]
        hf_names = list(self.available_hf_models.keys()) if self.enable_hf_model else []
        visible_names = list(dict.fromkeys(local_names + hf_names))

        return {
            'total_models': len(visible_names),
            'loaded_models': len(self.model_names),
            'model_names': visible_names,
            'local_models': local_names,
            'hf_models': hf_names,
            'preferred_model_name': self.get_preferred_model_name(),
            'threshold': self.threshold,
            'recommended_threshold': self.get_recommended_threshold(),
            'weights': {name: self.model_weights.get(name, 1.0) for name in visible_names},
        }

    def _get_message(self, is_deepfake: bool, confidence: float) -> str:
        if is_deepfake:
            if confidence > 85: return f"🚨 CRITICAL: High-confidence deepfake detection ({confidence:.1f}%)."
            if confidence > 65: return f"⚠️ SUSPICIOUS: Significant manipulation indicators detected ({confidence:.1f}%)."
            return f"ℹ️ LOW CONFIDENCE: Minor anomalies detected ({confidence:.1f}%)."
        else:
            if confidence > 85: return f"✅ AUTHENTIC: High-confidence verification ({confidence:.1f}%)."
            if confidence > 65: return f"✅ LIKELY AUTHENTIC: No major manipulation found ({confidence:.1f}%)."
            return f"😐 UNCERTAIN: Lacks strong indicators of either outcome ({confidence:.1f}%)."

    def _get_video_message(self, is_deepfake: bool, ratio: float, confidence: float) -> str:
        if is_deepfake:
            return f"🚨 FAKE VIDEO: Manipulation in {ratio*100:.1f}% of frames (Confidence: {confidence:.1f}%)"
        return f"✅ AUTHENTIC VIDEO: Content appears genuine (Deepfake indicators in {ratio*100:.1f}% of frames)"


# ── Lazy proxy singleton ──────────────────────────────────────────────────────
# Importing this module now costs almost nothing. TensorFlow is not touched
# until _load_existing_models() fires on the first analysis request.
class _LazyDeepfakeDetector:
    """Proxy that defers DeepfakeDetectorAdvanced construction until first use."""
    _instance = None

    def _get(self):
        if self._instance is None:
            object.__setattr__(self, '_instance', DeepfakeDetectorAdvanced(models_dir="models"))
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def __setattr__(self, name, value):
        if name == '_instance':
            object.__setattr__(self, name, value)
        else:
            setattr(self._get(), name, value)


deepfake_detector = _LazyDeepfakeDetector()
