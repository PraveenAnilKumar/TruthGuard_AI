"""
Deepfake Model Training Script for TruthGuard AI
Trains your existing .h5 models on new data to improve accuracy
"""

import numpy as np
import cv2
import os
import glob
import inspect
import math
import argparse
import logging
import warnings
from datetime import datetime
from typing import Tuple, List, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    MobileNetV2, EfficientNetB0, EfficientNetV2S, ResNet50V2,
    mobilenet_v2, xception, resnet_v2, efficientnet
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Named preprocessing functions (lambdas cannot be pickled by Keras callbacks)
def preprocess_efficientnet(x):
    """EfficientNet handles scaling internally - pass raw 0-255 pixels."""
    return x

def preprocess_resnet(x):
    return resnet_v2.preprocess_input(x)

def preprocess_mobilenet(x):
    return mobilenet_v2.preprocess_input(x)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import pandas as pd
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepfakeTrainer:
    """
    Handles training and fine-tuning of deepfake detection models
    """
    
    def __init__(self, models_dir: str = "models", data_dir: str = "datasets/deepfake"):
        """
        Initialize trainer
        
        Args:
            models_dir: Directory to save trained models
            data_dir: Directory containing training data
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.input_size = (224, 224)  # Balanced resolution for 8GB RAM
        self.batch_size = 8           # Optimized for usable memory
        self.epochs = 30              # More epochs for convergence
        self.steps_per_epoch = 1000   # Default; overridden by main() args
        self.validation_steps = 256   # Default cap for faster per-epoch feedback
        self.model_type = 'efficientnet_v2'  # Default; set by build_model()
        self.model = None
        self.history = None
        self.base_layer_count = 0
        self.recommended_threshold = 0.5
        self.validation_threshold_metrics = {}
        self.threshold_profiles = {}
        self.threshold_sweep = []
        self.label_smoothing = 0.02
        self.threshold_min = 0.15
        self.threshold_max = 0.85
        self.threshold_step = 0.01
        self.max_recall_drop_for_balanced = 0.05
        self.min_precision_gain_for_balanced = 0.03
        self.fine_tune_fraction = 0.22
        self.min_fine_tune_layers = 24
        self.max_fine_tune_layers = 140
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "train/real"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "train/fake"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "val/real"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "val/fake"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "test/real"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "test/fake"), exist_ok=True)
        
        logger.info(f"Trainer initialized. Models will be saved to {models_dir}")

    def _get_preprocessing_function(self):
        """Return the correct preprocessing function for the active backbone."""
        m_type = self.model_type if hasattr(self, 'model_type') else 'efficientnet'
        if 'mobilenet' in m_type:
            return preprocess_mobilenet
        if 'resnet' in m_type:
            return preprocess_resnet
        return preprocess_efficientnet

    def _count_split_files(self, split_name: str) -> Dict[str, int]:
        counts = {}
        for label in ['real', 'fake']:
            label_dir = os.path.join(self.data_dir, split_name, label)
            if not os.path.isdir(label_dir):
                counts[label] = 0
                continue
            counts[label] = sum(
                1 for entry in os.scandir(label_dir)
                if entry.is_file()
            )
        return counts

    def _has_explicit_validation_split(self) -> bool:
        counts = self._count_split_files('val')
        return all(counts.get(label, 0) > 0 for label in ['real', 'fake'])

    def _resolve_steps(self, generator, limit: int = None) -> int:
        sample_count = max(int(getattr(generator, 'samples', 0)), 0)
        batch_size = max(int(getattr(generator, 'batch_size', self.batch_size)), 1)
        full_steps = max(1, math.ceil(sample_count / batch_size))
        if limit is None or limit <= 0:
            return full_steps
        return min(full_steps, int(limit))

    def _allocate_epoch_budget(self, fine_tune: bool) -> Tuple[int, int]:
        """Split the total epoch budget across Stage 1 and Stage 2."""
        total_epochs = max(int(self.epochs), 1)
        if not fine_tune:
            return total_epochs, 0
        if total_epochs == 1:
            return 1, 0
        stage1_epochs = min(10, max(1, math.ceil(total_epochs / 2)))
        stage2_epochs = max(0, total_epochs - stage1_epochs)
        return stage1_epochs, stage2_epochs

    def _build_validation_monitor_generator(self, reference_generator):
        """Create a shuffled validation iterator for capped per-epoch monitoring."""
        datagen = getattr(reference_generator, "image_data_generator", None)
        directory = getattr(reference_generator, "directory", None)
        if datagen is None or not directory:
            logger.warning(
                "Unable to clone the validation generator for shuffled monitoring; "
                "falling back to the original validation iterator."
            )
            return reference_generator

        class_indices = getattr(reference_generator, "class_indices", {}) or {}
        classes = None
        if class_indices:
            classes = [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]

        flow_kwargs = {
            "target_size": getattr(reference_generator, "target_size", self.input_size),
            "batch_size": max(int(getattr(reference_generator, "batch_size", self.batch_size)), 1),
            "class_mode": getattr(reference_generator, "class_mode", "binary"),
            "shuffle": True,
        }
        if classes is not None:
            flow_kwargs["classes"] = classes

        for attr_name in ["color_mode", "data_format", "interpolation", "keep_aspect_ratio"]:
            attr_value = getattr(reference_generator, attr_name, None)
            if attr_value is not None:
                flow_kwargs[attr_name] = attr_value

        subset = getattr(reference_generator, "subset", None)
        if subset is not None:
            flow_kwargs["subset"] = subset

        follow_links = getattr(reference_generator, "follow_links", None)
        if follow_links is not None:
            flow_kwargs["follow_links"] = follow_links

        signature_params = inspect.signature(datagen.flow_from_directory).parameters
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature_params.values()
        )
        if not accepts_var_kwargs:
            supported_params = set(signature_params.keys())
            supported_params.discard("directory")
            flow_kwargs = {
                key: value
                for key, value in flow_kwargs.items()
                if key in supported_params
            }
        return datagen.flow_from_directory(directory, **flow_kwargs)

    def _build_metrics(self):
        return [
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(curve='PR', name='pr_auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]

    def _build_loss(self):
        return keras.losses.BinaryCrossentropy(label_smoothing=self.label_smoothing)

    def _apply_forensic_augmentations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply mild artifact-preserving augmentations.

        Deepfake cues often live in compression noise, resampling, and subtle
        texture artifacts, so these transforms are intentionally gentler than a
        generic image-classification recipe.
        """
        augmented = np.asarray(image, dtype=np.float32).copy()
        if augmented.ndim != 3 or augmented.shape[-1] != 3:
            return np.clip(augmented, 0.0, 255.0).astype(np.float32)

        height, width = augmented.shape[:2]

        if min(height, width) >= 48 and np.random.rand() < 0.30:
            scale = float(np.random.uniform(0.72, 0.95))
            target_w = max(32, int(round(width * scale)))
            target_h = max(32, int(round(height * scale)))
            reduced = cv2.resize(augmented, (target_w, target_h), interpolation=cv2.INTER_AREA)
            upsample_interp = cv2.INTER_CUBIC if scale < 0.82 else cv2.INTER_LINEAR
            augmented = cv2.resize(reduced, (width, height), interpolation=upsample_interp).astype(np.float32)

        if np.random.rand() < 0.35:
            quality = int(np.random.randint(55, 96))
            bgr = cv2.cvtColor(np.clip(augmented, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is not None:
                    augmented = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32)

        if np.random.rand() < 0.20:
            if np.random.rand() < 0.5:
                augmented = cv2.GaussianBlur(
                    augmented,
                    (3, 3),
                    sigmaX=float(np.random.uniform(0.1, 1.0)),
                )
            else:
                augmented = cv2.medianBlur(np.clip(augmented, 0, 255).astype(np.uint8), 3).astype(np.float32)

        if np.random.rand() < 0.28:
            noise_std = float(np.random.uniform(2.0, 10.0))
            augmented += np.random.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)

        if np.random.rand() < 0.18:
            contrast = float(np.random.uniform(0.92, 1.12))
            brightness = float(np.random.uniform(-8.0, 8.0))
            gamma = float(np.random.uniform(0.9, 1.1))
            augmented = np.clip((augmented * contrast) + brightness, 0.0, 255.0)
            augmented = 255.0 * np.power(np.clip(augmented / 255.0, 0.0, 1.0), gamma)

        return np.clip(augmented, 0.0, 255.0).astype(np.float32)

    def _apply_training_preprocessing(self, image: np.ndarray) -> np.ndarray:
        augmented = self._apply_forensic_augmentations(image)
        return self._get_preprocessing_function()(augmented)

    def _summarize_threshold_metrics(
        self,
        y_true: np.ndarray,
        y_pred_prob: np.ndarray,
        threshold: float,
    ) -> Dict[str, float]:
        y_pred = (y_pred_prob >= threshold).astype(np.int32)
        precision_fake = precision_score(y_true, y_pred, zero_division=0)
        recall_fake = recall_score(y_true, y_pred, zero_division=0)
        precision_real = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        recall_real = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1_fake = f1_score(y_true, y_pred, zero_division=0)
        f2_fake = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = (float(recall_real) + float(recall_fake)) / 2.0
        return {
            'threshold': float(threshold),
            'accuracy': float(accuracy),
            'precision_fake': float(precision_fake),
            'recall_fake': float(recall_fake),
            'precision_real': float(precision_real),
            'recall_real': float(recall_real),
            'f1_fake': float(f1_fake),
            'f2_fake': float(f2_fake),
            'balanced_accuracy': float(balanced_accuracy),
            'balanced_objective': float(
                (0.40 * f1_fake) +
                (0.30 * balanced_accuracy) +
                (0.20 * precision_fake) +
                (0.10 * recall_fake)
            ),
            'precision_objective': float(
                (0.55 * precision_fake) +
                (0.25 * f1_fake) +
                (0.20 * balanced_accuracy)
            ),
        }

    def _format_threshold_summary(self, summary: Dict[str, float]) -> Dict[str, float]:
        return {
            'recommended_threshold': float(summary['threshold']),
            'validation_accuracy': float(summary['accuracy']),
            'validation_precision_fake': float(summary['precision_fake']),
            'validation_recall_fake': float(summary['recall_fake']),
            'validation_precision_real': float(summary['precision_real']),
            'validation_recall_real': float(summary['recall_real']),
            'validation_f1_fake': float(summary['f1_fake']),
            'validation_f2_fake': float(summary['f2_fake']),
            'validation_balanced_accuracy': float(summary['balanced_accuracy']),
        }

    def _select_threshold_profile(
        self,
        summaries: List[Dict[str, float]],
        profile_name: str,
    ) -> Dict[str, float]:
        if profile_name == 'recall_first':
            return max(
                summaries,
                key=lambda s: (
                    s['f2_fake'],
                    s['f1_fake'],
                    s['recall_fake'],
                    s['balanced_accuracy'],
                    -abs(s['threshold'] - 0.35),
                ),
            )

        if profile_name == 'balanced':
            candidates = [
                summary for summary in summaries
                if summary['precision_fake'] >= 0.72 and summary['recall_fake'] >= 0.75
            ] or summaries
            return max(
                candidates,
                key=lambda s: (
                    s['balanced_objective'],
                    s['f1_fake'],
                    s['precision_fake'],
                    s['recall_real'],
                    s['recall_fake'],
                    -abs(s['threshold'] - 0.45),
                ),
            )

        if profile_name == 'precision_first':
            candidates = [
                summary for summary in summaries
                if summary['precision_fake'] >= 0.85
            ] or summaries
            return max(
                candidates,
                key=lambda s: (
                    s['precision_objective'],
                    s['precision_fake'],
                    s['f1_fake'],
                    s['recall_fake'],
                    -abs(s['threshold'] - 0.60),
                ),
            )

        raise ValueError(f"Unknown threshold profile: {profile_name}")

    def _configure_backbone_fine_tuning(self) -> int:
        if self.base_layer_count <= 0:
            return 0

        fine_tune_layers = int(round(self.base_layer_count * self.fine_tune_fraction))
        fine_tune_layers = max(self.min_fine_tune_layers, fine_tune_layers)
        fine_tune_layers = min(self.max_fine_tune_layers, fine_tune_layers, self.base_layer_count)
        fine_tune_start = max(0, self.base_layer_count - fine_tune_layers)

        unfrozen_layers = 0
        for index, layer in enumerate(self.model.layers[:self.base_layer_count]):
            should_train = index >= fine_tune_start and not isinstance(layer, layers.BatchNormalization)
            layer.trainable = should_train
            unfrozen_layers += int(should_train)

        for layer in self.model.layers[self.base_layer_count:]:
            layer.trainable = True

        return unfrozen_layers

    def _run_with_legacy_generator_warning_suppressed(self, fn, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Your `PyDataset` class should call `super\(\).__init__\(\*\*kwargs\)` in its constructor\.",
                category=UserWarning,
            )
            return fn(*args, **kwargs)

    def _collect_predictions(self, generator) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(generator, 'reset'):
            generator.reset()
        steps = self._resolve_steps(generator)
        y_pred_prob = self._run_with_legacy_generator_warning_suppressed(
            self.model.predict,
            generator,
            steps=steps,
            verbose=0,
        ).astype(np.float32).reshape(-1)
        y_true = np.asarray(generator.classes[:len(y_pred_prob)], dtype=np.int32)
        return y_true, y_pred_prob

    def tune_threshold(self, val_generator) -> Dict[str, float]:
        """
        Tune the decision threshold on validation data with a recall-weighted objective.

        Returns:
            Summary dict for the selected threshold.
        """
        y_true, y_pred_prob = self._collect_predictions(val_generator)

        if y_true.size == 0 or len(np.unique(y_true)) < 2:
            logger.warning("Validation threshold tuning skipped because the validation split lacks both classes.")
            self.recommended_threshold = 0.5
            self.threshold_profiles = {}
            self.threshold_sweep = []
            self.validation_threshold_metrics = {
                'recommended_threshold': 0.5,
                'selection_metric': 'fallback_default',
                'recommended_profile': 'default',
                'threshold_profiles': {},
                'threshold_sweep': [],
            }
            return dict(self.validation_threshold_metrics)

        thresholds = np.arange(
            self.threshold_min,
            self.threshold_max + (self.threshold_step / 2.0),
            self.threshold_step,
        )
        threshold_summaries = [
            self._summarize_threshold_metrics(y_true, y_pred_prob, round(float(threshold), 2))
            for threshold in thresholds
        ]

        recall_first = self._select_threshold_profile(threshold_summaries, 'recall_first')
        balanced = self._select_threshold_profile(threshold_summaries, 'balanced')
        precision_first = self._select_threshold_profile(threshold_summaries, 'precision_first')

        recommended_profile = 'recall_first'
        selection_metric = 'validation_f2_fake'
        recommended_summary = recall_first

        recall_drop = float(recall_first['recall_fake'] - balanced['recall_fake'])
        precision_gain = float(balanced['precision_fake'] - recall_first['precision_fake'])
        specificity_gain = float(balanced['recall_real'] - recall_first['recall_real'])
        if (
            recall_drop <= self.max_recall_drop_for_balanced
            and (
                precision_gain >= self.min_precision_gain_for_balanced
                or specificity_gain >= 0.08
                or balanced['f1_fake'] >= recall_first['f1_fake'] + 0.02
            )
            and balanced['f1_fake'] >= recall_first['f1_fake'] - 0.015
        ):
            recommended_profile = 'balanced'
            selection_metric = 'balanced_validation_tradeoff'
            recommended_summary = balanced

        threshold_profiles = {
            'recall_first': self._format_threshold_summary(recall_first),
            'balanced': self._format_threshold_summary(balanced),
            'precision_first': self._format_threshold_summary(precision_first),
        }
        threshold_profiles['recall_first']['profile'] = 'recall_first'
        threshold_profiles['balanced']['profile'] = 'balanced'
        threshold_profiles['precision_first']['profile'] = 'precision_first'

        formatted_summary = self._format_threshold_summary(recommended_summary)
        formatted_summary['selection_metric'] = selection_metric
        formatted_summary['recommended_profile'] = recommended_profile
        formatted_summary['threshold_profiles'] = threshold_profiles
        formatted_summary['threshold_sweep'] = threshold_summaries

        self.recommended_threshold = float(formatted_summary['recommended_threshold'])
        self.threshold_profiles = threshold_profiles
        self.threshold_sweep = threshold_summaries
        self.validation_threshold_metrics = dict(formatted_summary)
        logger.info(
            "Recommended threshold %.2f selected from validation using %s "
            "(F2=%.4f, fake recall=%.4f, fake precision=%.4f, real recall=%.4f).",
            self.recommended_threshold,
            recommended_profile,
            formatted_summary['validation_f2_fake'],
            formatted_summary['validation_recall_fake'],
            formatted_summary['validation_precision_fake'],
            formatted_summary['validation_recall_real'],
        )
        logger.info(
            "Threshold profiles: recall_first=%.2f, balanced=%.2f, precision_first=%.2f.",
            threshold_profiles['recall_first']['recommended_threshold'],
            threshold_profiles['balanced']['recommended_threshold'],
            threshold_profiles['precision_first']['recommended_threshold'],
        )
        return dict(formatted_summary)
    
    def prepare_data_generators(self, validation_split: float = 0.2) -> Tuple:
        """
        Prepare data generators with augmentation for training
        
        Args:
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Tuple of (train_generator, val_generator)
        """
        logger.info("Preparing data generators with augmentation...")
        preprocess_fn = self._get_preprocessing_function()

        train_datagen_kwargs = dict(
            rotation_range=12,
            width_shift_range=0.08,
            height_shift_range=0.08,
            shear_range=0.08,
            zoom_range=(0.90, 1.15),
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.92, 1.08],
            fill_mode='nearest',
            preprocessing_function=self._apply_training_preprocessing,
        )
        val_datagen_kwargs = dict(preprocessing_function=preprocess_fn)

        explicit_val_split = self._has_explicit_validation_split()
        if not explicit_val_split:
            train_datagen_kwargs['validation_split'] = validation_split
            val_datagen_kwargs['validation_split'] = validation_split
            logger.warning(
                "No explicit validation split found in %s. Falling back to validation_split on the train folder; "
                "for video-derived frames, a dedicated val/ split is recommended to reduce leakage.",
                self.data_dir,
            )
        else:
            val_counts = self._count_split_files('val')
            logger.info(
                "Using explicit validation split from %s (real=%s, fake=%s).",
                os.path.join(self.data_dir, 'val'),
                val_counts.get('real', 0),
                val_counts.get('fake', 0),
            )

        train_datagen = ImageDataGenerator(**train_datagen_kwargs)
        val_datagen = ImageDataGenerator(**val_datagen_kwargs)

        common_kwargs = dict(
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['real', 'fake'],
        )

        if explicit_val_split:
            train_generator = train_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'train'),
                shuffle=True,
                **common_kwargs,
            )
            val_generator = val_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'val'),
                shuffle=False,
                **common_kwargs,
            )
        else:
            train_generator = train_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'train'),
                subset='training',
                shuffle=True,
                **common_kwargs,
            )
            val_generator = val_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'train'),
                subset='validation',
                shuffle=False,
                **common_kwargs,
            )

        logger.info(f"Found {train_generator.samples} training samples, {val_generator.samples} validation samples")
        logger.info(f"Classes: {train_generator.class_indices}")
        logger.info("Using artifact-preserving augmentation tuned for forensic robustness.")
        
        return train_generator, val_generator
    
    def prepare_test_data(self) -> tf.keras.preprocessing.image.DirectoryIterator:
        """
        Prepare test data generator (no augmentation)
        
        Returns:
            Test data generator
        """
        test_datagen_kwargs = {
            'preprocessing_function': self._get_preprocessing_function(),
        }
        test_datagen = ImageDataGenerator(**test_datagen_kwargs)
        
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
            classes=['real', 'fake']
        )
        
        logger.info(f"Found {test_generator.samples} test samples")
        return test_generator
    
    def build_model(self, model_type: str = 'efficientnet', load_existing: str = None):
        """
        Build or load a model for training
        
        Args:
            model_type: Type of model to build ('efficientnet', 'mobilenet', 'resnet', 'custom')
            load_existing: Path to existing .h5 or .weights.h5 model to fine-tune
        """
        # If loading a FULL legacy model file
        if load_existing and os.path.exists(load_existing) and not load_existing.endswith('.weights.h5'):
            logger.info(f"Loading existing model structure and weights from {load_existing}...")
            self.model = keras.models.load_model(load_existing, compile=False)
            self.model_type = model_type  # record for data-generator preprocessing
            
            # Check if model needs to be recompiled with proper output layer
            if self.model.output_shape[-1] != 1:
                logger.info("Adapting model for binary classification...")
                x = self.model.layers[-2].output if len(self.model.layers) > 1 else self.model.output
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dropout(0.5)(x)
                x = layers.Dense(128, activation='relu')(x)
                x = layers.Dropout(0.3)(x)
                output = layers.Dense(1, activation='sigmoid', name='new_output')(x)
                self.model = models.Model(inputs=self.model.input, outputs=output)
            
            logger.info("Full model loaded successfully")
            return

        # Otherwise, BUILD the architecture first
        logger.info(f"Building {model_type} architecture...")
        self.model_type = model_type  # record so data generators use correct preprocessing
        
        if model_type == 'efficientnet_v2':
            logger.info("Building State-of-the-art EfficientNetV2S model...")
            base_model = EfficientNetV2S(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3),
                include_preprocessing=True # Internal V2 preprocessing
            )
        elif model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
        elif model_type == 'mobilenet':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
        elif model_type == 'resnet_v2':
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.input_size, 3)
            )
        else:  # custom CNN
            self.model = self._build_custom_cnn()
            self.base_layer_count = 0 # No frozen backbone
            # Skip architectural head addition for custom model
            base_model = None
        
        if base_model is not None:
            # Add refined classification head to pretrained models
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.BatchNormalization(momentum=0.9)(x) # Optimized for small batch size 8
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            output = layers.Dense(1, activation='sigmoid', name='deepfake_output')(x)

            self.model = models.Model(inputs=base_model.input, outputs=output)
            self.base_layer_count = len(base_model.layers)
            
            # Stage 1: Freeze base model to protect pretrained weights
            # BN layers should technically stay in training mode but for binary fake detect,
            # freezing everything in the backbone is a safer Stage 1.
            for layer in self.model.layers[:self.base_layer_count]:
                layer.trainable = False
            
        logger.info(f"Model architecture built for Stability (Base frozen, BN momentum 0.9).")
        
        # Finally, if we were requested to load weights into this architecture
        if load_existing and os.path.exists(load_existing) and load_existing.endswith('.weights.h5'):
            logger.info(f"Loading weights into architecture from {load_existing}...")
            self.model.load_weights(load_existing)
            logger.info("Weights loaded successfully.")
    
    def _build_custom_cnn(self) -> models.Model:
        """Build a custom CNN architecture"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.input_size, 3)),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def train(self, train_generator, val_generator, fine_tune: bool = False):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            fine_tune: Whether to fine-tune all layers
        """
        # Compile model with lower LR for stability
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
            loss=self._build_loss(),
            metrics=self._build_metrics(),
        )

        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(self.models_dir, f'deepfake_best_{timestamp}.weights.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_pr_auc',
            save_best_only=True,
            save_weights_only=True, # Saving only weights is much safer for pickling
            mode='max',
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_pr_auc',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode='max',
            verbose=1,
        )
        early_stopping = EarlyStopping(
            monitor='val_pr_auc',
            patience=4,
            min_delta=0.001,
            mode='max',
            restore_best_weights=False,
            verbose=1,
        )
        callbacks = [checkpoint, reduce_lr, early_stopping]
        
        # Calculate class weights to handle imbalance
        labels = train_generator.classes
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        # Ensure weights are plain Python types (ints and floats)
        class_weights_dict = {int(k): float(v) for k, v in enumerate(weights)}
        logger.info(f"Using class weights: {class_weights_dict}")

        train_steps = self._resolve_steps(train_generator, self.steps_per_epoch)
        full_val_steps = self._resolve_steps(val_generator)
        val_steps = self._resolve_steps(val_generator, self.validation_steps)
        stage1_epochs, stage2_epochs = self._allocate_epoch_budget(fine_tune)
        logger.info(
            "Training with %s train steps/epoch and %s validation steps/epoch.",
            train_steps,
            val_steps,
        )
        if val_steps < full_val_steps:
            logger.info(
                "Validation is capped at %s/%s steps for faster feedback. "
                "Use --validation-steps 0 to evaluate the full validation split each epoch.",
                val_steps,
                full_val_steps,
            )
        else:
            logger.info("Validation will use the full split each epoch (%s steps).", full_val_steps)
        logger.info(
            "Epoch budget: stage1=%s, stage2=%s, total=%s.",
            stage1_epochs,
            stage2_epochs,
            stage1_epochs + stage2_epochs,
        )
        validation_monitor = val_generator
        if val_steps < full_val_steps:
            validation_monitor = self._build_validation_monitor_generator(val_generator)
            if validation_monitor is not val_generator:
                logger.info(
                    "Using a shuffled validation monitor generator so capped validation steps cover both classes."
                )

        # Stage 1: Train only the top layers
        logger.info(
            "Stage 1: Training top layers for %s epoch(s) with up to %s steps/epoch...",
            stage1_epochs,
            self.steps_per_epoch,
        )
        history1 = self._run_with_legacy_generator_warning_suppressed(
            self.model.fit,
            train_generator,
            steps_per_epoch=train_steps,
            validation_data=validation_monitor,
            validation_steps=val_steps,
            epochs=stage1_epochs,
            callbacks=callbacks,
            class_weight=class_weights_dict,
            verbose=1,
        )
        
        # Reload best weights from checkpoint file (avoids pickle issues)
        if os.path.exists(checkpoint_path):
            logger.info(f"Reloading best Stage 1 weights from {checkpoint_path}")
            self.model.load_weights(checkpoint_path)
        completed_stage1_epochs = len(history1.history.get('loss', []))
        
        # Stage 2: Fine-tune all layers (if requested)
        if fine_tune and stage2_epochs > 0:
            logger.info("Stage 2: Fine-tuning started (focused unfreezing)...")
            unfrozen_layers = self._configure_backbone_fine_tuning()
            logger.info(
                "Fine-tuning %s backbone layer(s) plus the classification head.",
                unfrozen_layers,
            )
            
            # Recompile with very low learning rate
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.00003),
                loss=self._build_loss(),
                metrics=self._build_metrics(),
            )
            
            history2 = self._run_with_legacy_generator_warning_suppressed(
                self.model.fit,
                train_generator,
                steps_per_epoch=train_steps,
                validation_data=validation_monitor,
                validation_steps=val_steps,
                initial_epoch=completed_stage1_epochs,
                epochs=completed_stage1_epochs + stage2_epochs,
                callbacks=callbacks,
                class_weight=class_weights_dict,
                verbose=1,
            )
            
            # Combine histories
            self.history = {}
            history2_data = history2.history if history2 is not None else {}
            for key in set(history1.history.keys()) | set(history2_data.keys()):
                self.history[key] = history1.history.get(key, []) + history2_data.get(key, [])
        else:
            if fine_tune and stage2_epochs == 0:
                logger.info("Stage 2 skipped because the total epoch budget was exhausted in Stage 1.")
            self.history = history1.history

        if os.path.exists(checkpoint_path):
            logger.info(f"Reloading best overall validation weights from {checkpoint_path}")
            self.model.load_weights(checkpoint_path)
        
        # Save final model
        final_path = os.path.join(self.models_dir, f'deepfake_trained_{timestamp}.weights.h5')
        self.model.save_weights(final_path)
        logger.info(f"Final weights saved to {final_path}")
        self.tune_threshold(val_generator)
    
    def evaluate(self, test_generator, threshold: float = None) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            test_generator: Test data generator
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model on test data...")
        active_threshold = float(self.recommended_threshold if threshold is None else threshold)

        # Get predictions
        y_true, y_pred_prob = self._collect_predictions(test_generator)
        y_pred = (y_pred_prob >= active_threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'], output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        try:
            auc_roc = float(roc_auc_score(y_true, y_pred_prob))
        except ValueError:
            auc_roc = None
        try:
            auc_pr = float(average_precision_score(y_true, y_pred_prob))
        except ValueError:
            auc_pr = None
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        if auc_roc is not None:
            logger.info(f"ROC-AUC: {auc_roc:.4f}")
        if auc_pr is not None:
            logger.info(f"PR-AUC: {auc_pr:.4f}")
        logger.info(f"Precision (Fake): {report['Fake']['precision']:.4f}")
        logger.info(f"Recall (Fake): {report['Fake']['recall']:.4f}")
        logger.info(f"F1-Score (Fake): {report['Fake']['f1-score']:.4f}")
        logger.info(f"Threshold used: {active_threshold:.2f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Save evaluation results
        results = {
            'accuracy': float(accuracy),
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'precision_real': float(report['Real']['precision']),
            'recall_real': float(report['Real']['recall']),
            'f1_real': float(report['Real']['f1-score']),
            'precision_fake': float(report['Fake']['precision']),
            'recall_fake': float(report['Fake']['recall']),
            'f1_fake': float(report['Fake']['f1-score']),
            'recommended_threshold': float(self.recommended_threshold),
            'threshold_used': float(active_threshold),
            'validation_threshold_metrics': dict(self.validation_threshold_metrics),
            'threshold_profiles': dict(self.threshold_profiles),
            'recommended_threshold_profile': self.validation_threshold_metrics.get('recommended_profile'),
            'threshold_sweep': list(self.threshold_sweep),
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }

        threshold_candidates = {round(float(active_threshold), 2), 0.5}
        for profile_summary in self.threshold_profiles.values():
            candidate = profile_summary.get('recommended_threshold')
            if candidate is not None:
                threshold_candidates.add(round(float(candidate), 2))
        results['test_threshold_comparison'] = {
            f"{candidate:.2f}": {
                'accuracy': summary['accuracy'],
                'precision_fake': summary['precision_fake'],
                'recall_fake': summary['recall_fake'],
                'recall_real': summary['recall_real'],
                'f1_fake': summary['f1_fake'],
            }
            for candidate in sorted(threshold_candidates)
            for summary in [self._summarize_threshold_metrics(y_true, y_pred_prob, candidate)]
        }
        
        # Save to JSON
        results_path = os.path.join(self.models_dir, f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        if not self.history:
            logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history['loss'], label='Train Loss')
        axes[1].plot(self.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training plot saved to {save_path}")
        else:
            plt.show()
    
    def fine_tune_existing_model(self, model_path: str, train_generator, val_generator):
        """
        Fine-tune an existing .h5 model
        
        Args:
            model_path: Path to existing .h5 model
            train_generator: Training data generator
            val_generator: Validation data generator
        """
        logger.info(f"Fine-tuning existing model: {model_path}")
        self.build_model(load_existing=model_path)
        self.train(train_generator, val_generator, fine_tune=True)


def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--model-type', type=str, default='efficientnet_v2',
                       choices=['efficientnet_v2', 'efficientnet', 'mobilenet', 'resnet_v2', 'custom'],
                       help='Type of model to train')
    parser.add_argument('--fine-tune', type=str, default=None,
                       help='Path to existing .h5 model to fine-tune')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Total number of training epochs across all stages')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--data-dir', type=str, default='datasets/deepfake',
                       help='Directory containing training data')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--steps-per-epoch', type=int, default=1000,
                       help='Limit number of steps per epoch for faster iterations')
    parser.add_argument('--validation-steps', type=int, default=256,
                       help='Limit validation steps per epoch for faster feedback (0 uses the full validation split)')
    
    args = parser.parse_args()
    
    trainer = DeepfakeTrainer(models_dir=args.models_dir, data_dir=args.data_dir)
    trainer.epochs = args.epochs
    trainer.batch_size = args.batch_size
    trainer.steps_per_epoch = args.steps_per_epoch
    trainer.validation_steps = args.validation_steps
    
    # Prepare data
    train_gen, val_gen = trainer.prepare_data_generators()
    
    # Build or load model
    if args.fine_tune:
        trainer.build_model(load_existing=args.fine_tune)
        trainer.train(train_gen, val_gen, fine_tune=True)
    else:
        trainer.build_model(model_type=args.model_type)
        # Always run fine-tune stage for better results
        trainer.train(train_gen, val_gen, fine_tune=True)
    
    # Evaluate on test data
    test_gen = trainer.prepare_test_data()
    results = trainer.evaluate(test_gen)
    
    # Plot training history
    plot_path = os.path.join(args.models_dir, f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    trainer.plot_training_history(save_path=plot_path)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final accuracy: {results['accuracy']:.4f}")
    logger.info(f"Recommended threshold: {results['recommended_threshold']:.2f}")
    
    # Print recommendation based on results
    if results['accuracy'] > 0.95:
        logger.info("🎉 Excellent model! This will significantly improve your deepfake detection.")
    elif results['accuracy'] > 0.90:
        logger.info("✅ Good model! Should work well in production.")
    elif results['accuracy'] > 0.85:
        logger.info("👍 Decent model. Consider collecting more training data for better results.")
    else:
        logger.info("⚠️ Model needs improvement. Try collecting more diverse data or training longer.")


if __name__ == "__main__":
    main()
