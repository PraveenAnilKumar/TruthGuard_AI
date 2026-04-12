import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from train_deepfake import DeepfakeTrainer


class _FakeIterator:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.samples = batch_size * 2
        self.class_indices = {"real": 0, "fake": 1}
        self.classes = np.array([0, 1] * batch_size, dtype=np.int32)

    def reset(self):
        return None


class _FakeImageDataGenerator:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def flow_from_directory(self, directory, **kwargs):
        self.__class__.calls.append(
            {
                "directory": directory,
                "kwargs": dict(kwargs),
                "init_kwargs": dict(self.kwargs),
            }
        )
        return _FakeIterator(kwargs["batch_size"])


class _DummyPredictionModel:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 1)

    def predict(self, generator, steps=None, verbose=0):
        return self.predictions


class _DummyGenerator:
    def __init__(self, labels, batch_size=2):
        self.classes = np.asarray(labels, dtype=np.int32)
        self.samples = len(labels)
        self.batch_size = batch_size
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class TrainDeepfakeUpgradeTests(unittest.TestCase):
    def test_prepare_data_generators_prefers_explicit_val_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DeepfakeTrainer(models_dir=tmpdir, data_dir=tmpdir)
            for rel_path in [
                "val/real/sample_real.jpg",
                "val/fake/sample_fake.jpg",
            ]:
                abs_path = os.path.join(tmpdir, rel_path)
                with open(abs_path, "wb") as handle:
                    handle.write(b"test")

            _FakeImageDataGenerator.calls = []
            with patch("train_deepfake.ImageDataGenerator", _FakeImageDataGenerator):
                trainer.prepare_data_generators()

            self.assertEqual(len(_FakeImageDataGenerator.calls), 2)
            train_call, val_call = _FakeImageDataGenerator.calls
            self.assertTrue(train_call["directory"].endswith(os.path.join("train")))
            self.assertTrue(val_call["directory"].endswith(os.path.join("val")))
            self.assertNotIn("subset", train_call["kwargs"])
            self.assertNotIn("subset", val_call["kwargs"])
            self.assertFalse(val_call["kwargs"]["shuffle"])
            self.assertNotIn("validation_split", train_call["init_kwargs"])
            self.assertNotIn("validation_split", val_call["init_kwargs"])

    def test_tune_threshold_prefers_lower_threshold_for_fake_recall(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DeepfakeTrainer(models_dir=tmpdir, data_dir=tmpdir)
            trainer.model = _DummyPredictionModel([0.10, 0.35, 0.45, 0.90])
            generator = _DummyGenerator([0, 0, 1, 1])

            summary = trainer.tune_threshold(generator)

            self.assertLess(summary["recommended_threshold"], 0.5)
            self.assertAlmostEqual(summary["validation_f2_fake"], 1.0, places=4)
            self.assertIn("threshold_profiles", summary)
            self.assertIn("balanced", summary["threshold_profiles"])
            self.assertTrue(summary["threshold_sweep"])
            self.assertEqual(generator.reset_calls, 1)

    def test_build_validation_monitor_generator_shuffles_subset_monitoring(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DeepfakeTrainer(models_dir=tmpdir, data_dir=tmpdir)
            _FakeImageDataGenerator.calls = []
            datagen = _FakeImageDataGenerator(preprocessing_function=lambda x: x)
            reference = SimpleNamespace(
                image_data_generator=datagen,
                directory=os.path.join(tmpdir, "val"),
                target_size=(224, 224),
                batch_size=4,
                class_mode="binary",
                class_indices={"real": 0, "fake": 1},
                color_mode="rgb",
                data_format="channels_last",
                interpolation="nearest",
                keep_aspect_ratio=False,
                subset=None,
                follow_links=False,
            )

            trainer._build_validation_monitor_generator(reference)

            self.assertEqual(len(_FakeImageDataGenerator.calls), 1)
            call = _FakeImageDataGenerator.calls[0]
            self.assertTrue(call["directory"].endswith("val"))
            self.assertTrue(call["kwargs"]["shuffle"])
            self.assertEqual(call["kwargs"]["classes"], ["real", "fake"])


if __name__ == "__main__":
    unittest.main()
