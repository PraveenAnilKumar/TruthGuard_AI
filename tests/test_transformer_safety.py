import os
import unittest
from subprocess import CompletedProcess
from unittest.mock import patch

from deepfake_detector_advanced import DeepfakeDetectorAdvanced
from fake_news_detector import FakeNewsDetector
import safe_deepfake_runtime
from sentiment_analyzer import SentimentAnalyzer
from toxicity_detector import ToxicityDetector


class TransformerSafetyTests(unittest.TestCase):
    def test_fake_news_load_transformer_uses_isolated_backend(self):
        detector = FakeNewsDetector()
        model_path = os.path.join(
            "models",
            "fake_news",
            "transformer_distilbert-base-uncased_20260306_171509",
        )

        loaded = detector.load_transformer_model(model_path)

        self.assertTrue(loaded)
        self.assertTrue(detector.use_transformer)
        self.assertIsNotNone(detector.transformer_backend)
        self.assertIsNone(detector.transformer_pipeline)

    @patch(
        "fake_news_detector.run_isolated_text_classification",
        return_value={
            "ok": True,
            "result": {"label": "LABEL_1", "score": 0.91, "scores": [0.09, 0.91]},
        },
    )
    def test_fake_news_transformer_prediction_uses_isolated_worker(self, worker_mock):
        detector = FakeNewsDetector()
        detector.transformer_backend = {
            "model": "dummy-model",
            "tokenizer": "dummy-model",
            "local_files_only": False,
        }
        detector.use_transformer = True
        detector.is_trained = True

        label, confidence = detector._predict_transformer("Breaking story")

        self.assertEqual(label, "FAKE")
        self.assertAlmostEqual(confidence, 0.91, places=2)
        worker_mock.assert_called_once()

    @patch(
        "sentiment_analyzer.run_isolated_text_classification",
        return_value={
            "ok": True,
            "result": {"label": "POSITIVE", "score": 0.95, "scores": [0.05, 0.95]},
        },
    )
    def test_sentiment_distilbert_uses_isolated_worker(self, worker_mock):
        analyzer = SentimentAnalyzer(use_ensemble=True)
        analyzer._ensure_distilbert()

        result = analyzer._analyze_distilbert_detailed("I absolutely love this")

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "POSITIVE")
        self.assertGreater(result["signed_score"], 0.0)
        worker_mock.assert_called_once()

    @patch(
        "toxicity_detector.run_isolated_text_classification",
        return_value={
            "ok": True,
            "result": {"label": "LABEL_1", "score": 0.88, "scores": [0.12, 0.88]},
        },
    )
    def test_toxicity_transformer_prediction_uses_isolated_worker(self, worker_mock):
        detector = ToxicityDetector(use_ensemble=False)
        detector.is_trained = True
        detector.use_transformer = True
        detector.transformer_model_ref = "dummy-toxic-model"
        detector.transformer_tokenizer_ref = "dummy-toxic-model"

        is_toxic, probability, _categories = detector._predict_transformer("you are awful")

        self.assertTrue(is_toxic)
        self.assertAlmostEqual(probability, 0.88, places=2)
        worker_mock.assert_called_once()

    def test_deepfake_hf_models_hidden_by_default(self):
        detector = DeepfakeDetectorAdvanced()

        info = detector.get_model_info()

        self.assertFalse(detector.enable_hf_model)
        self.assertEqual(info["hf_models"], [])

    def test_deepfake_worker_env_forces_cpu_safe_settings(self):
        env = safe_deepfake_runtime._build_worker_env()

        self.assertEqual(env["TRUTHGUARD_DEEPFAKE_WORKER"], "1")
        self.assertEqual(env["TRUTHGUARD_ENABLE_ADVANCED_FACE_DETECTORS"], "1")
        self.assertEqual(env["TRUTHGUARD_ENABLE_HF_DEEPFAKE_MODELS"], "0")
        self.assertEqual(env["KERAS_BACKEND"], "tensorflow")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")

    def test_deepfake_worker_env_enables_hf_only_on_explicit_request(self):
        env = safe_deepfake_runtime._build_worker_env(enable_hf_models=True)

        self.assertEqual(env["TRUTHGUARD_ENABLE_HF_DEEPFAKE_MODELS"], "1")

    @patch("safe_deepfake_runtime.subprocess.run")
    def test_deepfake_worker_fatal_exit_returns_safe_error(self, run_mock):
        run_mock.return_value = CompletedProcess(
            args=["python", "worker"],
            returncode=3221225477,
            stdout="",
            stderr="fatal access violation",
        )

        result = safe_deepfake_runtime._run_worker({"action": "image_ensemble"}, timeout_seconds=30)

        self.assertIn("error", result)
        self.assertIn("The Streamlit app stayed alive", result["error"])


if __name__ == "__main__":
    unittest.main()
