import unittest
from unittest.mock import patch

from fake_news_detector import FakeNewsDetector


class FakeNewsGranularSelectionTests(unittest.TestCase):
    def _build_detector(self):
        with patch("fake_news_detector._download_nltk_data", return_value=None), \
             patch("fake_news_detector.stopwords.words", return_value=[]):
            return FakeNewsDetector()

    def test_hf_registration_can_be_selected_without_loading_every_model(self):
        detector = self._build_detector()
        hf_models = detector.get_huggingface_models()
        if not hf_models:
            self.skipTest("No fake-news HF models configured")

        target_hf = hf_models[0]
        loaded = detector.ensure_model_loaded(requested_models=[target_hf])

        self.assertTrue(loaded)
        self.assertEqual(list(detector._loaded_hf_pipelines.keys()), [target_hf])
        self.assertTrue(detector.is_trained)

    def test_requesting_local_model_unloads_hf_registrations(self):
        detector = self._build_detector()
        detector._loaded_hf_pipelines["HF_DistilRoBERTa"] = {
            "model": "mrm8488/distilroberta-finetuned-fake-news",
            "tokenizer": "mrm8488/distilroberta-finetuned-fake-news",
            "local_files_only": False,
        }
        detector.is_trained = True
        detector.use_transformer = True

        detector.ensure_model_loaded(requested_models=["Local_ML"])

        self.assertFalse(detector._loaded_hf_pipelines)


if __name__ == "__main__":
    unittest.main(verbosity=2)
