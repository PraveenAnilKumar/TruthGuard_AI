import unittest
from unittest.mock import patch

from fake_news_detector import FakeNewsDetector


class FakeNewsRobustPredictionTests(unittest.TestCase):
    def _build_detector(self):
        with patch("fake_news_detector._download_nltk_data", return_value=None), \
             patch("fake_news_detector.stopwords.words", return_value=[]), \
             patch.object(FakeNewsDetector, "get_available_models", return_value=[]):
            return FakeNewsDetector()

    def test_clickbait_score_accepts_string_and_sequence_inputs(self):
        detector = self._build_detector()

        string_score = detector._calculate_clickbait_score("Is this clickbait?")
        list_score = detector._calculate_clickbait_score(["This", "is", "a", "list", "?"])

        self.assertGreaterEqual(string_score, 0.0)
        self.assertLessEqual(string_score, 1.0)
        self.assertGreaterEqual(list_score, 0.0)
        self.assertLessEqual(list_score, 1.0)

    def test_traditional_prediction_falls_back_without_loaded_model(self):
        detector = self._build_detector()
        detector.model = None

        label, confidence = detector._predict_traditional("Some text")

        self.assertIn(label, {"REAL", "FAKE"})
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
