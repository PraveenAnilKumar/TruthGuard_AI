import unittest
from unittest.mock import patch

from sentiment_analyzer import SentimentAnalyzer
from toxicity_detector import ToxicityDetector


class SentimentUpgradeTests(unittest.TestCase):
    @patch("sentiment_analyzer.content_translator.translate_to_english", side_effect=lambda text: (text, "en", False))
    def test_sentiment_returns_richer_metadata(self, _translate):
        analyzer = SentimentAnalyzer(use_ensemble=False)

        label, confidence, meta = analyzer.analyze(
            "I absolutely love this product. It is amazing and makes me so happy!"
        )

        self.assertEqual(label, "POSITIVE")
        self.assertGreater(confidence, 0.55)
        self.assertIn("dominant_emotion", meta)
        self.assertIn("sentiment_score", meta)
        self.assertIn("model_breakdown", meta)
        self.assertGreater(meta["sentiment_score"], 0.2)
        self.assertIn(meta["dominant_emotion"], {"joy", "trust"})


class ToxicityUpgradeTests(unittest.TestCase):
    @patch("toxicity_detector.content_translator.translate_to_english", side_effect=lambda text: (text, "en", False))
    def test_targeted_insult_remains_toxic(self, _translate):
        with patch.object(ToxicityDetector, "_load_latest_model", lambda self: None):
            detector = ToxicityDetector(use_ensemble=False)

        detector.is_trained = False
        is_toxic, confidence, categories, explanation, meta = detector.predict(
            "You are a stupid idiot and a worthless loser."
        )

        self.assertTrue(is_toxic)
        self.assertGreater(confidence, detector.threshold)
        self.assertGreater(categories["insult"], 0.45)
        self.assertIn("severity", explanation)
        self.assertTrue(meta["analysis_context"]["targeted"])

    @patch("toxicity_detector.content_translator.translate_to_english", side_effect=lambda text: (text, "en", False))
    def test_deescalating_context_is_downweighted(self, _translate):
        with patch.object(ToxicityDetector, "_load_latest_model", lambda self: None):
            detector = ToxicityDetector(use_ensemble=False)

        detector.is_trained = False
        is_toxic, confidence, categories, explanation, meta = detector.predict(
            "Please do not call anyone an idiot. We should keep this respectful."
        )

        self.assertFalse(is_toxic)
        self.assertLess(confidence, detector.threshold)
        self.assertTrue(meta["analysis_context"]["deescalation"])
        self.assertIn("reasons", explanation)
        self.assertGreaterEqual(categories["insult"], 0.0)

    @patch("toxicity_detector.content_translator.translate_to_english", side_effect=lambda text: (text, "en", False))
    def test_benign_greeting_is_not_flagged_even_with_noisy_model_score(self, _translate):
        with patch.object(ToxicityDetector, "_load_latest_model", lambda self: None):
            detector = ToxicityDetector(use_ensemble=False)

        detector.is_trained = True
        detector.threshold = 0.45
        empty_categories = {category: 0.0 for category in detector.categories}

        with patch.object(detector, "_predict_sklearn", return_value=(True, 0.52, empty_categories)):
            is_toxic, confidence, categories, explanation, meta = detector.predict("hello how are you")

        self.assertFalse(is_toxic)
        self.assertLess(confidence, 0.25)
        self.assertTrue(meta["analysis_context"]["benign_conversational"])
        self.assertIn("low risk", explanation["reasons"][0].lower())
        self.assertLess(max(categories.values()), 0.05)

    @patch("toxicity_detector.content_translator.translate_to_english", side_effect=lambda text: (text, "en", False))
    def test_corrective_language_stays_below_aggressive_ui_threshold(self, _translate):
        with patch.object(ToxicityDetector, "_load_latest_model", lambda self: None):
            detector = ToxicityDetector(use_ensemble=False)

        detector.is_trained = True
        detector.threshold = 0.45
        model_categories = {category: 0.0 for category in detector.categories}
        model_categories["toxicity"] = 0.28
        model_categories["insult"] = 0.32

        with patch.object(detector, "_predict_sklearn", return_value=(True, 0.61, model_categories)):
            is_toxic, confidence, categories, explanation, meta = detector.predict(
                "Please do not call anyone an idiot. We should keep this respectful."
            )

        self.assertFalse(is_toxic)
        self.assertLess(confidence, detector.threshold)
        self.assertTrue(meta["analysis_context"]["deescalation"])
        self.assertIn("safe", explanation["reasons"][0].lower())
        self.assertLess(categories["insult"], 0.2)


if __name__ == "__main__":
    unittest.main()
