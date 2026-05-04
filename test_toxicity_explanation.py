import unittest

from toxicity_detector import ToxicityDetector
from toxicity_viz import ToxicityVisualizer


class ToxicityExplanationTests(unittest.TestCase):
    def test_prediction_explanation_and_highlighting_are_consistent(self):
        detector = ToxicityDetector(use_ensemble=False)
        text = "You are a total idiot and a stupid loser."

        is_toxic, confidence, categories, explanation, _meta = detector.predict(text)
        highlighted_html = ToxicityVisualizer.render_toxic_highlights(text, explanation)
        card_html = ToxicityVisualizer.create_explanation_card("insult", ["idiot", "stupid"], 0.85)

        self.assertIsInstance(is_toxic, bool)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(categories, dict)
        self.assertIn("reasons", explanation)
        self.assertIn("word_impact", explanation)
        self.assertIn("<span", highlighted_html)
        self.assertIn("Insult", card_html)
        self.assertIn("85.0%", card_html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
