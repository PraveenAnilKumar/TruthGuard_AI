import unittest

from toxicity_detector import ToxicityDetector
from toxicity_viz import ToxicityVisualizer


class ToxicitySmokeTests(unittest.TestCase):
    def test_keyword_fallback_prediction_returns_explanation(self):
        detector = ToxicityDetector(use_ensemble=False)

        is_toxic, confidence, categories, explanation, meta = detector.predict(
            "You are such an idiot and a stupid loser."
        )

        self.assertIsInstance(is_toxic, bool)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(categories, dict)
        self.assertIsInstance(explanation, dict)
        self.assertIn("reasons", explanation)
        self.assertIsInstance(meta, dict)

    def test_visualizer_generates_highlight_html(self):
        explanation = {
            "word_impact": {
                "idiot": {"score": 0.85},
                "stupid": {"score": 0.70},
            },
        }

        html = ToxicityVisualizer.render_toxic_highlights(
            "That was an idiot and stupid remark.",
            explanation,
        )

        self.assertIn("<span", html)
        self.assertIn("idiot", html.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
