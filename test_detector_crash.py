import unittest
from pathlib import Path

from fake_news_detector import FakeNewsDetector


class FakeNewsModelLoadTests(unittest.TestCase):
    def test_transformer_model_registration_does_not_crash_when_present(self):
        model_path = Path("models/fake_news/transformer_distilbert-base-uncased_20260306_171509")
        if not model_path.exists():
            self.skipTest(f"Optional transformer model not present: {model_path}")

        detector = FakeNewsDetector()
        loaded = detector.load_transformer_model(str(model_path))

        self.assertTrue(loaded)
        self.assertIsNotNone(detector.transformer_backend)


if __name__ == "__main__":
    unittest.main(verbosity=2)
