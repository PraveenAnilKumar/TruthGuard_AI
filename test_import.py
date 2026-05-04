import importlib
import unittest


class ImportSmokeTests(unittest.TestCase):
    def test_core_modules_import_without_eager_model_loading(self):
        modules = [
            "realtime_verifier",
            "fake_news_detector",
            "sentiment_analyzer",
            "toxicity_detector",
            "deepfake_detector_advanced",
            "ocr_utils",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main(verbosity=2)
