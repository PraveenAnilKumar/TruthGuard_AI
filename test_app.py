import unittest
from pathlib import Path


class AppSmokeTests(unittest.TestCase):
    def test_app_entrypoint_exists(self):
        self.assertTrue(Path("app.py").exists())

    def test_required_runtime_files_exist(self):
        required = [
            "fake_news_detector.py",
            "realtime_verifier.py",
            "deepfake_detector_advanced.py",
            "sentiment_analyzer.py",
            "toxicity_detector.py",
            "ocr_utils.py",
        ]

        for path in required:
            with self.subTest(path=path):
                self.assertTrue(Path(path).exists())

    def test_ocr_status_probe_returns_structured_result(self):
        from ocr_utils import get_image_reader_status

        status = get_image_reader_status()

        self.assertIsInstance(status, dict)
        self.assertIn("available", status)
        self.assertIn("backend", status)


if __name__ == "__main__":
    unittest.main(verbosity=2)
