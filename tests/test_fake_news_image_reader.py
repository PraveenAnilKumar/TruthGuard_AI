import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image

import ocr_utils
from fake_news_detector import FakeNewsDetector


class OCRUtilsTests(unittest.TestCase):
    def test_parse_ocr_json_file_reads_utf8_payload(self):
        output_path = Path("temp/test_ocr_payload.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"text":"UTF8 payload","line_count":1,"word_count":2}', encoding="utf-8")

        try:
            payload = ocr_utils._parse_ocr_json_file(output_path)
        finally:
            output_path.unlink(missing_ok=True)

        self.assertEqual(payload["text"], "UTF8 payload")
        self.assertEqual(payload["line_count"], 1)

    def test_parse_ocr_stdout_accepts_extra_powershell_output(self):
        payload = ocr_utils._parse_ocr_stdout(
            "\n".join([
                "Windows OCR initialized",
                '{"text":"Breaking news headline","line_count":1,"word_count":3}',
                "Completed successfully",
            ])
        )

        self.assertEqual(payload["text"], "Breaking news headline")
        self.assertEqual(payload["line_count"], 1)
        self.assertEqual(payload["word_count"], 3)

    @patch("ocr_utils.subprocess.run")
    @patch("ocr_utils._parse_ocr_json_file", return_value={
        "text": "Headline text",
        "line_count": 1,
        "word_count": 2,
        "language": "en-US",
    })
    def test_run_windows_ocr_prefers_result_file(self, parse_file_mock, run_mock):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = "Completed successfully\n"
        run_mock.return_value.stderr = ""

        result_path = ocr_utils.OCR_TEMP_DIR / "headline_existing12.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text('{"text":"Headline text"}', encoding="utf-8")

        with patch("ocr_utils.uuid.uuid4") as uuid_mock:
            uuid_mock.return_value.hex = "existing1234abcd"
            payload = ocr_utils._run_windows_ocr(Path("headline.png"))

        self.assertEqual(payload["text"], "Headline text")
        self.assertEqual(payload["language"], "en-US")
        parse_file_mock.assert_called_once_with(result_path)
        self.assertFalse(result_path.exists())

    @patch("ocr_utils._run_windows_ocr", return_value={
        "text": "Breaking news headline",
        "line_count": 1,
        "word_count": 3,
        "language": "en-US",
        "width": 640,
        "height": 360,
    })
    @patch("ocr_utils.get_image_reader_status", return_value={"available": True, "backend": "windows_ocr"})
    def test_extract_text_from_image_returns_metadata(self, _status, _ocr):
        image = Image.new("RGB", (320, 200), "white")

        payload = ocr_utils.extract_text_from_image(image, image_name="headline.png")

        self.assertEqual(payload["text"], "Breaking news headline")
        self.assertEqual(payload["backend"], "windows_ocr")
        self.assertEqual(payload["image_name"], "headline.png")
        self.assertEqual(payload["line_count"], 1)
        self.assertEqual(payload["word_count"], 3)
        self.assertGreater(payload["preprocessed_size"][0], 0)
        self.assertGreater(payload["preprocessed_size"][1], 0)


class FakeNewsImageReaderTests(unittest.TestCase):
    def _build_detector(self):
        with patch("fake_news_detector._download_nltk_data", return_value=None), \
             patch("fake_news_detector.stopwords.words", return_value=[]), \
             patch.object(FakeNewsDetector, "get_available_models", return_value=[]):
            return FakeNewsDetector()

    def test_predict_from_image_merges_ocr_metadata(self):
        detector = self._build_detector()
        base_meta = {
            "original_language": "en",
            "was_translated": False,
            "processed_text": None,
            "realtime_result": None,
            "ensemble_mode": False,
            "individual_scores": {},
        }

        with patch("fake_news_detector.extract_text_from_image", return_value={
            "text": "City council approves long-term clean water funding package",
            "backend": "windows_ocr",
            "image_name": "article_card.png",
            "line_count": 2,
            "word_count": 8,
            "language": "en-US",
            "width": 1200,
            "height": 800,
            "preprocessed_size": [1200, 800],
        }), patch.object(detector, "predict", return_value=("REAL", 0.91, 0.08, base_meta)) as predict_mock:
            label, confidence, clickbait, meta = detector.predict_from_image(
                object(),
                image_name="article_card.png",
                check_realtime=True,
            )

        predict_mock.assert_called_once_with(
            "City council approves long-term clean water funding package",
            check_realtime=True,
            use_ensemble=False,
            requested_models=None,
        )
        self.assertEqual(label, "REAL")
        self.assertAlmostEqual(confidence, 0.91)
        self.assertAlmostEqual(clickbait, 0.08)
        self.assertEqual(meta["source_type"], "image")
        self.assertEqual(meta["ocr"]["backend"], "windows_ocr")
        self.assertEqual(meta["ocr"]["image_name"], "article_card.png")
        self.assertEqual(meta["ocr"]["line_count"], 2)
        self.assertEqual(meta["ocr"]["word_count"], 8)

    def test_predict_from_image_rejects_short_extractions(self):
        detector = self._build_detector()

        with patch("fake_news_detector.extract_text_from_image", return_value={
            "text": "too short",
            "backend": "windows_ocr",
        }):
            with self.assertRaises(ValueError):
                detector.predict_from_image(object(), image_name="small.png")


if __name__ == "__main__":
    unittest.main()
