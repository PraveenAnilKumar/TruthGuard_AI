import unittest
from unittest.mock import patch

from fake_news_detector import FakeNewsDetector
from realtime_verifier import RealtimeNewsVerifier


class RealtimeVerifierContradictionTests(unittest.TestCase):
    def test_verify_claim_detects_contradictory_outcome_language(self):
        verifier = RealtimeNewsVerifier()

        verifier._search_deep = lambda claim_text: {
            "queries": ["Australia 2023 ICC Men's Cricket World Cup"],
            "results": [
                {
                    "title": "Australia won the 2023 ICC Men's Cricket World Cup final",
                    "url": "https://example.com/reuters-aus-win",
                    "pub_date": "Mon, 20 Nov 2023 10:00:00 GMT",
                    "source": "Reuters",
                    "queries": ["Australia 2023 ICC Men's Cricket World Cup"],
                },
                {
                    "title": "Australia beat India to become Cricket World Cup champions",
                    "url": "https://example.com/ap-aus-win",
                    "pub_date": "Mon, 20 Nov 2023 11:00:00 GMT",
                    "source": "Associated Press",
                    "queries": ["Australia 2023 ICC Men's Cricket World Cup"],
                },
            ],
        }
        verifier._fetch_article_text = lambda url: (
            "Australia won the 2023 ICC Men's Cricket World Cup after beating India in the final in Ahmedabad."
        )

        result = verifier.verify_claim("Australia lost the 2023 ICC Men's Cricket World Cup.")

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["verdict_code"], "CONTRADICTED_BY_SOURCES")
        self.assertGreaterEqual(result["contradiction_score"], 0.2)
        self.assertTrue(result["sources"])
        self.assertEqual(result["sources"][0]["stance_status"], "contradictory")


class FakeNewsDetectorRealtimeAdjustmentTests(unittest.TestCase):
    def _build_detector(self):
        with patch("fake_news_detector._download_nltk_data", return_value=None), \
             patch("fake_news_detector.stopwords.words", return_value=[]), \
             patch.object(FakeNewsDetector, "get_available_models", return_value=[]):
            return FakeNewsDetector()

    def test_fallback_prediction_uses_contradictory_realtime_result(self):
        detector = self._build_detector()

        class StubRealtimeVerifier:
            def verify_claim(self, text):
                return {
                    "status": "SUCCESS",
                    "consensus_score": 0.04,
                    "contradiction_score": 0.38,
                    "verdict_code": "CONTRADICTED_BY_SOURCES",
                    "sources": [
                        {
                            "title": "Australia won the 2023 ICC Men's Cricket World Cup final",
                            "stance_status": "contradictory",
                        }
                    ],
                }

        with patch.object(detector, "get_realtime_verifier", return_value=StubRealtimeVerifier()):
            label, confidence, _clickbait, meta = detector.predict(
                "Australia lost the 2023 ICC Men's Cricket World Cup.",
                check_realtime=True,
            )

        self.assertEqual(label, "FAKE")
        self.assertGreaterEqual(confidence, 0.72)
        self.assertIn("contradicted", meta.get("realtime_impact", "").lower())

    def test_short_claim_without_strong_verification_is_unverified(self):
        detector = self._build_detector()

        class StubRealtimeVerifier:
            def verify_claim(self, text):
                return {
                    "status": "SUCCESS",
                    "consensus_score": 0.18,
                    "contradiction_score": 0.0,
                    "verdict_code": "UNVERIFIED",
                    "sources": [],
                }

        with patch.object(detector, "get_realtime_verifier", return_value=StubRealtimeVerifier()):
            label, confidence, _clickbait, meta = detector.predict(
                "Australia lost the 2023 ICC Men's Cricket World Cup.",
                check_realtime=True,
            )

        self.assertEqual(label, "UNVERIFIED")
        self.assertLessEqual(confidence, 0.6)
        self.assertIn("unverified", meta.get("realtime_impact", "").lower())


if __name__ == "__main__":
    unittest.main()
