import unittest
from unittest.mock import patch

from fake_news_detector import FakeNewsDetector


class FakeNewsRealtimeIntegrationTests(unittest.TestCase):
    def _build_detector(self):
        with patch("fake_news_detector._download_nltk_data", return_value=None), \
             patch("fake_news_detector.stopwords.words", return_value=[]), \
             patch.object(FakeNewsDetector, "get_available_models", return_value=[]):
            return FakeNewsDetector()

    def test_prediction_attaches_realtime_and_global_outlet_metadata(self):
        detector = self._build_detector()

        class StubRealtimeVerifier:
            def verify_claim(self, text):
                return {
                    "status": "SUCCESS",
                    "consensus_score": 0.82,
                    "contradiction_score": 0.0,
                    "verdict_code": "VERIFIED_ONLINE",
                    "sources": [
                        {
                            "title": "Reuters confirms the same claim",
                            "source": "Reuters",
                            "global_outlet": True,
                            "global_outlet_name": "Reuters",
                            "score": 0.84,
                        }
                    ],
                    "global_outlet_comparison": {
                        "status": "SUPPORTED_BY_GLOBAL_OUTLETS",
                        "matched_outlet_count": 1,
                        "strong_match_count": 1,
                        "contradiction_count": 0,
                        "coverage_score": 0.84,
                    },
                }

        with patch.object(detector, "get_realtime_verifier", return_value=StubRealtimeVerifier()):
            label, confidence, _clickbait, meta = detector.predict(
                "Reuters confirmed that the central claim was accurate.",
                check_realtime=True,
            )

        self.assertIn(label, {"REAL", "UNVERIFIED", "FAKE"})
        self.assertGreaterEqual(confidence, 0.0)
        self.assertIn("realtime_result", meta)
        self.assertEqual(
            meta["realtime_result"]["global_outlet_comparison"]["status"],
            "SUPPORTED_BY_GLOBAL_OUTLETS",
        )
        self.assertIn("final_verdict", meta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
