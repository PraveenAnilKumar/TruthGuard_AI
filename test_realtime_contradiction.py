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

    def test_global_outlet_comparison_tracks_credible_sources(self):
        verifier = RealtimeNewsVerifier()

        verifier._search_deep = lambda claim_text: {
            "queries": ["NASA Artemis II crew return"],
            "global_outlet_queries": ["NASA Artemis II crew return site:reuters.com"],
            "global_outlets_checked": verifier._global_outlet_names(),
            "results": [
                {
                    "title": "Reuters: NASA Artemis II crew returns after moon mission",
                    "url": "https://example.com/reuters-artemis",
                    "pub_date": "Fri, 10 Apr 2026 10:00:00 GMT",
                    "source": "Reuters",
                    "queries": ["NASA Artemis II crew return"],
                    "global_outlet_check": True,
                },
                {
                    "title": "BBC News - Artemis II astronauts return from moon mission",
                    "url": "https://example.com/bbc-artemis",
                    "pub_date": "Fri, 10 Apr 2026 11:00:00 GMT",
                    "source": "BBC News",
                    "queries": ["NASA Artemis II crew return"],
                    "global_outlet_check": True,
                },
            ],
        }
        verifier._fetch_article_text = lambda url: (
            "NASA confirmed the Artemis II astronauts returned after completing their mission around the moon."
        )

        result = verifier.verify_claim(
            "NASA confirmed the Artemis II astronauts returned after completing their mission around the moon."
        )

        comparison = result["global_outlet_comparison"]
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(comparison["status"], "SUPPORTED_BY_GLOBAL_OUTLETS")
        self.assertGreaterEqual(comparison["matched_outlet_count"], 2)
        self.assertGreaterEqual(comparison["strong_match_count"], 2)
        self.assertTrue(any(source["global_outlet"] for source in result["sources"]))


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

    def test_realtime_verifier_receives_full_article_text(self):
        detector = self._build_detector()
        captured = {}

        class StubRealtimeVerifier:
            def verify_claim(self, text):
                captured["text"] = text
                return {
                    "status": "NO_RESULTS",
                    "consensus_score": 0.0,
                    "sources": [],
                    "global_outlet_comparison": {},
                }

        long_text = (
            "NASA confirmed the Artemis II astronauts returned after completing their mission around the moon. "
            + "background detail " * 40
        )

        with patch.object(detector, "get_realtime_verifier", return_value=StubRealtimeVerifier()):
            detector.predict(long_text, check_realtime=True)

        self.assertGreater(len(captured.get("text", "")), 300)


if __name__ == "__main__":
    unittest.main()
