import unittest

from realtime_verifier import RealtimeNewsVerifier


class RealtimeVerifierDeepSearchTests(unittest.TestCase):
    def test_build_query_variants_creates_multiple_distinct_queries(self):
        verifier = RealtimeNewsVerifier()
        text = 'NASA says "Artemis II" astronauts will travel around the Moon in 2026.'

        queries = verifier._build_query_variants(text)

        self.assertGreaterEqual(len(queries), 3)
        self.assertEqual(len(queries), len(set(q.lower() for q in queries)))
        self.assertTrue(any("Artemis II" in q or "artemis ii" in q.lower() for q in queries))

    def test_verify_claim_uses_credibility_and_body_match(self):
        verifier = RealtimeNewsVerifier()

        verifier._search_deep = lambda claim_text: {
            "queries": ["nasa artemis ii moon 2026", '"Artemis II" NASA Moon'],
            "results": [
                {
                    "title": "NASA confirms Artemis II Moon mission timeline",
                    "url": "https://www.reuters.com/world/us/nasa-confirms-artemis-ii-moon-mission-timeline-2026/",
                    "pub_date": "Fri, 20 Mar 2026 10:00:00 GMT",
                    "source": "Reuters",
                    "queries": ["nasa artemis ii moon 2026"],
                },
                {
                    "title": "NASA provides Artemis II mission update",
                    "url": "https://apnews.com/article/artemis-ii-mission-update",
                    "pub_date": "Fri, 20 Mar 2026 11:00:00 GMT",
                    "source": "Associated Press",
                    "queries": ['"Artemis II" NASA Moon'],
                },
            ],
        }
        verifier._fetch_article_text = lambda url: (
            "NASA confirmed Artemis II astronauts will fly around the Moon in 2026 after a mission review."
            if "reuters" in url else
            "The agency said Artemis II remains scheduled for a crewed lunar flyby in 2026."
        )

        result = verifier.verify_claim("NASA confirms Artemis II astronauts will travel around the Moon in 2026.")

        self.assertEqual(result["status"], "SUCCESS")
        self.assertGreaterEqual(result["consensus_score"], 0.55)
        self.assertIn(result["verdict_code"], {"VERIFIED_ONLINE", "PARTIALLY_SUPPORTED"})
        self.assertGreaterEqual(len(result["sources"]), 2)
        top = result["sources"][0]
        self.assertGreater(top["credibility_score"], 0.85)
        self.assertGreater(top["body_similarity"], 0.05)
        self.assertTrue(result["search_queries"])


if __name__ == "__main__":
    unittest.main()
