import unittest

try:
    from tests.selenium_test_utils import TruthGuardSeleniumBase, make_user
except ImportError:
    from selenium_test_utils import TruthGuardSeleniumBase, make_user


class CommunicationUnifiedScanSeleniumTestCase(TruthGuardSeleniumBase):
    initial_users = {"case_user": make_user("StrongPass123")}
    stub_modules = ["sentiment_analyzer", "toxicity_detector", "toxicity_viz"]

    def test_unified_communication_analysis(self):
        self.login("case_user", "StrongPass123")
        self.click_text("Launch Communication Analysis")
        self.wait_for_text("Sentiment, Emotion & Safety")

        self.fill_textarea(
            "Analyze Tone and Safety Together",
            "The platform response was helpful, respectful, and very clear.",
        )
        self.click_text("Run Communication Analysis")
        self.wait_for_text("POSITIVE SENTIMENT", timeout=40)
        self.wait_for_text("CONTENT SECURE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
