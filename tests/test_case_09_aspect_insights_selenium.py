import unittest

try:
    from tests.selenium_test_utils import TruthGuardSeleniumBase, make_user
except ImportError:
    from selenium_test_utils import TruthGuardSeleniumBase, make_user


class AspectInsightsSeleniumTestCase(TruthGuardSeleniumBase):
    initial_users = {"case_user": make_user("StrongPass123")}
    stub_modules = [
        "sentiment_analyzer",
        "toxicity_detector",
        "toxicity_viz",
        "aspect_sentiment",
    ]

    def test_aspect_insights_analysis(self):
        self.login("case_user", "StrongPass123")
        self.click_text("Launch Communication Analysis")
        self.wait_for_text("Sentiment, Emotion & Safety")

        self.click_text("Aspect Insights")
        self.wait_for_text("Analyze Specific Qualities")

        self.fill_textarea(
            "Analyze Specific Qualities",
            "The service was excellent but the price was high.",
        )
        self.click_text("Extract Aspect Sentiment")
        self.wait_for_text("Aspect-Based Breakdown", timeout=40)
        self.wait_for_text("service")
        self.wait_for_text("price")


if __name__ == "__main__":
    unittest.main(verbosity=2)
