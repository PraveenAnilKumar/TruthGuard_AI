import unittest

try:
    from tests.selenium_test_utils import TruthGuardSeleniumBase, make_user
except ImportError:
    from selenium_test_utils import TruthGuardSeleniumBase, make_user


class BatchProcessingSeleniumTestCase(TruthGuardSeleniumBase):
    initial_users = {"case_user": make_user("StrongPass123")}
    stub_modules = [
        "sentiment_analyzer",
        "toxicity_detector",
        "toxicity_viz",
        "batch_sentiment",
        "sentiment_viz",
    ]

    def test_batch_processing(self):
        self.login("case_user", "StrongPass123")
        self.click_text("Launch Communication Analysis")
        self.wait_for_text("Sentiment, Emotion & Safety")

        self.click_text("Batch Processing")
        self.wait_for_text("Upload Content")

        txt_file = self.make_test_text_file()
        self.upload_file(txt_file)

        self.click_text("Execute Batch Analysis")
        self.wait_for_text("Batch Analysis Overview", timeout=40)
        self.wait_for_text("Detailed Transmission Log")


if __name__ == "__main__":
    unittest.main(verbosity=2)
