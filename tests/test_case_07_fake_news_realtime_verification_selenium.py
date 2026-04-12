import unittest

try:
    from tests.selenium_test_utils import TruthGuardSeleniumBase, make_user
except ImportError:
    from selenium_test_utils import TruthGuardSeleniumBase, make_user


class FakeNewsRealtimeVerificationSeleniumTestCase(TruthGuardSeleniumBase):
    initial_users = {"case_user": make_user("StrongPass123")}
    stub_modules = ["fake_news_detector", "realtime_verifier"]

    def test_fake_news_realtime_verification(self):
        self.login("case_user", "StrongPass123")
        self.click_text("Launch Verifier")
        self.wait_for_text("Fake News Verification")

        self.click_text("Real-time Verification")
        self.wait_for_text("Enter Claim or Headline")

        self.fill_textarea(
            "Enter Claim or Headline",
            "Breaking health advisory confirmed by agencies",
        )
        self.click_text("Verify Against Live News")
        self.wait_for_text("Live News Comparison", timeout=40)
        self.wait_for_text("HIGH CONSENSUS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
