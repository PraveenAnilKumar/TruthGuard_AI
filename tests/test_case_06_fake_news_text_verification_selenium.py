import unittest

try:
    from tests.selenium_test_utils import TruthGuardSeleniumBase, make_user
except ImportError:
    from selenium_test_utils import TruthGuardSeleniumBase, make_user


class FakeNewsTextVerificationSeleniumTestCase(TruthGuardSeleniumBase):
    initial_users = {"case_user": make_user("StrongPass123")}
    stub_modules = ["fake_news_detector"]

    def test_fake_news_text_verification(self):
        self.login("case_user", "StrongPass123")
        self.click_text("Launch Verifier")
        self.wait_for_text("Fake News Verification")

        self.fill_textarea(
            "News Article Content",
            "Official public safety bulletin confirmed by multiple agencies.",
        )
        self.click_text("Run Forensic Analysis")
        self.wait_for_text("CREDIBLE NEWS", timeout=40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
