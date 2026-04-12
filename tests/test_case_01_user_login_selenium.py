import unittest

try:
    from tests.selenium_test_utils import TruthGuardSeleniumBase, make_user
except ImportError:
    from selenium_test_utils import TruthGuardSeleniumBase, make_user


class UserLoginSeleniumTestCase(TruthGuardSeleniumBase):
    initial_users = {"case_user": make_user("StrongPass123")}

    def test_user_login(self):
        self.login("case_user", "StrongPass123")
        self.wait_for_text("Launch Deepfake Analyzer")
        self.wait_for_text("Launch Verifier")
        self.wait_for_text("Launch Communication Analysis")


if __name__ == "__main__":
    unittest.main(verbosity=2)
