import unittest

try:
    from tests.selenium_test_utils import TruthGuardSeleniumBase, make_user
except ImportError:
    from selenium_test_utils import TruthGuardSeleniumBase, make_user


class LogoutSeleniumTestCase(TruthGuardSeleniumBase):
    initial_users = {"case_user": make_user("StrongPass123")}

    def test_user_logout(self):
        self.login("case_user", "StrongPass123")
        self.click_text("Logout")
        self.wait_for_text("Login")
        self.wait_for_text("Username")


if __name__ == "__main__":
    unittest.main(verbosity=2)
