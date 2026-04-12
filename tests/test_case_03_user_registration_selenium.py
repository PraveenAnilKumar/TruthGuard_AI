import unittest

try:
    from tests.selenium_test_utils import TruthGuardSeleniumBase
except ImportError:
    from selenium_test_utils import TruthGuardSeleniumBase


class UserRegistrationSeleniumTestCase(TruthGuardSeleniumBase):
    def test_user_registration(self):
        self.click_text("Register")
        self.fill_input("New username", "newuser")
        self.fill_input("New password", "SecurePass123")
        self.fill_input("Confirm password", "SecurePass123")
        self.click_text("Create account")
        self.wait_for_text("Account created successfully")


if __name__ == "__main__":
    unittest.main(verbosity=2)
