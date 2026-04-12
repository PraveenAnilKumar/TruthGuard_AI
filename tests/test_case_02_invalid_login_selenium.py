import unittest

try:
    from tests.selenium_test_utils import TruthGuardSeleniumBase, make_user
except ImportError:
    from selenium_test_utils import TruthGuardSeleniumBase, make_user


class InvalidLoginSeleniumTestCase(TruthGuardSeleniumBase):
    initial_users = {"case_user": make_user("StrongPass123")}

    def test_invalid_login(self):
        self.fill_input("Username", "case_user")
        self.fill_input("Password", "WrongPassword")
        self.click_text("Login")
        self.wait_for_text("Invalid credentials")


if __name__ == "__main__":
    unittest.main(verbosity=2)
