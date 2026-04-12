import unittest

try:
    from tests.selenium_test_utils import TruthGuardSeleniumBase, make_user
except ImportError:
    from selenium_test_utils import TruthGuardSeleniumBase, make_user


class DeepfakeImageAnalysisSeleniumTestCase(TruthGuardSeleniumBase):
    initial_users = {"case_user": make_user("StrongPass123")}
    stub_modules = ["deepfake_detector_advanced", "safe_deepfake_runtime"]

    def test_deepfake_image_analysis(self):
        self.login("case_user", "StrongPass123")
        self.click_text("Launch Deepfake Analyzer")
        self.wait_for_text("Deepfake Detection Analysis")
        self.wait_for_text("Model Selection Mode")

        image_path = self.make_test_image()
        self.upload_file(image_path)

        self.wait_for_text("Target Image")
        self.click_text("Run Image Deepfake Analysis")
        self.wait_for_text("Deepfake Forensic Report", timeout=40)
        self.wait_for_text("LIKELY AUTHENTIC")


if __name__ == "__main__":
    unittest.main(verbosity=2)
