import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

import bcrypt
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = PROJECT_ROOT / "app.py"


def make_user(password, role="user"):
    return {
        "password": bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8"),
        "role": role,
    }


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _launcher_source(app_file, project_root, selected_stubs):
    return f"""
import runpy
import sys
import types

APP_FILE = {app_file!r}
PROJECT_ROOT = {project_root!r}
SELECTED_STUBS = set({list(selected_stubs)!r})

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_deepfake_detector_module():
    module = types.ModuleType("deepfake_detector_advanced")

    class DeepfakeDetectorAdvanced:
        def __init__(self, models_dir="models"):
            self.models_dir = models_dir
            self.available_hf_models = {{}}

        def get_model_info(self):
            return {{
                "local_models": ["xception_net"],
                "model_paths": ["models/xception_net.h5"],
                "preferred_model_name": "xception_net",
                "recommended_threshold": 0.55,
                "weights": {{"xception_net": 1.0}},
            }}

    module.DeepfakeDetectorAdvanced = DeepfakeDetectorAdvanced
    module.deepfake_detector = object()
    return module


def build_safe_deepfake_runtime_module():
    module = types.ModuleType("safe_deepfake_runtime")

    def run_isolated_deepfake_image_analysis(*args, **kwargs):
        return {{
            "is_deepfake": False,
            "confidence": 91.2,
            "ensemble_score": 0.14,
            "consistency": 0.93,
            "ela_score": 0.08,
            "fft_score": 0.10,
            "face_count": 1,
            "analyzed_region_count": 1,
            "message": "No strong manipulation signals detected.",
            "model_scores": {{"xception_net": 0.14}},
            "heatmap": None,
            "ela_image": None,
            "fft_image": None,
        }}

    def run_isolated_deepfake_video_analysis(*args, **kwargs):
        return {{
            "is_deepfake": False,
            "confidence": 89.5,
            "ensemble_score": 0.18,
            "consistency": 0.91,
            "ela_score": 0.09,
            "fft_score": 0.11,
            "face_count": 1,
            "analyzed_region_count": 1,
            "message": "No strong manipulation signals detected.",
            "model_scores": {{"xception_net": 0.18}},
            "heatmap": None,
            "ela_image": None,
            "fft_image": None,
            "deepfake_ratio": 0.12,
            "total_frames_analyzed": 12,
            "temporal_scores": [0.10, 0.15, 0.18, 0.12],
        }}

    module.run_isolated_deepfake_image_analysis = run_isolated_deepfake_image_analysis
    module.run_isolated_deepfake_video_analysis = run_isolated_deepfake_video_analysis
    return module


def build_fake_news_module():
    module = types.ModuleType("fake_news_detector")

    class FakeNewsDetector:
        def get_available_models(self):
            return [{{
                "name": "RF Baseline",
                "type": "random_forest",
                "path": "models/rf.pkl",
            }}]

        def get_model_info(self):
            return {{"model_type": "random_forest"}}

        def get_huggingface_models(self):
            return []

        def ensure_model_loaded(self, *args, **kwargs):
            return True

        def get_image_reader_status(self):
            return {{"available": False, "error": "OCR disabled in tests"}}

        def predict(self, text, check_realtime=True, use_ensemble=False, requested_models=None):
            return (
                "REAL",
                0.88,
                0.12,
                {{
                    "summary": "credible source language",
                    "model_votes": {{"rf": "REAL"}},
                    "realtime": {{"checked": check_realtime}},
                }},
            )

        def predict_from_image(self, image, image_name=None, check_realtime=True, use_ensemble=False, requested_models=None):
            return self.predict("image text", check_realtime, use_ensemble, requested_models)

    module.FakeNewsDetector = FakeNewsDetector
    return module


def build_realtime_verifier_module():
    module = types.ModuleType("realtime_verifier")

    class FakeRealtimeVerifier:
        def verify_claim(self, claim_text):
            return {{
                "status": "MATCH_FOUND",
                "consensus_score": 0.82,
                "verdict_code": "HIGH_CONSENSUS",
                "message": "Multiple reputable sources report similar information.",
                "search_query": claim_text,
                "search_queries": [claim_text],
                "sources": [{{
                    "source": "Reuters",
                    "domain": "reuters.com",
                    "title": "Agency confirms update",
                    "evidence_snippet": "Officials confirmed the reported update.",
                    "published": "2026-04-07",
                    "url": "https://example.com/reuters",
                    "score": 0.88,
                    "credibility_score": 0.95,
                    "body_similarity": 0.83,
                    "pure_similarity": 0.80,
                }}],
            }}

    module.realtime_verifier = FakeRealtimeVerifier()
    return module


def build_sentiment_module():
    module = types.ModuleType("sentiment_analyzer")

    class SentimentAnalyzer:
        def __init__(self, use_ensemble=True):
            self.use_ensemble = use_ensemble

        def analyze(self, text):
            return (
                "POSITIVE",
                0.91,
                {{
                    "dominant_emotion": "joy",
                    "sentiment_score": 0.82,
                    "model_breakdown": {{"stub_model": 0.91}},
                }},
            )

    module.SentimentAnalyzer = SentimentAnalyzer
    return module


def build_toxicity_module():
    module = types.ModuleType("toxicity_detector")

    class ToxicityDetector:
        def __init__(self, use_ensemble=True):
            self.use_ensemble = use_ensemble
            self.threshold = 0.45

        def predict(self, text):
            return (
                False,
                0.08,
                {{
                    "insult": 0.01,
                    "threat": 0.00,
                    "obscene": 0.00,
                }},
                {{
                    "severity": "low",
                    "reasons": ["No harmful language pattern detected."],
                }},
                {{
                    "analysis_context": {{
                        "targeted": False,
                        "deescalation": True,
                    }}
                }},
            )

    module.ToxicityDetector = ToxicityDetector
    return module


def build_toxicity_viz_module():
    module = types.ModuleType("toxicity_viz")

    class ToxicityVisualizer:
        @staticmethod
        def render_toxic_highlights(text, explanation):
            return text

    module.ToxicityVisualizer = ToxicityVisualizer
    return module


def build_aspect_sentiment_module():
    module = types.ModuleType("aspect_sentiment")

    class AspectSentimentAnalyzer:
        def analyze_aspects(self, text):
            return {{
                "service": {{
                    "label": "POSITIVE",
                    "confidence": 0.88,
                    "sentences": [{{
                        "sentence": "The service was excellent.",
                        "label": "POSITIVE",
                        "confidence": 0.88,
                    }}],
                }},
                "price": {{
                    "label": "NEGATIVE",
                    "confidence": 0.76,
                    "sentences": [{{
                        "sentence": "The price was high.",
                        "label": "NEGATIVE",
                        "confidence": 0.76,
                    }}],
                }},
            }}

    module.AspectSentimentAnalyzer = AspectSentimentAnalyzer
    return module


def build_batch_sentiment_module():
    module = types.ModuleType("batch_sentiment")

    class BatchSentimentProcessor:
        def process_texts(self, texts):
            import pandas as pd

            rows = []
            for text in texts:
                value = text.lower()
                if "good" in value:
                    label, confidence = "POSITIVE", 0.90
                elif "bad" in value:
                    label, confidence = "NEGATIVE", 0.86
                else:
                    label, confidence = "NEUTRAL", 0.60

                rows.append({{
                    "text": text,
                    "sentiment": label,
                    "confidence": confidence,
                    "language": "en",
                    "length": len(text),
                }})

            return pd.DataFrame(rows)

    module.BatchSentimentProcessor = BatchSentimentProcessor
    return module


def build_sentiment_viz_module():
    module = types.ModuleType("sentiment_viz")

    class SentimentVisualizer:
        @staticmethod
        def create_pie_chart(df):
            import plotly.graph_objects as go
            return go.Figure()

        @staticmethod
        def create_bar_chart(df):
            import plotly.graph_objects as go
            return go.Figure()

    module.SentimentVisualizer = SentimentVisualizer
    return module


if "deepfake_detector_advanced" in SELECTED_STUBS:
    sys.modules["deepfake_detector_advanced"] = build_deepfake_detector_module()

if "safe_deepfake_runtime" in SELECTED_STUBS:
    sys.modules["safe_deepfake_runtime"] = build_safe_deepfake_runtime_module()

if "fake_news_detector" in SELECTED_STUBS:
    sys.modules["fake_news_detector"] = build_fake_news_module()

if "realtime_verifier" in SELECTED_STUBS:
    sys.modules["realtime_verifier"] = build_realtime_verifier_module()

if "sentiment_analyzer" in SELECTED_STUBS:
    sys.modules["sentiment_analyzer"] = build_sentiment_module()

if "toxicity_detector" in SELECTED_STUBS:
    sys.modules["toxicity_detector"] = build_toxicity_module()

if "toxicity_viz" in SELECTED_STUBS:
    sys.modules["toxicity_viz"] = build_toxicity_viz_module()

if "aspect_sentiment" in SELECTED_STUBS:
    sys.modules["aspect_sentiment"] = build_aspect_sentiment_module()

if "batch_sentiment" in SELECTED_STUBS:
    sys.modules["batch_sentiment"] = build_batch_sentiment_module()

if "sentiment_viz" in SELECTED_STUBS:
    sys.modules["sentiment_viz"] = build_sentiment_viz_module()

runpy.run_path(APP_FILE, run_name="__main__")
"""


class TruthGuardSeleniumBase(unittest.TestCase):
    initial_users = {}
    stub_modules = []

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        (cls.temp_path / "users.json").write_text(
            json.dumps(cls.initial_users or {}, indent=2),
            encoding="utf-8",
        )

        launcher_file = cls.temp_path / "launcher.py"
        launcher_file.write_text(
            _launcher_source(str(APP_FILE), str(PROJECT_ROOT), cls.stub_modules),
            encoding="utf-8",
        )

        cls.port = _free_port()
        cls.base_url = f"http://127.0.0.1:{cls.port}"

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        cls.app_process = subprocess.Popen(
            [
                sys.executable,
                str(launcher_file),
                "--server.headless",
                "true",
                "--server.port",
                str(cls.port),
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=str(cls.temp_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        deadline = time.time() + 90
        while time.time() < deadline:
            if cls.app_process.poll() is not None:
                raise RuntimeError("TruthGuard AI server exited before startup.")
            try:
                urllib.request.urlopen(cls.base_url, timeout=2)
                return
            except Exception:
                time.sleep(1)

        raise RuntimeError("Timed out waiting for TruthGuard AI server to start.")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "app_process") and cls.app_process and cls.app_process.poll() is None:
            cls.app_process.terminate()
            try:
                cls.app_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.app_process.kill()

        if hasattr(cls, "temp_dir"):
            cls.temp_dir.cleanup()

    def setUp(self):
        self.driver = self._build_driver()
        self.driver.get(self.base_url)
        self.wait_for_text("Login", timeout=40)

    def tearDown(self):
        if hasattr(self, "driver"):
            self.driver.quit()

    def _build_driver(self):
        browser = os.getenv("TRUTHGUARD_BROWSER", "edge").lower()
        headless = os.getenv("TRUTHGUARD_HEADLESS", "1") == "1"

        if browser == "chrome":
            options = webdriver.ChromeOptions()
            if headless:
                options.add_argument("--headless=new")
            options.add_argument("--window-size=1440,1200")
            return webdriver.Chrome(options=options)

        if browser == "firefox":
            options = webdriver.FirefoxOptions()
            if headless:
                options.add_argument("-headless")
            return webdriver.Firefox(options=options)

        options = webdriver.EdgeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--window-size=1440,1200")
        return webdriver.Edge(options=options)

    def _xpath_literal(self, value):
        if "'" not in value:
            return f"'{value}'"
        if '"' not in value:
            return f'"{value}"'
        parts = value.split("'")
        return "concat(" + ", \"'\", ".join(f"'{part}'" for part in parts) + ")"

    def _find_visible(self, xpaths, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            for xpath in xpaths:
                for element in self.driver.find_elements(By.XPATH, xpath):
                    try:
                        if element.is_displayed():
                            return element
                    except Exception:
                        pass
            time.sleep(0.25)
        raise AssertionError(f"Visible element not found for xpaths: {xpaths}")

    def wait_for_text(self, text, timeout=20):
        literal = self._xpath_literal(text)
        return self._find_visible([f"//*[contains(normalize-space(.), {literal})]"], timeout=timeout)

    def click_text(self, text, timeout=20):
        literal = self._xpath_literal(text)
        element = self._find_visible(
            [
                f"//button[contains(normalize-space(.), {literal})]",
                f"//*[@role='tab'][contains(normalize-space(.), {literal})]",
                f"//label[contains(normalize-space(.), {literal})]",
            ],
            timeout=timeout,
        )
        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
        self.driver.execute_script("arguments[0].click();", element)

    def fill_input(self, label, value, timeout=20):
        literal = self._xpath_literal(label)
        element = self._find_visible(
            [
                f"//*[@data-testid='stTextInput'][.//label[contains(normalize-space(.), {literal})]]//input",
                f"//label[contains(normalize-space(.), {literal})]/following::input[1]",
            ],
            timeout=timeout,
        )
        element.click()
        element.send_keys(Keys.CONTROL, "a")
        element.send_keys(Keys.DELETE)
        element.send_keys(value)

    def fill_textarea(self, label, value, timeout=20):
        literal = self._xpath_literal(label)
        element = self._find_visible(
            [
                f"//*[@data-testid='stTextArea'][.//label[contains(normalize-space(.), {literal})]]//textarea",
                f"//label[contains(normalize-space(.), {literal})]/following::textarea[1]",
            ],
            timeout=timeout,
        )
        element.click()
        element.send_keys(Keys.CONTROL, "a")
        element.send_keys(Keys.DELETE)
        element.send_keys(value)

    def upload_file(self, file_path, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            for element in inputs:
                try:
                    self.driver.execute_script(
                        "arguments[0].style.display='block';"
                        "arguments[0].style.visibility='visible';",
                        element,
                    )
                    element.send_keys(str(file_path))
                    return
                except Exception:
                    pass
            time.sleep(0.25)
        raise AssertionError("File input not found.")

    def make_test_image(self, name="sample.png"):
        image_path = self.temp_path / name
        Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_path)
        return image_path

    def make_test_text_file(self, name="sample.txt", content="good support\nbad quality\nnormal update\n"):
        file_path = self.temp_path / name
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def login(self, username, password):
        self.fill_input("Username", username)
        self.fill_input("Password", password)
        self.click_text("Login")
        self.wait_for_text("Launch Verifier", timeout=40)
