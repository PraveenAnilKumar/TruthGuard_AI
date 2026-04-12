import logging
import sys
from fake_news_detector import FakeNewsDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_robust_prediction():
    print("=== Testing FakeNewsDetector Robustness ===")
    detector = FakeNewsDetector()
    
    # Test 1: String input (Normal)
    print("\nTest 1: Normal string input")
    try:
        score = detector._calculate_clickbait_score("Is this clickbait?")
        print(f"✅ String score: {score}")
    except Exception as e:
        print(f"❌ String failed: {e}")

    # Test 2: List input (The reported bug)
    print("\nTest 2: List input (simulating problematic translator output)")
    try:
        # Previously this would fail with 'list' object has no attribute 'strip'
        score = detector._calculate_clickbait_score(["This", "is", "a", "list", "?"])
        print(f"✅ List score: {score}")
    except Exception as e:
        print(f"❌ List failed: {e}")

    # Test 3: Traditional prediction with list-y input (mocked)
    print("\nTest 3: Traditional prediction robustness")
    # We don't necessarily need a trained model here if we just want to see it catch the error
    # but let's try calling it on a known empty model state
    detector.model = None 
    try:
        label, conf = detector._predict_traditional("Some text")
        print(f"✅ Prediction fallback handled: {label} ({conf})")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

    print("\n=== Robustness tests complete ===")

if __name__ == "__main__":
    test_robust_prediction()
