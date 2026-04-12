import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

print("="*50)
print("Testing Imports")
print("="*50)

# Test realtime_verifier
print("\n1. Testing realtime_verifier...")
try:
    from realtime_verifier import realtime_verifier
    print(f"   realtime_verifier type: {type(realtime_verifier)}")
    print("   ✅ realtime_verifier imported successfully")
except Exception as e:
    print(f"   ❌ Error importing realtime_verifier: {e}")

# Test fake_news_detector
print("\n2. Testing fake_news_detector...")
try:
    from fake_news_detector import fake_news_detector
    print(f"   fake_news_detector type: {type(fake_news_detector)}")
    print("   ✅ fake_news_detector imported successfully")
except Exception as e:
    print(f"   ❌ Error importing fake_news_detector: {e}")

# Test sentiment_analyzer
print("\n3. Testing sentiment_analyzer...")
try:
    from sentiment_analyzer import sentiment_analyzer
    print(f"   sentiment_analyzer type: {type(sentiment_analyzer)}")
    print("   ✅ sentiment_analyzer imported successfully")
except Exception as e:
    print(f"   ❌ Error importing sentiment_analyzer: {e}")

# Test toxicity_detector
print("\n4. Testing toxicity_detector...")
try:
    from toxicity_detector import toxicity_detector
    print(f"   toxicity_detector type: {type(toxicity_detector)}")
    print("   ✅ toxicity_detector imported successfully")
except Exception as e:
    print(f"   ❌ Error importing toxicity_detector: {e}")

# Test deepfake_detector
print("\n5. Testing deepfake_detector...")
try:
    from deepfake_detector_advanced import deepfake_detector
    print(f"   deepfake_detector type: {type(deepfake_detector)}")
    print("   ✅ deepfake_detector imported successfully")
except Exception as e:
    print(f"   ❌ Error importing deepfake_detector: {e}")

print("\n" + "="*50)
print("Import test complete!")
print("="*50)