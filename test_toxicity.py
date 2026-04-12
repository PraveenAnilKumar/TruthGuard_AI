# test_toxicity.py
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

print("="*50)
print("Testing Toxicity Modules")
print("="*50)

print("\nTesting toxicity_detector import...")
try:
    from toxicity_detector import ToxicityDetector, toxicity_detector
    print("✅ toxicity_detector imported successfully")
except Exception as e:
    print(f"❌ Error importing toxicity_detector: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting toxicity_viz import...")
try:
    from toxicity_viz import ToxicityVisualizer, toxicity_viz
    print("✅ toxicity_viz imported successfully")
except Exception as e:
    print(f"❌ Error importing toxicity_viz: {e}")
    import traceback
    traceback.print_exc()

# Test creating detector
print("\nTesting ToxicityDetector creation...")
try:
    detector = ToxicityDetector(use_ensemble=True)
    print("✅ ToxicityDetector created successfully")
    print(f"   Model trained: {detector.is_trained}")
    print(f"   Available models: {list(detector.models.keys()) if detector.models else 'None'}")
except Exception as e:
    print(f"❌ Error creating ToxicityDetector: {e}")
    import traceback
    traceback.print_exc()

# Test prediction
print("\nTesting prediction...")
try:
    test_text = "You are such an idiot!"
    is_toxic, conf, cats, explanation, meta = detector.predict(test_text)
    print(f"   Text: {test_text}")
    print(f"   Is Toxic: {is_toxic}")
    print(f"   Confidence: {conf:.2%}")
    print("✅ Prediction successful")
except Exception as e:
    print(f"❌ Error during prediction: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Test complete!")
print("="*50)