import os
import sys
import traceback
import importlib

def analyze_system():
    print("=== Analyzing TruthGuard AI System ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    modules_to_test = [
        "app",
        "deepfake_detector_advanced",
        "fake_news_detector",
        "sentiment_analyzer",
        "toxicity_detector",
        "toxicity_viz",
        "utils",
        "aspect_sentiment",
        "batch_sentiment"
    ]
    
    conflicts_found = False
    successful = []
    failed = []
    
    for module in modules_to_test:
        try:
            print(f"\nTesting import of {module}...")
            mod = __import__(module)
            print(f"✅ {module} loaded successfully from {mod.__file__ if hasattr(mod, '__file__') else 'unknown'}")
            successful.append(module)
        except Exception as e:
            conflicts_found = True
            print(f"❌ Error loading {module}:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            traceback.print_exc()
            failed.append(module)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Successful imports ({len(successful)}):")
    for m in successful:
        print(f"  ✅ {m}")
    
    if failed:
        print(f"\nFailed imports ({len(failed)}):")
        for m in failed:
            print(f"  ❌ {m}")
    
    if conflicts_found:
        print("\n⚠️ Conflicts or errors were found in the system.")
        print("   Check the error messages above for details.")
    else:
        print("\n✅ No import conflicts or syntax errors detected in major modules.")
    
    return successful, failed

if __name__ == "__main__":
    successful, failed = analyze_system()