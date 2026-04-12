import numpy as np
import cv2
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_hf_integration():
    print("="*50)
    print("Testing HuggingFace Integration")
    print("="*50)
    
    print("\nInitializing detector...")
    try:
        from deepfake_detector_advanced import DeepfakeDetectorAdvanced
        detector = DeepfakeDetectorAdvanced()
        print("✅ Detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    print(f"\nAvailable HF models: {list(detector.available_hf_models.keys())}")
    
    # Create a dummy image (random noise)
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    test_models = ["HF_Transformer_V1", "HF_Transformer_V2", "SDXL_Detector"]
    
    for model_name in test_models:
        print(f"\n--- Testing {model_name} ---")
        try:
            # We use detect_deepfake_ensemble which will lazy-load the model
            res = detector.detect_deepfake_ensemble(dummy_img, requested_models=[model_name])
            print(f"Result: {res.get('is_deepfake')}, Score: {res.get('ensemble_score')}")
            
            if model_name in detector.model_names:
                print(f"✅ {model_name} successfully loaded and scored")
            else:
                print(f"❌ {model_name} failed to load")
        except Exception as e:
            print(f"❌ Error during testing {model_name}: {e}")
    
    print("\n" + "="*50)
    print("Test complete!")
    print("="*50)

if __name__ == "__main__":
    test_hf_integration()