
import sys
import os
import gc
import logging
from typing import List

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.getcwd())

from fake_news_detector import FakeNewsDetector

def test_granular_loading():
    print("\n--- Testing Granular Loading and Memory Management ---")
    detector = FakeNewsDetector()
    
    # 1. Check available models
    hf_models = detector.get_huggingface_models()
    print(f"Available HF models: {hf_models}")
    
    local_models = detector.get_available_models()
    local_paths = [m['path'] for m in local_models]
    print(f"Available Local models: {[m['name'] for m in local_models]}")
    
    # 2. Load a specific HF model
    if hf_models:
        target_hf = hf_models[0]
        print(f"\nRequesting only HF model: {target_hf}")
        detector.ensure_model_loaded(requested_models=[target_hf])
        
        loaded = list(detector._loaded_hf_pipelines.keys())
        print(f"Actually loaded HF models: {loaded}")
        assert target_hf in loaded
        assert len(loaded) == 1
        
        # 3. Predict with only this model
        print("\nPredicting with selected model...")
        label, conf, click, meta = detector.predict("Test news article", requested_models=[target_hf])
        print(f"Result: {label} ({conf:.2f})")
        print(f"Individual scores: {meta.get('individual_scores')}")
        assert target_hf in meta.get('individual_scores')
        assert len(meta.get('individual_scores')) == 1
        
        # 4. Unload and switch
        # If we had multiple HF models, we'd test swapping. 
        # Since we only have one enabled by default, let's test unloading it by requesting a local one.
        if local_paths:
            target_local = local_paths[0]
            print(f"\nRequesting only Local model: {target_local}")
            detector.ensure_model_loaded(requested_models=[target_local])
            
            loaded = list(detector._loaded_hf_pipelines.keys())
            print(f"Actually loaded HF models after swap: {loaded}")
            # The HF model should have been unloaded
            assert len(loaded) == 0
            
            print("\nPredicting with selected local model...")
            label, conf, click, meta = detector.predict("Test news article", requested_models=[target_local])
            print(f"Result: {label} ({conf:.2f})")
            print(f"Individual scores: {meta.get('individual_scores')}")
            # Should only have one result from a local model
            assert len(meta.get('individual_scores')) == 1
            
    print("\n✅ Granular loading and unloading test PASSED!")

if __name__ == "__main__":
    try:
        test_granular_loading()
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
