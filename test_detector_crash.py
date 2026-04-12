import os
import sys
import logging

# Set up logging to match app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_load():
    try:
        logger.info("Importing FakeNewsDetector...")
        from fake_news_detector import FakeNewsDetector
        
        logger.info("Initializing FakeNewsDetector...")
        detector = FakeNewsDetector()
        
        logger.info("Attempting to load transformer model...")
        model_path = r"models/fake_news/transformer_distilbert-base-uncased_20260306_171509"
        
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            return
            
        success = detector.load_transformer_model(model_path)
        if success:
            logger.info("✅ Transformer model loaded successfully in test script")
        else:
            logger.error("❌ Transformer model load failed in test script")
            
    except Exception as e:
        logger.exception(f"CRASH in test script: {e}")

if __name__ == "__main__":
    test_load()
