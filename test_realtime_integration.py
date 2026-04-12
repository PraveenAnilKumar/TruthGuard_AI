import os
import sys

# Ensure current directory is in path
sys.path.insert(0, os.getcwd())

print("="*50)
print("Testing Real-time Integration")
print("="*50)

try:
    from fake_news_detector import fake_news_detector
    print("✅ Fake news detector loaded")
except Exception as e:
    print(f"❌ Error loading fake_news_detector: {e}")
    sys.exit(1)

test_text = "Scientists have discovered a new planet made entirely of diamonds in the Andromeda galaxy."
print(f"\nTesting text: {test_text}")

try:
    label, conf, click_score, meta = fake_news_detector.predict(test_text, check_realtime=True)
    
    print(f"\nLabel: {label}")
    print(f"Confidence: {conf:.2%}")
    print(f"Clickbait Score: {click_score:.2f}")
    print(f"Real-time Result present: {meta.get('realtime_result') is not None}")

    if meta.get('realtime_result'):
        rt = meta['realtime_result']
        print(f"\nReal-time Status: {rt.get('status')}")
        print(f"Real-time Score: {rt.get('consensus_score', 0):.2%}")
        if rt.get('sources'):
            print(f"Top Source: {rt['sources'][0]['title'][:100]}")
            print(f"Source URL: {rt['sources'][0]['url']}")
    
    print("\n✅ Test completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error during prediction: {e}")
    import traceback
    traceback.print_exc()