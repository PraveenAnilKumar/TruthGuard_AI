import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_explanation():
    print("="*50)
    print("Starting Toxicity Explanation Verification...")
    print("="*50)
    
    # Initialize detector
    print("\n1. Loading ToxicityDetector...")
    try:
        from toxicity_detector import ToxicityDetector
        detector = ToxicityDetector(use_ensemble=True)
        print("   ✅ Detector loaded")
    except Exception as e:
        print(f"   ❌ Error loading detector: {e}")
        return
    
    # Load visualizer
    print("\n2. Loading ToxicityVisualizer...")
    try:
        from toxicity_viz import ToxicityVisualizer
        viz = ToxicityVisualizer()
        print("   ✅ Visualizer loaded")
    except Exception as e:
        print(f"   ❌ Error loading visualizer: {e}")
        return
    
    # Test case 1: Personal insult
    test_text = "You are a total idiot and a stupid loser."
    print(f"\n3. Testing: '{test_text}'")
    
    try:
        is_toxic, conf, cats, explanation, meta = detector.predict(test_text)
        
        print(f"   Result: {'TOXIC' if is_toxic else 'NON-TOXIC'} ({conf:.1%})")
        print(f"   Reasons: {explanation['reasons']}")
        print(f"   Matched words: {explanation['all_matched_words']}")
        
        # Verify expected words are matched
        expected_words = ['idiot', 'stupid', 'loser']
        found_words = explanation['all_matched_words']
        for w in expected_words:
            if w in found_words:
                print(f"   ✅ Found expected word: {w}")
            else:
                print(f"   ⚠️ Missing expected word: {w} (may be due to stemming)")
                
        # Test HTML generation
        print("\n4. Testing UI Rendering...")
        highlighted_html = ToxicityVisualizer.render_toxic_highlights(test_text, explanation)
        if '<span' in highlighted_html:
            print("   ✅ HTML highlighting generated successfully")
        else:
            print("   ❌ HTML highlighting failed")
            
        # Test explanation card
        card_html = ToxicityVisualizer.create_explanation_card('insult', ['idiot', 'stupid'], 0.85)
        if 'Insult' in card_html and '85.0%' in card_html:
            print("   ✅ Explanation card generated successfully")
        else:
            print("   ❌ Explanation card failed")
            
        print("\n✅ Verification Complete!")
        
    except Exception as e:
        print(f"   ❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_explanation()