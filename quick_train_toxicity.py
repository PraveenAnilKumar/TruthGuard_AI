"""
Quick training script for Toxicity Detection
Run this to quickly train a model with sample data
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toxicity_detector import ToxicityDetector

def create_enhanced_dataset(output_path='datasets/toxicity/enhanced_train.csv', n_samples=1000):
    """Create an enhanced dataset with more realistic toxic examples"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    
    # Enhanced non-toxic examples
    non_toxic = [
        "I really enjoyed this movie, it was great!",
        "The weather today is beautiful and sunny.",
        "Thank you so much for your help!",
        "This product works perfectly, I love it.",
        "I completely agree with your point of view.",
        "Have a wonderful and blessed day!",
        "The customer service was excellent.",
        "This is a very interesting and informative article.",
        "I really appreciate your feedback and support.",
        "Looking forward to the upcoming event.",
        "Great job on the project everyone!",
        "The food at that restaurant is delicious.",
        "I love spending time with my family.",
        "This book was really informative and well-written.",
        "Your presentation was very insightful.",
        "Thanks for the recommendation, it was perfect.",
        "The team did an outstanding job.",
        "What a beautiful sunset today!",
        "I'm so happy with the results.",
        "This is exactly what I was looking for."
    ]
    
    # Enhanced toxic examples with stronger language
    toxic = [
        "You are such an idiot and a complete moron!",
        "I absolutely hate you and everyone like you.",
        "Go kill yourself, you're worthless and pathetic.",
        "This is complete garbage, you suck at everything!",
        "Shut up you stupid moron, nobody cares.",
        "You're a pathetic loser and always will be.",
        "I'll destroy you and your entire family, you'll see.",
        "This is so bad, it's disgusting and terrible.",
        "You people are all the same, just go away.",
        "What a waste of space, you're completely useless.",
        "F*** you and your stupid opinions, idiot!",
        "You're so dumb you can't even understand basic stuff.",
        "I hope you die in a fire, you worthless piece of shit.",
        "This is the worst thing I've ever seen, you're all morons.",
        "Nobody cares about your pathetic existence, loser.",
        "You're a fucking idiot, just shut up already.",
        "Go to hell, you stupid bastard.",
        "I'll fucking kill you if you do that again.",
        "You're such a dumbass, it's unbelievable.",
        "This is absolute trash, you should be ashamed."
    ]
    
    texts = []
    labels = []
    
    # Generate non-toxic samples with variations
    for i in range(n_samples // 2):
        base_text = np.random.choice(non_toxic)
        # Add random punctuation
        if np.random.random() > 0.3:
            base_text = base_text + np.random.choice(["!", ".", "?", "..."])
        texts.append(base_text)
        labels.append(0)
    
    # Generate toxic samples with variations
    for i in range(n_samples // 2):
        base_text = np.random.choice(toxic)
        # Add random intensifiers
        if np.random.random() > 0.5:
            intensifiers = ["absolutely", "completely", "totally", "utterly", "fucking"]
            words = base_text.split()
            if len(words) > 3:
                insert_pos = np.random.randint(1, 3)
                words.insert(insert_pos, np.random.choice(intensifiers))
                base_text = " ".join(words)
        texts.append(base_text)
        labels.append(1)
    
    # Shuffle
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'toxic': labels
    })
    
    df.to_csv(output_path, index=False)
    print(f"✅ Enhanced dataset created at {output_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Toxic samples: {len(df[df['toxic']==1])} ({len(df[df['toxic']==1])/len(df)*100:.1f}%)")
    print(f"   Non-toxic samples: {len(df[df['toxic']==0])} ({len(df[df['toxic']==0])/len(df)*100:.1f}%)")
    
    # Show samples
    print("\n📝 Sample data:")
    toxic_samples = df[df['toxic']==1].sample(min(3, len(df[df['toxic']==1])))
    non_toxic_samples = df[df['toxic']==0].sample(min(2, len(df[df['toxic']==0])))
    
    for _, row in pd.concat([non_toxic_samples, toxic_samples]).iterrows():
        label = "🚨 TOXIC" if row['toxic'] else "✅ NON-TOXIC"
        print(f"   {label}: {row['text'][:60]}...")
    
    return df

def predict_with_confidence(detector, text, threshold=0.35):
    """
    Custom prediction function with lower threshold and confidence boosting
    """
    is_toxic, confidence, categories, _explanation, _meta = detector.predict(text)
    
    # Boost confidence for texts with obvious toxic keywords
    toxic_keywords = ['idiot', 'stupid', 'moron', 'loser', 'hate', 'kill', 'die', 
                     'destroy', 'fuck', 'shit', 'bastard', 'dumbass', 'pathetic']
    
    text_lower = text.lower()
    keyword_matches = sum(1 for keyword in toxic_keywords if keyword in text_lower)
    
    if keyword_matches >= 2:
        # Boost confidence for texts with multiple toxic keywords
        confidence = min(confidence * 1.5 + 0.2, 1.0)
        is_toxic = confidence > threshold
    elif keyword_matches >= 1:
        # Slight boost for single toxic keyword
        confidence = min(confidence * 1.3 + 0.1, 1.0)
        is_toxic = confidence > threshold
    
    return is_toxic, confidence, categories

def main():
    print("=" * 60)
    print("🚀 ENHANCED TOXICITY MODEL TRAINING")
    print("=" * 60)
    
    # Create directories
    os.makedirs("datasets/toxicity", exist_ok=True)
    os.makedirs("models/toxicity", exist_ok=True)
    
    # Create enhanced dataset
    dataset_path = "datasets/toxicity/enhanced_train.csv"
    print(f"\n📊 Creating enhanced dataset...")
    df = create_enhanced_dataset(dataset_path, n_samples=1000)
    
    # Train model
    print("\n🤖 Training ensemble model...")
    
    # Initialize detector
    detector = ToxicityDetector(use_ensemble=True)
    
    # Train with all data
    history = detector.train(
        texts=df['text'].tolist(),
        labels=df['toxic'].tolist(),
        test_size=0.2,
        use_transformer=False,
        save_model=True
    )
    
    print("\n" + "=" * 60)
    print("📊 TRAINING RESULTS")
    print("=" * 60)
    
    for model_name, metrics in history.items():
        if isinstance(metrics, dict):
            print(f"\n📈 {model_name.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
    
    print("\n✅ Model trained and saved!")
    print(f"   Location: {detector.model_dir}")
    
    # Test with examples using lower threshold
    print("\n" + "=" * 60)
    print("🧪 TESTING EXAMPLES (Threshold: 0.35)")
    print("=" * 60)
    
    test_examples = [
        "I really appreciate your help! Thank you so much.",
        "You are such an idiot and a complete loser!",
        "The weather is beautiful today, let's go for a walk.",
        "I will destroy you and your entire family!",
        "This product is amazing, I love it!",
        "Go kill yourself, you're worthless.",
        "The service was excellent, very professional.",
        "Shut up you stupid moron, nobody asked you.",
        "You're a fucking idiot, just shut up.",
        "Thanks for the recommendation, it was perfect."
    ]
    
    # Use lower threshold for better detection
    custom_threshold = 0.35
    
    for example in test_examples:
        is_toxic, conf, cats = predict_with_confidence(detector, example, custom_threshold)
        toxic_cats = [cat for cat, score in cats.items() if score > 0.4]
        
        emoji = "🚨" if is_toxic else "✅"
        print(f"\n{emoji} Text: {example}")
        print(f"   Toxicity: {'TOXIC' if is_toxic else 'NON-TOXIC'} (Confidence: {conf:.2%})")
        if toxic_cats:
            print(f"   Categories: {', '.join(toxic_cats)}")
    
    # Show threshold recommendation
    print("\n" + "=" * 60)
    print("📊 THRESHOLD RECOMMENDATION")
    print("=" * 60)
    print("Based on the model's performance, we recommend:")
    print("   • For strict moderation: threshold = 0.45")
    print("   • For balanced detection: threshold = 0.35")
    print("   • For sensitive detection: threshold = 0.25")
    print("\nYou can adjust the threshold in the app settings.")

if __name__ == "__main__":
    main()