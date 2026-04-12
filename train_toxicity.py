"""
Training script for Toxicity Detection Models
Run this to train and save toxicity detection models
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toxicity_detector import ToxicityDetector

def create_sample_dataset(output_path='datasets/toxicity/sample_toxicity.csv'):
    """Create a sample toxicity dataset for testing"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample data with labels
    data = {
        'text': [
            # Non-toxic examples
            "I really enjoyed this movie, it was great!",
            "The weather today is beautiful.",
            "Thank you for your help!",
            "This product works perfectly.",
            "I agree with your point of view.",
            "Have a wonderful day!",
            "The service was excellent.",
            "This is a very interesting article.",
            "I appreciate your feedback.",
            "Looking forward to the event.",
            "Can you please explain that again?",
            "The food at that restaurant is delicious.",
            "I love spending time with my family.",
            "This book was really informative.",
            "Great job on the project everyone!",
            
            # Toxic examples
            "You are such an idiot!",
            "I hate you and everyone like you.",
            "Go kill yourself, you're worthless.",
            "This is complete garbage, you suck!",
            "Shut up you stupid moron.",
            "You're a pathetic loser.",
            "I'll destroy you and your family.",
            "This is so bad, it's disgusting.",
            "You people are all the same, go away.",
            "What a waste of space, you're useless.",
            "F*** you and your stupid opinions!",
            "You're so dumb you can't even understand basic stuff.",
            "I hope you die in a fire.",
            "This is the worst thing I've ever seen, you're all morons.",
            "Nobody cares about your pathetic existence."
        ],
        'toxic': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 15 non-toxic
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1   # 15 toxic
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Sample dataset created at {output_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Toxic samples: {len(df[df['toxic']==1])}")
    print(f"   Non-toxic samples: {len(df[df['toxic']==0])}")
    
    return df

def download_jigsaw_dataset(output_path='datasets/toxicity/jigsaw_train.csv', sample_size=10000):
    """
    Download Jigsaw Toxic Comment dataset
    Note: This requires the dataset to be downloaded from Kaggle
    """
    print("📥 To download the full Jigsaw dataset:")
    print("   1. Go to: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data")
    print("   2. Download train.csv.zip")
    print("   3. Extract and place in datasets/toxicity/")
    print("\nFor now, using sample dataset...")
    return create_sample_dataset(output_path)

def main():
    parser = argparse.ArgumentParser(description='Train Toxicity Detection Model')
    parser.add_argument('--dataset', type=str, default='datasets/toxicity/sample_toxicity.csv',
                        help='Path to training dataset CSV')
    parser.add_argument('--model-type', type=str, default='ensemble',
                        choices=['ensemble', 'logistic', 'svm', 'random_forest', 'transformer'],
                        help='Type of model to train')
    parser.add_argument('--use-transformer', action='store_true',
                        help='Use transformer model (requires transformers)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Training epochs for transformer')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Validation split size')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create a sample dataset if none exists')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 TOXICITY DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        if args.create_sample:
            print(f"📊 Dataset not found. Creating sample dataset...")
            create_sample_dataset(args.dataset)
        else:
            print(f"❌ Dataset not found: {args.dataset}")
            print("   Use --create-sample to create a sample dataset")
            return
    
    # Load dataset
    print(f"\n📂 Loading dataset from {args.dataset}")
    df = pd.read_csv(args.dataset)
    
    # Check required columns
    if 'text' not in df.columns or 'toxic' not in df.columns:
        print("❌ Dataset must contain 'text' and 'toxic' columns")
        print(f"   Available columns: {list(df.columns)}")
        return
    
    # Display dataset info
    print(f"\n📊 Dataset Info:")
    print(f"   Total samples: {len(df)}")
    print(f"   Toxic samples: {len(df[df['toxic']==1])} ({len(df[df['toxic']==1])/len(df)*100:.1f}%)")
    print(f"   Non-toxic samples: {len(df[df['toxic']==0])} ({len(df[df['toxic']==0])/len(df)*100:.1f}%)")
    
    # Show sample
    print("\n📝 Sample data:")
    for i in range(min(3, len(df))):
        print(f"   {i+1}. {df.iloc[i]['text'][:80]}... -> {'TOXIC' if df.iloc[i]['toxic'] else 'NON-TOXIC'}")
    
    # Initialize detector
    print("\n🤖 Initializing Toxicity Detector...")
    detector = ToxicityDetector(use_ensemble=(args.model_type == 'ensemble'))
    
    # Train model
    print(f"\n🚀 Training {args.model_type} model...")
    
    if args.use_transformer or args.model_type == 'transformer':
        print("   Using transformer model (this may take a while)...")
        history = detector.train(
            texts=df['text'].tolist(),
            labels=df['toxic'].tolist(),
            use_transformer=True,
            epochs=args.epochs,
            save_model=True
        )
    else:
        history = detector.train(
            texts=df['text'].tolist(),
            labels=df['toxic'].tolist(),
            test_size=args.test_size,
            use_transformer=False,
            save_model=True
        )
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 TRAINING RESULTS")
    print("=" * 60)
    
    if isinstance(history, dict):
        if 'error' in history:
            print(f"❌ Error: {history['error']}")
        else:
            for model_name, metrics in history.items():
                if isinstance(metrics, dict):
                    print(f"\n📈 {model_name.upper()} Model:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"   {metric}: {value:.4f}")
    
    print("\n✅ Training complete!")
    print(f"   Model saved in: {detector.model_dir}")
    
    # Test with some examples
    print("\n" + "=" * 60)
    print("🧪 TESTING WITH EXAMPLES")
    print("=" * 60)
    
    test_examples = [
        "I love this product, it's amazing!",
        "You are such an idiot and a loser!",
        "The weather today is quite pleasant.",
        "I will destroy you and your entire family!",
        "Thank you for your excellent service."
    ]
    
    for example in test_examples:
        is_toxic, confidence, categories, explanation, meta = detector.predict(example)
        
        print(f"\n📝 Text: {example}")
        print(f"   Toxicity: {'🚨 TOXIC' if is_toxic else '✅ NON-TOXIC'}")
        print(f"   Confidence: {confidence:.2%}")
        
        if is_toxic:
            toxic_cats = [cat for cat, score in categories.items() if score > 0.5]
            if toxic_cats:
                print(f"   Categories: {', '.join(toxic_cats)}")

if __name__ == "__main__":
    main()