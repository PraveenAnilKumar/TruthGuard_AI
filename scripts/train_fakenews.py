#!/usr/bin/env python
"""
train_fakenews.py - Optimized Train Fake News Detection Model
Usage: python train_fakenews.py --dataset path/to/data.csv [options]
"""

import argparse
import os
import pandas as pd
import torch
from fake_news_detector import FakeNewsDetector
import json
from datetime import datetime

def check_gpu():
    """Check and report GPU availability"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {gpu_name} ({gpu_count} device(s))")
        return True
    else:
        print("⚠️  No GPU detected - training will be SLOW on CPU")
        return False

def main():
    parser = argparse.ArgumentParser(description='Optimized Fake News Detector Training')
    parser.add_argument('--dataset', required=True, help='Path to CSV file')
    parser.add_argument('--text-col', default='text', help='Column containing text')
    parser.add_argument('--label-col', default='label', help='Column containing label')
    parser.add_argument('--transformer', action='store_true', help='Use transformer model')
    parser.add_argument('--model-name', default='distilbert-base-uncased', 
                       help='Transformer model name (use smaller models for speed)')
    parser.add_argument('--save-path', default='models/fake_news/', help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--test-size', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--max-samples', type=int, default=None, help='Max rows to load')
    parser.add_argument('--fast-mode', action='store_true', 
                       help='Use faster but smaller model (MiniLM)')
    parser.add_argument('--mixed-precision', action='store_true', 
                       help='Use mixed precision training (faster on GPU)')
    parser.add_argument('--num-workers', type=int, default=4, 
                       help='Number of data loading workers')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--save-every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()

    print("="*50)
    print("🚀 Optimized Fake News Training Started")
    print("="*50)
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Override model name if fast mode is enabled
    if args.fast_mode:
        args.model_name = "microsoft/MiniLM-L12-H384-uncased"
        print(f"⚡ Fast mode enabled - using {args.model_name}")
    
    # Print configuration
    print(f"\n📊 Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model_name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Mixed precision: {args.mixed_precision and has_gpu}")
    print(f"  Data workers: {args.num_workers}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    
    # Load data
    print(f"\n📁 Loading data...")
    temp_file = None
    if args.max_samples:
        print(f"  Using first {args.max_samples} rows")
        df = pd.read_csv(args.dataset, nrows=args.max_samples)
        temp_file = "temp_sample.csv"
        df.to_csv(temp_file, index=False)
        dataset_path = temp_file
    else:
        dataset_path = args.dataset
    
    # Create detector
    detector = FakeNewsDetector(use_transformer=args.transformer, model_name=args.model_name)
    print("✅ FakeNewsDetector created")
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(args.save_path, f"run_{timestamp}")
    os.makedirs(model_save_path, exist_ok=True)
    
    # Save training arguments
    with open(os.path.join(model_save_path, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n🔧 Starting training (saving to {model_save_path})...")
    
    # Custom training loop with optimizations
    try:
        # Train the model with all optimizations
        detector.train(
            csv_path=dataset_path,
            text_column=args.text_col,
            label_column=args.label_col,
            test_size=args.test_size,
            save_path=model_save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            mixed_precision=args.mixed_precision and has_gpu,
            num_workers=args.num_workers,
            gradient_accumulation_steps=args.gradient_accumulation,
            resume_from=args.resume_from,
            save_every=args.save_every
        )
        
        print("\n" + "="*50)
        print("✅ Training completed successfully!")
        print(f"📁 Model saved to: {model_save_path}")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        
        # Save emergency checkpoint
        if hasattr(detector, 'model') and detector.model is not None:
            emergency_path = os.path.join(model_save_path, 'emergency_checkpoint')
            os.makedirs(emergency_path, exist_ok=True)
            
            if hasattr(detector.model, 'save_pretrained'):
                detector.model.save_pretrained(emergency_path)
                if hasattr(detector, 'tokenizer'):
                    detector.tokenizer.save_pretrained(emergency_path)
                print(f"💾 Emergency checkpoint saved to: {emergency_path}")
        
        print("\n🏁 Training stopped. Progress saved.")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise
        
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    main()