#!/usr/bin/env python
import argparse
import os
from sentiment_analyzer import SentimentAnalyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--text-col', default='text')
    parser.add_argument('--label-col', default='sentiment')
    parser.add_argument('--output-dir', default='models/sentiment')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--model-name', default='distilbert-base-uncased-finetuned-sst-2-english')
    args = parser.parse_args()

    analyzer = SentimentAnalyzer(model_name=args.model_name)
    analyzer.fine_tune(args.dataset, args.text_col, args.label_col,
                       args.output_dir, args.epochs, args.batch_size)

if __name__ == '__main__':
    main()