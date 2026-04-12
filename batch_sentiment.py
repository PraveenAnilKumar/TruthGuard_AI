"""
Batch Sentiment Processing
Process multiple texts efficiently
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_sentiment_analyzer = None

def _get_sentiment_analyzer():
    """Lazy load sentiment analyzer"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            from sentiment_analyzer import sentiment_analyzer
            _sentiment_analyzer = sentiment_analyzer
        except ImportError:
            logger.error("Could not import sentiment_analyzer")
            raise
    return _sentiment_analyzer

class BatchSentimentProcessor:
    """
    Process multiple texts for sentiment analysis
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize batch processor
        
        Args:
            batch_size: Number of texts to process at once
        """
        self.batch_size = batch_size
    
    def process_file(self, filepath: str, text_column: str = None) -> pd.DataFrame:
        """
        Process file with multiple texts
        
        Args:
            filepath: Path to file (CSV or TXT)
            text_column: Column name for CSV files
            
        Returns:
            DataFrame with results
        """
        # Load file
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            if text_column:
                texts = df[text_column].tolist()
            else:
                # Assume first column is text
                texts = df.iloc[:, 0].tolist()
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        # Process in batches
        results = self.process_texts(texts)
        
        # Add original text if from CSV
        if filepath.endswith('.csv') and text_column:
            results['original_text'] = texts
        
        return results
    
    def process_texts(self, texts: List[str]) -> pd.DataFrame:
        """
        Process multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with results
        """
        sentiment_analyzer = _get_sentiment_analyzer()
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            for text in batch:
                try:
                    label, conf, meta = sentiment_analyzer.analyze(text)
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'sentiment': label,
                        'confidence': conf,
                        'language': meta.get('original_language', 'en'),
                        'length': len(text)
                    })
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
                    results.append({
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'sentiment': 'ERROR',
                        'confidence': 0.0,
                        'length': len(text),
                        'error': str(e)
                    })
            
            if (i // self.batch_size) % 10 == 0:
                logger.info(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} texts")
        
        return pd.DataFrame(results)
    
    def get_statistics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate statistics from results
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        if results_df.empty:
            return stats
        
        # Counts
        stats['total'] = len(results_df)
        stats['positive'] = len(results_df[results_df['sentiment'] == 'POSITIVE'])
        stats['negative'] = len(results_df[results_df['sentiment'] == 'NEGATIVE'])
        stats['neutral'] = len(results_df[results_df['sentiment'] == 'NEUTRAL'])
        stats['errors'] = len(results_df[results_df['sentiment'] == 'ERROR'])
        
        # Percentages
        if stats['total'] > 0:
            stats['positive_pct'] = stats['positive'] / stats['total'] * 100
            stats['negative_pct'] = stats['negative'] / stats['total'] * 100
            stats['neutral_pct'] = stats['neutral'] / stats['total'] * 100
        
        # Confidence
        valid_results = results_df[results_df['sentiment'] != 'ERROR']
        if not valid_results.empty:
            stats['avg_confidence'] = valid_results['confidence'].mean()
            stats['min_confidence'] = valid_results['confidence'].min()
            stats['max_confidence'] = valid_results['confidence'].max()
        else:
            stats['avg_confidence'] = 0
        
        # Text length statistics
        if 'length' in results_df.columns:
            stats['avg_length'] = results_df['length'].mean()
            stats['min_length'] = results_df['length'].min()
            stats['max_length'] = results_df['length'].max()
        
        return stats