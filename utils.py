"""
Utility functions for TruthGuard AI
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Utils:
    """
    Utility class for common functions
    """
    
    @staticmethod
    def ensure_dir(directory: str):
        """
        Ensure directory exists
        
        Args:
            directory: Directory path
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict, filepath: str):
        """
        Save data to JSON file
        
        Args:
            data: Dictionary to save
            filepath: Output file path
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
    
    @staticmethod
    def load_json(filepath: str) -> Optional[Dict]:
        """
        Load data from JSON file
        
        Args:
            filepath: Input file path
            
        Returns:
            Dictionary or None if error
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return None
    
    @staticmethod
    def hash_string(text: str) -> str:
        """
        Create SHA-256 hash of string
        
        Args:
            text: Input string
            
        Returns:
            Hash string
        """
        return hashlib.sha256(text.encode()).hexdigest()
    
    @staticmethod
    def get_timestamp() -> str:
        """
        Get current timestamp
        
        Returns:
            Timestamp string
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """
        Format file size in human-readable format
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted string
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    @staticmethod
    def validate_image(filename: str) -> bool:
        """
        Validate if file is an image
        
        Args:
            filename: File name
            
        Returns:
            True if valid image
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        ext = os.path.splitext(filename)[1].lower()
        return ext in valid_extensions
    
    @staticmethod
    def validate_video(filename: str) -> bool:
        """
        Validate if file is a video
        
        Args:
            filename: File name
            
        Returns:
            True if valid video
        """
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
        ext = os.path.splitext(filename)[1].lower()
        return ext in valid_extensions
    
    @staticmethod
    def validate_text(filename: str) -> bool:
        """
        Validate if file is a text file
        
        Args:
            filename: File name
            
        Returns:
            True if valid text file
        """
        valid_extensions = {'.txt', '.csv', '.json', '.md'}
        ext = os.path.splitext(filename)[1].lower()
        return ext in valid_extensions
    
    @staticmethod
    def safe_divide(a: float, b: float, default: float = 0.0) -> float:
        """
        Safe division with zero check
        
        Args:
            a: Numerator
            b: Denominator
            default: Default value if denominator is zero
            
        Returns:
            Division result or default
        """
        return a / b if b != 0 else default
    
    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        """
        Normalize a list of scores to [0, 1]
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    @staticmethod
    def create_summary_stats(df: pd.DataFrame) -> Dict:
        """
        Create summary statistics for DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        try:
            stats['row_count'] = len(df)
            stats['column_count'] = len(df.columns)
            stats['missing_values'] = int(df.isnull().sum().sum())
            
            # Numeric columns stats
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats['numeric_columns'] = list(numeric_cols)
                # Convert to serializable format
                desc = df[numeric_cols].describe().to_dict()
                stats['numeric_stats'] = {k: {k2: float(v2) for k2, v2 in v.items()} 
                                         for k, v in desc.items()}
            
            # Categorical columns stats
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                stats['categorical_columns'] = list(cat_cols)
                for col in cat_cols[:5]:  # Limit to first 5
                    stats[f'{col}_top_values'] = df[col].value_counts().head(3).to_dict()
            
        except Exception as e:
            logger.error(f"Error creating summary stats: {e}")
        
        return stats
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Input text
            chunk_size: Maximum chunk size (in words)
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def get_file_info(filepath: str) -> Dict:
        """
        Get file information
        
        Args:
            filepath: Path to file
            
        Returns:
            Dictionary with file info
        """
        info = {}
        
        try:
            if os.path.exists(filepath):
                stat = os.stat(filepath)
                info['name'] = os.path.basename(filepath)
                info['size'] = stat.st_size
                info['size_formatted'] = Utils.format_file_size(stat.st_size)
                info['created'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
                info['modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                info['extension'] = os.path.splitext(filepath)[1].lower()
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
        
        return info


# Create singleton instance
utils = Utils()