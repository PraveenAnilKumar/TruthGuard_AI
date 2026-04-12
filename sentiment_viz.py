"""
Sentiment Visualization Module
Create various charts for sentiment analysis results
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class SentimentVisualizer:
    """
    Create visualizations for sentiment analysis
    """
    
    @staticmethod
    def create_pie_chart(results_df: pd.DataFrame) -> go.Figure:
        """
        Create pie chart of sentiment distribution
        
        Args:
            results_df: DataFrame with results
            
        Returns:
            Plotly figure
        """
        if results_df.empty:
            return go.Figure()
        
        counts = results_df['sentiment'].value_counts()
        
        colors = {
            'POSITIVE': '#2ecc71',
            'NEGATIVE': '#e74c3c',
            'NEUTRAL': '#95a5a6'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=counts.index,
            values=counts.values,
            marker_colors=[colors.get(s, '#3498db') for s in counts.index],
            textinfo='label+percent',
            insidetextorientation='radial'
        )])
        
        fig.update_layout(
            title="Sentiment Distribution",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_bar_chart(results_df: pd.DataFrame) -> go.Figure:
        """
        Create bar chart of sentiment counts
        
        Args:
            results_df: DataFrame with results
            
        Returns:
            Plotly figure
        """
        if results_df.empty:
            return go.Figure()
        
        counts = results_df['sentiment'].value_counts().reset_index()
        counts.columns = ['sentiment', 'count']
        
        colors = {
            'POSITIVE': '#2ecc71',
            'NEGATIVE': '#e74c3c',
            'NEUTRAL': '#95a5a6'
        }
        
        fig = go.Figure(data=[go.Bar(
            x=counts['sentiment'],
            y=counts['count'],
            marker_color=[colors.get(s, '#3498db') for s in counts['sentiment']],
            text=counts['count'],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Sentiment Counts",
            xaxis_title="Sentiment",
            yaxis_title="Count",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_confidence_histogram(results_df: pd.DataFrame) -> go.Figure:
        """
        Create histogram of confidence scores
        
        Args:
            results_df: DataFrame with results
            
        Returns:
            Plotly figure
        """
        if results_df.empty:
            return go.Figure()
        
        valid_results = results_df[results_df['sentiment'] != 'ERROR']
        
        if valid_results.empty:
            return go.Figure()
        
        fig = go.Figure(data=[go.Histogram(
            x=valid_results['confidence'],
            nbinsx=20,
            marker_color='#3498db'
        )])
        
        fig.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_timeline_chart(df: pd.DataFrame, date_col: str) -> go.Figure:
        """
        Create timeline chart of sentiment over time
        
        Args:
            df: DataFrame with dates and sentiment
            date_col: Column name with dates
            
        Returns:
            Plotly figure
        """
        if df.empty or date_col not in df.columns:
            return go.Figure()
        
        # Convert to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Group by date and sentiment
        timeline = df.groupby([pd.Grouper(key=date_col, freq='D'), 'sentiment']).size().reset_index()
        timeline.columns = [date_col, 'sentiment', 'count']
        
        # Pivot for plotting
        pivot = timeline.pivot(index=date_col, columns='sentiment', values='count').fillna(0)
        
        fig = go.Figure()
        
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            if sentiment in pivot.columns:
                fig.add_trace(go.Scatter(
                    x=pivot.index,
                    y=pivot[sentiment],
                    name=sentiment,
                    mode='lines+markers',
                    line=dict(
                        color={'POSITIVE': '#2ecc71', 
                               'NEGATIVE': '#e74c3c', 
                               'NEUTRAL': '#95a5a6'}[sentiment],
                        width=2
                    )
                ))
        
        fig.update_layout(
            title="Sentiment Timeline",
            xaxis_title="Date",
            yaxis_title="Count",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_word_cloud_data(texts: List[str]) -> Dict:
        """
        Prepare data for word cloud
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with word frequencies
        """
        from collections import Counter
        import re
        
        # Combine all texts
        all_text = ' '.join(texts).lower()
        
        # Remove punctuation and split
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        
        # Count frequencies
        word_counts = Counter(words)
        
        # Get top 100 words
        top_words = word_counts.most_common(100)
        
        return dict(top_words)
    
    @staticmethod
    def create_sentiment_heatmap(results_df: pd.DataFrame, category_col: str) -> go.Figure:
        """
        Create heatmap of sentiment by category
        
        Args:
            results_df: DataFrame with results
            category_col: Column name for categories
            
        Returns:
            Plotly figure
        """
        if results_df.empty or category_col not in results_df.columns:
            return go.Figure()
        
        # Create contingency table
        contingency = pd.crosstab(
            results_df[category_col], 
            results_df['sentiment'],
            normalize='index'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=contingency.values,
            x=contingency.columns,
            y=contingency.index,
            colorscale='RdYlGn',
            text=np.round(contingency.values * 100, 1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Percentage")
        ))
        
        fig.update_layout(
            title=f"Sentiment Distribution by {category_col}",
            xaxis_title="Sentiment",
            yaxis_title=category_col,
            height=400
        )
        
        return fig