"""
Visualization utilities for Toxicity Detection
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class ToxicityVisualizer:
    """Create visualizations for toxicity analysis"""
    
    @staticmethod
    def create_toxicity_gauge(score, threshold=0.65):
        """
        Create a gauge chart for toxicity score
        
        Args:
            score: Toxicity score (0-1)
            threshold: Classification threshold
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            number={'suffix': "%"},
            title={'text': "Toxicity Score"},
            delta={'reference': threshold * 100, 'position': "top"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "red" if score > threshold else "green"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'lightgreen'},
                    {'range': [30, 65], 'color': 'yellow'},
                    {'range': [65, 100], 'color': 'salmon'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'size': 12}
        )
        
        return fig
    
    @staticmethod
    def create_category_chart(categories):
        """
        Create a bar chart of toxicity categories
        
        Args:
            categories: Dictionary of category scores
            
        Returns:
            Plotly figure
        """
        df = pd.DataFrame({
            'Category': list(categories.keys()),
            'Score': list(categories.values())
        })
        
        # Color based on score
        colors = ['red' if x > 0.6 else 'orange' if x > 0.3 else 'green' 
                 for x in df['Score']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['Score'],
                y=df['Category'],
                orientation='h',
                marker_color=colors,
                text=[f"{s:.1%}" for s in df['Score']],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Toxicity Categories",
            xaxis_title="Score",
            xaxis_range=[0, 1],
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'size': 11}
        )
        
        return fig
    
    @staticmethod
    def create_comparison_chart(results_df):
        """
        Create comparison chart for batch results
        
        Args:
            results_df: DataFrame with results
            
        Returns:
            Plotly figure
        """
        # Create subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Toxicity Distribution', 'Confidence Distribution',
                          'Category Heatmap', 'Score Distribution'),
            specs=[[{'type': 'pie'}, {'type': 'histogram'}],
                   [{'type': 'heatmap'}, {'type': 'box'}]]
        )
        
        # 1. Toxicity Distribution Pie
        toxic_count = len(results_df[results_df['is_toxic'] == True])
        non_toxic_count = len(results_df[results_df['is_toxic'] == False])
        
        fig.add_trace(
            go.Pie(
                labels=['Toxic', 'Non-Toxic'],
                values=[toxic_count, non_toxic_count],
                marker=dict(colors=['red', 'green']),
                hole=0.3,
                textinfo='label+percent',
                name='Distribution'
            ),
            row=1, col=1
        )
        
        # 2. Confidence Histogram
        fig.add_trace(
            go.Histogram(
                x=results_df['confidence_score'] if 'confidence_score' in results_df.columns else results_df['confidence'],
                nbinsx=20,
                marker_color='blue',
                name='Confidence'
            ),
            row=1, col=2
        )
        
        # 3. Category Heatmap (if category columns exist)
        category_cols = [col for col in results_df.columns 
                        if col.endswith('_score')]
        
        if category_cols and len(results_df) > 0:
            # Sample up to 20 rows for heatmap
            sample_size = min(20, len(results_df))
            heatmap_data = results_df[category_cols].iloc[:sample_size].T
            
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data.values,
                    x=[f"Text {i+1}" for i in range(sample_size)],
                    y=[col.replace('_score', '') for col in category_cols],
                    colorscale='RdYlGn_r',
                    zmin=0,
                    zmax=1,
                    name='Categories'
                ),
                row=2, col=1
            )
        
        # 4. Box plots for category scores
        if category_cols:
            for col in category_cols[:3]:  # Limit to 3 categories
                fig.add_trace(
                    go.Box(
                        y=results_df[col],
                        name=col.replace('_score', ''),
                        boxmean='sd'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'size': 10},
            title_text="Toxicity Analysis Dashboard"
        )
        
        return fig
    
    @staticmethod
    def create_word_cloud_data(texts, labels):
        """
        Prepare data for word cloud visualization
        
        Args:
            texts: List of texts
            labels: List of labels
            
        Returns:
            Dictionary with word frequencies
        """
        from collections import Counter
        import re
        
        toxic_words = []
        non_toxic_words = []
        
        for text, label in zip(texts, labels):
            # Simple tokenization
            words = re.findall(r'\b\w+\b', text.lower())
            
            if label == 1:  # Toxic
                toxic_words.extend(words)
            else:
                non_toxic_words.extend(words)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'this', 'that', 'is', 'was'}
        
        toxic_words = [w for w in toxic_words if w not in stop_words and len(w) > 2]
        non_toxic_words = [w for w in non_toxic_words if w not in stop_words and len(w) > 2]
        
        return {
            'toxic': dict(Counter(toxic_words).most_common(50)),
            'non_toxic': dict(Counter(non_toxic_words).most_common(50))
        }

    @staticmethod
    def render_toxic_highlights(text, explanation):
        """
        Render text with HTML highlights for toxic words
        """
        import re
        from html import escape

        text = str(text or "")
        word_impact = explanation.get('word_impact', {})
        if not word_impact:
            return f'<div style="font-size: 1.2rem; line-height: 1.6; color: #f8fafc;">{escape(text)}</div>'
        
        # Sort words by length descending to avoid substring replacement issues
        sorted_words = [word for word in sorted(word_impact.keys(), key=len, reverse=True) if word]
        score_by_word = {
            str(word).lower(): float((meta or {}).get('score', 0.0))
            for word, meta in word_impact.items()
        }

        highlighted_parts = []
        last_index = 0
        if sorted_words:
            pattern = re.compile(
                r"\b(" + "|".join(re.escape(word) for word in sorted_words) + r")\b",
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                highlighted_parts.append(escape(text[last_index:match.start()]))
                original_word = match.group(0)
                score = score_by_word.get(original_word.lower(), 0.0)
                color = "#ef4444" if score > 0.6 else "#f59e0b" if score > 0.3 else "#10b981"
                bg_opacity = 0.3 if score > 0.6 else 0.2
                highlighted_parts.append(
                    f'<span style="background-color: {color}{int(bg_opacity*255):02x}; '
                    f'border-bottom: 2px solid {color}; padding: 2px 4px; border-radius: 4px; '
                    f'font-weight: 600; cursor: help;" title="Importance: {score:.1%}">'
                    f'{escape(original_word)}</span>'
                )
                last_index = match.end()
        highlighted_parts.append(escape(text[last_index:]))
        highlighted_text = "".join(highlighted_parts)

        return f'<div style="font-size: 1.2rem; line-height: 1.6; color: #f8fafc; background: rgba(30,41,59,0.5); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">{highlighted_text}</div>'

    @staticmethod
    def create_explanation_card(category, words, score):
        """
        Create a detailed HTML card for a specific toxicity category
        """
        color = "#ef4444" if score > 0.6 else "#f59e0b" if score > 0.3 else "#10b981"
        bg_gradient = f"linear-gradient(135deg, {color}22 0%, #1e293b 100%)"
        
        cat_name = category.replace('_', ' ').title()
        words_str = ", ".join(words) if words else "Overall Tone"
        
        html = f'''
        <div style="background: {bg_gradient}; padding: 1.2rem; border-radius: 16px; border-left: 6px solid {color}; margin-bottom: 1rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <h4 style="margin: 0; color: #f8fafc; font-size: 1.1rem;">⚠️ {cat_name}</h4>
                <span style="background: {color}; color: white; padding: 2px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 700;">{score:.1%}</span>
            </div>
            <p style="margin: 0 0 0.5rem 0; color: #cbd5e1; font-size: 0.9rem;">Triggered by: <span style="color: {color}; font-weight: 500;">{words_str}</span></p>
            <div style="width: 100%; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px;">
                <div style="width: {score*100}%; height: 100%; background: {color}; border-radius: 2px;"></div>
            </div>
        </div>
        '''
        return html


# Create singleton instance
toxicity_viz = ToxicityVisualizer()
