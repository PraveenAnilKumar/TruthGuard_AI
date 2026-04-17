"""
Aspect-Based Sentiment Analysis
Analyzes sentiment for specific aspects in text
"""

import re
import logging
from typing import Dict, List, Tuple
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_sentiment_analyzer = None

def _get_sentiment_analyzer():
    """Lazy load sentiment analyzer to avoid circular imports"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            from sentiment_analyzer import sentiment_analyzer
            _sentiment_analyzer = sentiment_analyzer
        except ImportError:
            logger.error("Could not import sentiment_analyzer")
            raise
    return _sentiment_analyzer

class AspectSentimentAnalyzer:
    """
    Analyze sentiment for specific aspects like product quality, price, service, etc.
    """
    
    def __init__(self):
        """Initialize aspect analyzer"""
        self.aspect_keywords = {
            'quality': ['quality', 'build', 'material', 'durable', 'sturdy', 
                       'well-made', 'craftsmanship', 'construction'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 
                     'value', 'worth', 'overpriced', 'budget'],
            'service': ['service', 'support', 'customer service', 'help', 
                       'assistance', 'response', 'staff', 'representative'],
            'delivery': ['delivery', 'shipping', 'arrived', 'package', 
                        'shipped', 'shipping time', 'delivery time'],
            'usability': ['easy', 'difficult', 'interface', 'user-friendly', 
                         'intuitive', 'complicated', 'setup', 'installation'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'efficient',
                           'battery', 'power', 'response time'],
            'design': ['design', 'look', 'appearance', 'style', 'beautiful',
                      'ugly', 'aesthetic', 'color', 'finish']
        }
        
    def analyze_aspects(self, text: str) -> Dict[str, Dict]:
        """
        Analyze sentiment for each aspect
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with aspect sentiments
        """
        text_lower = text.lower()
        aspects_found = {}
        sentiment_analyzer = _get_sentiment_analyzer()
        
        for aspect, keywords in self.aspect_keywords.items():
            # Check if aspect is mentioned
            mentioned = any(keyword in text_lower for keyword in keywords)
            
            if mentioned:
                # Extract sentences containing aspect keywords
                sentences = self._extract_aspect_sentences(text, keywords)
                
                if sentences:
                    # Analyze sentiment of these sentences
                    aspect_sentiment = self._analyze_sentences(sentences, sentiment_analyzer)
                    aspects_found[aspect] = aspect_sentiment
                else:
                    # Analyze full text as fallback
                    label, conf, meta = sentiment_analyzer.analyze(text)
                    aspects_found[aspect] = {
                        'label': label,
                        'confidence': conf,
                        'dominant_emotion': meta.get('dominant_emotion', 'balanced'),
                        'emotion_scores': meta.get('emotion_scores', {}),
                        'explanation': {
                            'summary': (
                                f"{aspect.title()} was classified as {label.lower()} from the overall message context."
                            ),
                            'evidence': list((meta.get('explanation') or {}).get('signal_evidence', [])),
                        },
                        'sentences': [],
                    }
        
        return aspects_found
    
    def _extract_aspect_sentences(self, text: str, keywords: List[str]) -> List[str]:
        """
        Extract sentences containing aspect keywords
        
        Args:
            text: Input text
            keywords: Aspect keywords
            
        Returns:
            List of relevant sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        relevant = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant.append(sentence.strip())
        
        return relevant
    
    def _analyze_sentences(self, sentences: List[str], sentiment_analyzer) -> Dict:
        """
        Analyze sentiment of sentences
        
        Args:
            sentences: List of sentences
            sentiment_analyzer: SentimentAnalyzer instance
            
        Returns:
            Aggregated sentiment
        """
        if not sentences:
            return {
                'label': 'NEUTRAL',
                'confidence': 0.5,
                'dominant_emotion': 'balanced',
                'emotion_scores': {},
                'explanation': {
                    'summary': 'No aspect-specific evidence was found, so the aspect result stayed neutral.',
                    'evidence': [],
                },
                'sentences': [],
            }

        results = []
        emotion_scores = Counter()
        for sentence in sentences:
            label, conf, meta = sentiment_analyzer.analyze(sentence)
            sentence_explanation = meta.get('explanation') or {}
            dominant_emotion = meta.get('dominant_emotion', 'balanced')
            for emotion, score in (meta.get('emotion_scores') or {}).items():
                emotion_scores[emotion] += score
            results.append({
                'sentence': sentence,
                'label': label,
                'confidence': conf,
                'emotion': dominant_emotion,
                'explanation': sentence_explanation.get('summary', ''),
                'evidence': list(sentence_explanation.get('signal_evidence', [])),
            })

        # Aggregate
        labels = [r['label'] for r in results]
        label_counts = Counter(labels)
        most_common = label_counts.most_common(1)[0][0]
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        dominant_emotion = 'balanced'
        if emotion_scores:
            if hasattr(sentiment_analyzer, "_resolve_dominant_emotion"):
                dominant_emotion = sentiment_analyzer._resolve_dominant_emotion(
                    {
                        'emotion_scores': dict(emotion_scores),
                        'dominant_emotion': emotion_scores.most_common(1)[0][0],
                    },
                    most_common,
                )
            else:
                dominant_emotion = emotion_scores.most_common(1)[0][0]
        strongest_sentences = sorted(results, key=lambda item: item['confidence'], reverse=True)[:2]
        sentence_refs = [item['sentence'] for item in strongest_sentences]

        return {
            'label': most_common,
            'confidence': avg_confidence,
            'dominant_emotion': dominant_emotion,
            'emotion_scores': dict(emotion_scores),
            'explanation': {
                'summary': (
                    f"{label_counts.get(most_common, 0)} of {len(results)} aspect-specific sentence(s) leaned "
                    f"{most_common.lower()}, with {dominant_emotion.replace('_', ' ')} as the strongest emotion."
                ),
                'evidence': [
                    "Strongest evidence came from: " + " | ".join(sentence_refs)
                ] if sentence_refs else [],
            },
            'sentences': results,
        }
