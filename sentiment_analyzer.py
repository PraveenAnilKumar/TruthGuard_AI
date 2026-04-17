"""
Sentiment analysis with lightweight ensemble scoring.

The module keeps the same public API as before:
`analyze(text) -> (label, confidence, meta)`.
"""

import logging
import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from safe_transformers import run_isolated_text_classification
from translator_utils import content_translator

import importlib.util as _ilu

TRANSFORMERS_AVAILABLE = (
    _ilu.find_spec("transformers") is not None and
    _ilu.find_spec("torch") is not None
)
TEXTBLOB_AVAILABLE = _ilu.find_spec("textblob") is not None
del _ilu


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_VADER_SETUP_ATTEMPTED = False


def _ensure_vader_lexicon() -> None:
    global _VADER_SETUP_ATTEMPTED
    if _VADER_SETUP_ATTEMPTED:
        return

    _VADER_SETUP_ATTEMPTED = True
    try:
        import nltk

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
            return
        except LookupError:
            try:
                nltk.data.find("sentiment/vader_lexicon")
                return
            except LookupError:
                logger.info("Downloading NLTK resource: vader_lexicon")
                nltk.download("vader_lexicon", quiet=True)
    except Exception as e:
        logger.warning(f"VADER lexicon setup skipped: {e}")


class SentimentAnalyzer:
    """
    Sentiment analyzer with calibrated ensemble scoring.

    VADER is loaded when the analyzer is constructed.
    DistilBERT is loaded lazily on first use.
    """

    def __init__(self, use_ensemble: bool = True):
        self.use_ensemble = use_ensemble
        self.is_trained = False
        self.ensemble_models: List[str] = []
        self.ensemble_names: List[str] = []
        self.positive_lexicon = {
            "good": 0.8,
            "great": 1.0,
            "excellent": 1.2,
            "amazing": 1.2,
            "love": 1.3,
            "best": 1.0,
            "fantastic": 1.2,
            "wonderful": 1.1,
            "awesome": 1.1,
            "happy": 0.9,
            "glad": 0.8,
            "grateful": 0.9,
            "helpful": 0.7,
            "calm": 0.6,
            "confident": 0.8,
            "impressed": 0.8,
            "delightful": 1.0,
            "excited": 0.9,
            "positive": 0.8,
            "relieved": 0.8,
            "hopeful": 0.8,
            "optimistic": 0.9,
            "joyful": 1.0,
            "pleased": 0.8,
            "reassured": 0.8,
            "comforted": 0.7,
            "peaceful": 0.7,
            "bright": 0.6,
            "promising": 0.7,
            "reassuring": 0.8,
        }
        self.negative_lexicon = {
            "bad": 0.8,
            "terrible": 1.2,
            "awful": 1.1,
            "hate": 1.3,
            "worst": 1.2,
            "horrible": 1.2,
            "disappointing": 1.0,
            "poor": 0.8,
            "sad": 0.8,
            "angry": 1.0,
            "upset": 0.9,
            "frustrated": 0.9,
            "worried": 0.8,
            "anxious": 0.9,
            "annoying": 0.9,
            "broken": 0.9,
            "useless": 1.0,
            "negative": 0.8,
            "furious": 1.2,
            "concerned": 0.7,
            "distress": 1.1,
            "distressed": 1.1,
            "devastated": 1.2,
            "miserable": 1.0,
            "stressed": 0.9,
            "overwhelmed": 0.9,
            "frustrating": 1.0,
            "frustration": 0.9,
            "helpless": 0.9,
            "alarmed": 0.9,
            "panic": 1.1,
            "panicked": 1.1,
            "stressful": 0.9,
        }
        self.negations = {
            "not",
            "no",
            "never",
            "don't",
            "didn't",
            "isn't",
            "wasn't",
            "can't",
            "couldn't",
            "won't",
            "wouldn't",
            "shouldn't",
            "ain't",
        }
        self.intensifiers = {
            "very",
            "extremely",
            "really",
            "absolutely",
            "incredibly",
            "deeply",
            "highly",
            "so",
            "too",
            "truly",
            "remarkably",
            "especially",
        }
        self.downtoners = {
            "slightly",
            "somewhat",
            "fairly",
            "barely",
            "maybe",
            "perhaps",
            "kind",
            "sort",
            "mostly",
            "partly",
        }
        self.emotion_lexicon = {
            "happiness": {
                "happy", "glad", "love", "delighted", "excited", "cheerful",
                "thrilled", "great", "joyful", "pleased", "smiling",
            },
            "relief": {"relieved", "reassured", "reassuring", "comforted", "finally", "safe", "settled"},
            "optimism": {"hopeful", "optimistic", "promising", "encouraged", "confident", "bright"},
            "trust": {"confident", "secure", "reliable", "safe", "steady", "certain", "assured", "reassuring"},
            "surprise": {"surprised", "astonished", "shocked", "amazed", "unexpected", "wow"},
            "anger": {"angry", "furious", "hate", "annoyed", "outraged", "irritated", "mad"},
            "frustration": {"frustrated", "frustrating", "stuck", "blocked", "annoyed", "irritated", "delayed"},
            "sadness": {"sad", "disappointed", "down", "unhappy", "upset", "heartbroken", "depressed", "devastated"},
            "distress": {
                "distress", "distressed", "worried", "afraid", "anxious", "concerned",
                "nervous", "scared", "uneasy", "panic", "panicked", "overwhelmed",
                "shaken", "helpless", "stressed", "stressful",
            },
            "disgust": {"disgusted", "gross", "nasty", "revolting", "sickened"},
        }

        try:
            _ensure_vader_lexicon()
            from nltk.sentiment import SentimentIntensityAnalyzer

            self.vader = SentimentIntensityAnalyzer()
            self.ensemble_models.append("vader")
            self.ensemble_names.append("VADER")
            logger.info("VADER initialised")
        except Exception as e:
            self.vader = None
            logger.warning(f"VADER not available: {e}")

        self._distilbert = None
        self._distilbert_loaded = False
        self._distilbert_ok = False
        self._distilbert_model_ref = "distilbert-base-uncased-finetuned-sst-2-english"

        logger.info(
            f"SentimentAnalyzer ready with {len(self.ensemble_models)} instant model(s). "
            "DistilBERT will be loaded on first use."
        )

    def _ensure_distilbert(self) -> None:
        if self._distilbert_loaded:
            return
        self._distilbert_loaded = True

        if not (TRANSFORMERS_AVAILABLE and self.use_ensemble):
            return

        self._distilbert = {"model": self._distilbert_model_ref, "tokenizer": self._distilbert_model_ref}
        self._distilbert_ok = True
        if "distilbert" not in self.ensemble_models:
            self.ensemble_models.append("distilbert")
            self.ensemble_names.append("DistilBERT")
        logger.info("DistilBERT enabled in isolated inference mode.")

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())
        return text

    def analyze(self, text: str) -> Tuple[str, float, Dict]:
        if not text or not text.strip():
            return ("NEUTRAL", 0.5, {})

        translated_text, original_lang, was_translated = content_translator.translate_to_english(text)
        if isinstance(translated_text, (list, tuple, set)):
            translated_text = " ".join(str(part) for part in translated_text if part is not None)
        elif translated_text is None:
            translated_text = ""
        elif not isinstance(translated_text, str):
            translated_text = str(translated_text)
        meta = {
            "original_language": original_lang,
            "was_translated": was_translated,
            "processed_text": translated_text if was_translated else None,
        }

        self._ensure_distilbert()
        tone_features = self._extract_tone_features(translated_text)

        if self.use_ensemble:
            model_results = self._collect_model_outputs(translated_text)
            label, conf, sentiment_score, agreement = self._combine_sentiment_results(
                model_results, tone_features
            )
        else:
            detail = self._analyze_single_detailed(translated_text, tone_features)
            model_results = [detail]
            label = detail["label"]
            conf = float(np.clip(detail["confidence"] + (tone_features["intensity"] * 0.06), 0.5, 0.99))
            sentiment_score = float(detail["signed_score"])
            agreement = 1.0

        tone_features = dict(tone_features)
        tone_features["dominant_emotion"] = self._resolve_dominant_emotion(tone_features, label)

        explanation = self._build_explanation(
            label=label,
            confidence=conf,
            sentiment_score=sentiment_score,
            agreement=agreement,
            tone_features=tone_features,
            model_results=model_results,
        )
        meta.update(
            {
                "sentiment_score": sentiment_score,
                "intensity": tone_features["intensity"],
                "dominant_emotion": tone_features["dominant_emotion"],
                "emotion_scores": tone_features["emotion_scores"],
                "emotion_rankings": explanation["top_emotions"],
                "tone_flags": tone_features["flags"],
                "agreement": agreement,
                "explanation": explanation,
                "model_breakdown": [
                    {
                        "source": result["source"],
                        "label": result["label"],
                        "confidence": result["confidence"],
                        "signed_score": result["signed_score"],
                    }
                    for result in model_results
                ],
            }
        )
        return (label, conf, meta)

    def _analyze_ensemble(self, text: str) -> Tuple[str, float]:
        tone_features = self._extract_tone_features(text)
        results = self._collect_model_outputs(text)
        label, conf, _score, _agreement = self._combine_sentiment_results(results, tone_features)
        return (label, conf)

    def _analyze_single(self, text: str) -> Tuple[str, float]:
        detail = self._analyze_single_detailed(text, self._extract_tone_features(text))
        return (detail["label"], detail["confidence"])

    def _label_from_signed_score(self, signed_score: float, neutral_band: float = 0.18) -> str:
        if signed_score >= neutral_band:
            return "POSITIVE"
        if signed_score <= -neutral_band:
            return "NEGATIVE"
        return "NEUTRAL"

    def _emotion_alignment(self, emotion: str) -> int:
        if emotion in {"happiness", "relief", "optimism", "trust"}:
            return 1
        if emotion in {"anger", "frustration", "sadness", "distress", "disgust"}:
            return -1
        return 0

    def _select_dominant_emotion(self, emotion_scores: Counter, signed_score: float) -> str:
        if not emotion_scores:
            return "balanced"

        def sort_key(item):
            emotion, score = item
            alignment = self._emotion_alignment(emotion)
            if signed_score > 0.1:
                alignment_bonus = 1 if alignment > 0 else 0
            elif signed_score < -0.1:
                alignment_bonus = 1 if alignment < 0 else 0
            else:
                alignment_bonus = 1 if alignment == 0 else 0
            return (int(score), alignment_bonus, abs(alignment))

        return max(emotion_scores.items(), key=sort_key)[0]

    def _resolve_dominant_emotion(self, tone_features: Dict, label: str) -> str:
        emotion_scores = Counter(tone_features.get("emotion_scores") or {})
        if not emotion_scores:
            return str(tone_features.get("dominant_emotion", "balanced"))

        def sort_key(item):
            emotion, score = item
            alignment = self._emotion_alignment(emotion)
            if label == "POSITIVE":
                alignment_bonus = 1 if alignment > 0 else 0
            elif label == "NEGATIVE":
                alignment_bonus = 1 if alignment < 0 else 0
            else:
                alignment_bonus = 1 if alignment == 0 else 0
            return (int(score), alignment_bonus, abs(alignment))

        return max(emotion_scores.items(), key=sort_key)[0]

    def _top_emotions(self, tone_features: Dict, limit: int = 3) -> List[Dict]:
        emotion_scores = tone_features.get("emotion_scores") or {}
        matched_emotion_terms = tone_features.get("matched_emotion_terms") or {}
        ranked = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)

        if not ranked:
            dominant_emotion = tone_features.get("dominant_emotion")
            if dominant_emotion and dominant_emotion != "balanced":
                ranked = [(dominant_emotion, 1)]

        top_emotions: List[Dict] = []
        for emotion, score in ranked[:limit]:
            top_emotions.append(
                {
                    "emotion": emotion,
                    "score": int(score),
                    "terms": list((matched_emotion_terms.get(emotion) or [])[:3]),
                }
            )
        return top_emotions

    def _build_explanation(
        self,
        label: str,
        confidence: float,
        sentiment_score: float,
        agreement: float,
        tone_features: Dict,
        model_results: List[Dict],
    ) -> Dict:
        positive_terms = list((tone_features.get("matched_positive_terms") or [])[:4])
        negative_terms = list((tone_features.get("matched_negative_terms") or [])[:4])
        matched_emotion_terms = tone_features.get("matched_emotion_terms") or {}
        dominant_emotion = str(tone_features.get("dominant_emotion", "balanced")).replace("_", " ")
        top_emotions = self._top_emotions(tone_features)
        evidence: List[str] = []

        if label == "POSITIVE":
            if positive_terms:
                evidence.append(f"Positive language outweighed negative cues: {', '.join(positive_terms)}.")
            else:
                evidence.append(f"The combined polarity score stayed above the positive threshold ({sentiment_score:+.2f}).")
        elif label == "NEGATIVE":
            if negative_terms:
                evidence.append(f"Negative or critical wording dominated the message: {', '.join(negative_terms)}.")
            else:
                evidence.append(f"The combined polarity score stayed below the negative threshold ({sentiment_score:+.2f}).")
        else:
            if positive_terms and negative_terms:
                evidence.append(
                    f"Positive and negative cues were both present ({', '.join(positive_terms[:2])} vs {', '.join(negative_terms[:2])}), which balanced the final score."
                )
            else:
                evidence.append("The wording stayed close to balanced, so the score remained in the neutral band.")

        dominant_terms = list((matched_emotion_terms.get(tone_features.get("dominant_emotion")) or [])[:3])
        if dominant_terms:
            evidence.append(
                f"Emotion cues pointed to {dominant_emotion} through words like {', '.join(dominant_terms)}."
            )
        elif dominant_emotion != "balanced":
            evidence.append(f"The strongest emotional pattern aligned with {dominant_emotion}.")

        flags = tone_features.get("flags") or []
        emphasis_bits: List[str] = []
        if "intensifier" in flags:
            emphasis_bits.append("intensifiers")
        if "strong_emphasis" in flags:
            emphasis_bits.append("repeated exclamation")
        if "all_caps_emphasis" in flags:
            emphasis_bits.append("all-caps emphasis")
        if emphasis_bits:
            evidence.append(f"Intensity increased because of {', '.join(emphasis_bits)}.")
        if "negation" in flags:
            evidence.append("Negations were detected, which can flip the meaning of nearby words.")
        if "mixed_polarity" in flags:
            evidence.append("The text contains mixed polarity, so the model weighed conflicting cues together.")

        if model_results:
            label_counts = Counter(result["label"] for result in model_results)
            support_count = label_counts.get(label, 0)
            supporting_sources = [
                str(result["source"]).upper()
                for result in model_results
                if result.get("label") == label
            ]
            if support_count:
                evidence.append(
                    f"{support_count} of {len(model_results)} analyzers supported the {label.lower()} conclusion ({', '.join(supporting_sources[:3])})."
                )

        emotion_suffix = (
            f" Strongest emotion signal: {dominant_emotion}."
            if dominant_emotion != "balanced"
            else ""
        )

        if label == "POSITIVE":
            summary = (
                f"Detected a positive tone with {confidence:.0%} confidence because supportive wording outweighed "
                f"negative signals.{emotion_suffix}"
            )
        elif label == "NEGATIVE":
            summary = (
                f"Detected a negative tone with {confidence:.0%} confidence because critical wording outweighed "
                f"positive signals.{emotion_suffix}"
            )
        else:
            summary = (
                f"Detected a neutral tone with {confidence:.0%} confidence because the message stayed balanced "
                f"or only showed mild emotional cues.{emotion_suffix}"
            )

        return {
            "summary": summary,
            "signal_evidence": evidence,
            "top_emotions": top_emotions,
            "matched_positive_terms": positive_terms,
            "matched_negative_terms": negative_terms,
            "matched_emotion_terms": matched_emotion_terms,
            "agreement": float(agreement),
        }

    def _extract_tone_features(self, text: str) -> Dict:
        raw_text = text or ""
        tokens = re.findall(r"[a-z']+", raw_text.lower())
        lexical_total = 0.0
        positive_hits = 0
        negative_hits = 0
        flags: List[str] = []
        emotion_scores = Counter()
        positive_terms: List[str] = []
        negative_terms: List[str] = []
        matched_emotion_terms: Dict[str, List[str]] = {}
        negation_scope = 0

        for idx, token in enumerate(tokens):
            if token in self.negations:
                negation_scope = 2
                if "negation" not in flags:
                    flags.append("negation")
                continue

            modifier = 1.0
            if idx > 0 and tokens[idx - 1] in self.intensifiers:
                modifier *= 1.25
                if "intensifier" not in flags:
                    flags.append("intensifier")
            if idx > 0 and tokens[idx - 1] in self.downtoners:
                modifier *= 0.8
                if "downtoner" not in flags:
                    flags.append("downtoner")
            if negation_scope > 0:
                modifier *= -0.85
                negation_scope -= 1

            if token in self.positive_lexicon:
                lexical_total += self.positive_lexicon[token] * modifier
                positive_hits += 1
                if token not in positive_terms:
                    positive_terms.append(token)
            elif token in self.negative_lexicon:
                lexical_total -= self.negative_lexicon[token] * modifier
                negative_hits += 1
                if token not in negative_terms:
                    negative_terms.append(token)

            for emotion, words in self.emotion_lexicon.items():
                if token in words:
                    emotion_scores[emotion] += 1
                    matched_emotion_terms.setdefault(emotion, [])
                    if token not in matched_emotion_terms[emotion]:
                        matched_emotion_terms[emotion].append(token)

        signed_score = float(np.tanh(lexical_total / 3.0)) if lexical_total else 0.0
        exclamation_count = raw_text.count("!")
        question_count = raw_text.count("?")
        caps_words = len(re.findall(r"\b[A-Z]{3,}\b", raw_text))
        intensity = float(
            np.clip(
                abs(signed_score) * 0.45
                + min(exclamation_count, 4) * 0.09
                + min(question_count, 3) * 0.04
                + min(caps_words, 3) * 0.10,
                0.0,
                1.0,
            )
        )

        if positive_hits and negative_hits:
            flags.append("mixed_polarity")
        if exclamation_count >= 2:
            flags.append("strong_emphasis")
        if caps_words:
            flags.append("all_caps_emphasis")

        if emotion_scores:
            dominant_emotion = self._select_dominant_emotion(emotion_scores, signed_score)
        elif signed_score > 0.45:
            dominant_emotion = "happiness"
        elif signed_score > 0.18:
            dominant_emotion = "optimism"
        elif signed_score < -0.5:
            dominant_emotion = "distress"
        elif signed_score < -0.18:
            dominant_emotion = "frustration" if "strong_emphasis" in flags else "sadness"
        else:
            dominant_emotion = "balanced"

        return {
            "lexical_score": signed_score,
            "intensity": intensity,
            "dominant_emotion": dominant_emotion,
            "emotion_scores": dict(emotion_scores),
            "matched_positive_terms": positive_terms,
            "matched_negative_terms": negative_terms,
            "matched_emotion_terms": matched_emotion_terms,
            "flags": flags,
            "positive_hits": positive_hits,
            "negative_hits": negative_hits,
        }

    def _collect_model_outputs(self, text: str) -> List[Dict]:
        results: List[Dict] = []

        if "vader" in self.ensemble_models and self.vader:
            result = self._analyze_vader_detailed(text)
            if result:
                results.append(result)

        if self._distilbert_ok and self._distilbert:
            result = self._analyze_distilbert_detailed(text)
            if result:
                results.append(result)

        if TEXTBLOB_AVAILABLE:
            result = self._analyze_textblob_detailed(text)
            if result:
                results.append(result)

        results.append(self._analyze_heuristic_detailed(text))
        return results

    def _combine_sentiment_results(self, results: List[Dict], tone_features: Dict) -> Tuple[str, float, float, float]:
        if not results:
            detail = self._analyze_heuristic_detailed("", tone_features)
            return detail["label"], detail["confidence"], detail["signed_score"], 0.6

        weights = {
            "distilbert": 1.15,
            "vader": 1.0,
            "textblob": 0.75,
            "heuristic": 0.55,
        }
        signed_scores = np.array([result["signed_score"] for result in results], dtype=np.float32)
        model_weights = np.array([weights.get(result["source"], 0.6) for result in results], dtype=np.float32)

        weighted_score = float(np.average(signed_scores, weights=model_weights))
        combined_score = float(np.clip((weighted_score * 0.82) + (tone_features["lexical_score"] * 0.18), -1.0, 1.0))

        agreement = 1.0
        if len(signed_scores) > 1:
            agreement = max(0.0, 1.0 - min(float(np.std(signed_scores)) / 0.85, 1.0))
        if "mixed_polarity" in tone_features["flags"]:
            agreement *= 0.9

        label = self._label_from_signed_score(combined_score)
        model_confidence = float(np.mean([result["confidence"] for result in results]))
        magnitude = abs(combined_score)
        if label == "NEUTRAL":
            confidence = (
                0.50
                + agreement * 0.18
                + (1.0 - min(magnitude / 0.22, 1.0)) * 0.10
                + model_confidence * 0.10
                + tone_features["intensity"] * 0.04
            )
        else:
            confidence = (
                0.46
                + magnitude * 0.24
                + agreement * 0.13
                + model_confidence * 0.12
                + tone_features["intensity"] * 0.08
            )
        confidence = float(np.clip(confidence, 0.5, 0.99))
        return label, confidence, combined_score, float(agreement)

    def _analyze_single_detailed(self, text: str, tone_features: Optional[Dict] = None) -> Dict:
        if self._distilbert_ok and self._distilbert:
            result = self._analyze_distilbert_detailed(text)
            if result:
                return result

        if self.vader:
            result = self._analyze_vader_detailed(text)
            if result:
                return result

        if TEXTBLOB_AVAILABLE:
            result = self._analyze_textblob_detailed(text)
            if result:
                return result

        return self._analyze_heuristic_detailed(text, tone_features)

    def _analyze_vader(self, text: str) -> Optional[Tuple[str, float]]:
        result = self._analyze_vader_detailed(text)
        if result is None:
            return None
        return (result["label"], result["confidence"])

    def _analyze_vader_detailed(self, text: str) -> Optional[Dict]:
        try:
            scores = self.vader.polarity_scores(text)
            compound = float(scores["compound"])
            label = self._label_from_signed_score(compound, neutral_band=0.10)
            confidence = float(np.clip(0.5 + (abs(compound) * 0.42), 0.5, 0.95))
            return {
                "source": "vader",
                "label": label,
                "confidence": confidence,
                "signed_score": compound,
                "raw": scores,
            }
        except Exception:
            return None

    def _analyze_distilbert(self, text: str) -> Optional[Tuple[str, float]]:
        result = self._analyze_distilbert_detailed(text)
        if result is None:
            return None
        return (result["label"], result["confidence"])

    def _analyze_distilbert_detailed(self, text: str) -> Optional[Dict]:
        try:
            result = run_isolated_text_classification(
                self._distilbert["model"],
                text[:512],
                tokenizer_ref=self._distilbert.get("tokenizer"),
                local_files_only=False,
            )
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "Unknown isolated inference failure"))
            result = result["result"]
            raw_label = result["label"].upper()
            score = float(result["score"])
            polarity = max((score - 0.5) * 2.0, 0.0)
            signed_score = polarity if "POS" in raw_label else -polarity
            label = self._label_from_signed_score(signed_score)
            confidence = float(np.clip(0.52 + polarity * 0.43, 0.5, 0.99))
            return {
                "source": "distilbert",
                "label": label,
                "confidence": confidence,
                "signed_score": float(signed_score),
                "raw": result,
            }
        except Exception as exc:
            logger.warning(f"DistilBERT sentiment inference unavailable: {exc}")
            self._distilbert_ok = False
            if "distilbert" in self.ensemble_models:
                self.ensemble_models.remove("distilbert")
            if "DistilBERT" in self.ensemble_names:
                self.ensemble_names.remove("DistilBERT")
            return None

    def _analyze_textblob(self, text: str) -> Optional[Tuple[str, float]]:
        result = self._analyze_textblob_detailed(text)
        if result is None:
            return None
        return (result["label"], result["confidence"])

    def _analyze_textblob_detailed(self, text: str) -> Optional[Dict]:
        try:
            from textblob import TextBlob

            blob = TextBlob(text)
            polarity = float(blob.sentiment.polarity)
            label = self._label_from_signed_score(polarity, neutral_band=0.12)
            confidence = float(np.clip(0.5 + abs(polarity) * 0.38, 0.5, 0.92))
            return {
                "source": "textblob",
                "label": label,
                "confidence": confidence,
                "signed_score": polarity,
            }
        except Exception:
            return None

    def _analyze_heuristic(self, text: str) -> Tuple[str, float]:
        result = self._analyze_heuristic_detailed(text)
        return (result["label"], result["confidence"])

    def _analyze_heuristic_detailed(self, text: str, tone_features: Optional[Dict] = None) -> Dict:
        features = tone_features or self._extract_tone_features(text)
        signed_score = float(features["lexical_score"])
        label = self._label_from_signed_score(signed_score)
        confidence = float(
            np.clip(
                0.5
                + abs(signed_score) * 0.24
                + features["intensity"] * 0.12
                + (0.05 if features["positive_hits"] or features["negative_hits"] else 0.0),
                0.5,
                0.88,
            )
        )
        return {
            "source": "heuristic",
            "label": label,
            "confidence": confidence,
            "signed_score": signed_score,
        }

    def create_gauge(self, confidence: float, label: str):
        import plotly.graph_objects as go

        colors = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"}
        color = colors.get(label, "blue")
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={"text": f"Confidence - {label}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 33], "color": "lightgray"},
                        {"range": [33, 66], "color": "gray"},
                        {"range": [66, 100], "color": "darkgray"},
                    ],
                },
            )
        )
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        return fig


class _LazySentimentAnalyzer:
    """Proxy that defers SentimentAnalyzer construction until first use."""
    _instance = None

    def _get(self):
        if self._instance is None:
            object.__setattr__(self, "_instance", SentimentAnalyzer())
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def __setattr__(self, name, value):
        if name == "_instance":
            object.__setattr__(self, name, value)
        else:
            setattr(self._get(), name, value)


sentiment_analyzer = _LazySentimentAnalyzer()
