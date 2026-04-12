"""
Translator Utilities for Multilingual Support
Handles language detection and translation back to English for analysis

OPTIMIZED: googletrans.Translator() is now lazy-initialised on first use.
The old code called Translator() at module level which triggers an HTTP
request to Google's servers and can hang indefinitely on startup.
"""

import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# langdetect and googletrans are imported lazily to avoid blocking startup
_langdetect_available = False
_googletrans_available = False

try:
    from langdetect import detect, detect_langs
    _langdetect_available = True
except ImportError:
    logger.warning("langdetect not installed. Language detection will be skipped.")

try:
    from googletrans import Translator as _GoogleTranslator
    _googletrans_available = True
except ImportError:
    logger.warning("googletrans not installed. Translation will be skipped.")


class ContentTranslator:
    """
    Handles language detection and English translation.

    The googletrans.Translator instance is created lazily on first use so
    that importing this module never triggers a network call.
    """

    def __init__(self):
        self._translator = None          # lazy: created on first translate call
        self.supported_languages = [
            'en', 'es', 'fr', 'de', 'hi', 'zh-cn', 'ar', 'pt', 'ru', 'ja'
        ]

    def _get_translator(self):
        """Return the googletrans Translator, creating it only once."""
        if self._translator is None and _googletrans_available:
            try:
                self._translator = _GoogleTranslator()
            except Exception as e:
                logger.warning(f"Could not create Translator: {e}")
        return self._translator

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.

        Returns language code (e.g. 'en', 'es'), or 'en' on failure.
        """
        if not text or len(text.strip()) < 5:
            return 'en'
        if not _langdetect_available:
            return 'en'
        try:
            return detect(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en'

    def translate_to_english(self, text: str) -> Tuple[str, str, bool]:
        """
        Translate text to English if it is not already in English.

        Returns:
            (translated_text, original_language, was_translated)
        """
        if not text or not text.strip():
            return text, 'en', False

        lang = self.detect_language(text)

        if lang == 'en':
            return text, 'en', False

        translator = self._get_translator()
        if translator is None:
            # Translation unavailable — return original text unchanged
            return text, lang, False

        try:
            logger.info(f"Translating from {lang} to English...")
            translated = translator.translate(text, dest='en')
            return translated.text, lang, True
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text, lang, False


# Module-level singleton — instant creation, no network call
content_translator = ContentTranslator()