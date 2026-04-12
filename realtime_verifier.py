"""
Real-Time News Verification Module
Uses deeper multi-query news retrieval and source credibility scoring to
verify claims against live reporting.

OPTIMIZED: Module-level singleton uses a lazy proxy so sklearn/numpy imports
don't block startup. The real __init__ only fires on first use after login.
"""

import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional
from urllib.parse import quote, urlparse

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from translator_utils import content_translator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional dependencies with graceful failure
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 not available. Real-time RSS search will be disabled.")

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    logger.warning("newspaper3k not available. Article content extraction will be limited.")


class RealtimeNewsVerifier:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TruthGuardAI/1.0 (+https://truthguard.local)",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.max_results_per_query = 8
        self.max_sources_returned = 10
        self.max_article_fetches = 5
        self.article_cache: Dict[str, str] = {}
        self.reputable_sources = {
            'reuters.com': 1.0,
            'apnews.com': 0.99,
            'bbc.com': 0.96,
            'bloomberg.com': 0.95,
            'wsj.com': 0.94,
            'nytimes.com': 0.93,
            'theguardian.com': 0.9,
            'npr.org': 0.9,
            'ft.com': 0.9,
            'abcnews.go.com': 0.88,
            'cbsnews.com': 0.88,
            'nbcnews.com': 0.88,
            'cnn.com': 0.86,
            'aljazeera.com': 0.85,
        }

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r'\s+', ' ', (text or '').strip())

    def _get_domain(self, url: str) -> str:
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return ""

    def _normalize_url(self, url: str) -> str:
        if not url:
            return ""
        url = url.strip()
        if url.startswith("https://news.google.com/rss/articles/"):
            return url
        parsed = urlparse(url)
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return clean.rstrip("/")

    def _parse_datetime(self, value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            if value.endswith("Z"):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            dt = parsedate_to_datetime(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            try:
                dt = datetime.fromisoformat(value)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                return None

    def _freshness_score(self, published: str) -> float:
        dt = self._parse_datetime(published)
        if not dt:
            return 0.45
        age_hours = max((datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600, 0)
        if age_hours <= 12:
            return 1.0
        if age_hours <= 24:
            return 0.92
        if age_hours <= 72:
            return 0.8
        if age_hours <= 168:
            return 0.65
        return 0.45

    def _domain_credibility(self, url: str, source_name: str = "") -> float:
        domain = self._get_domain(url)
        source_text = (source_name or "").lower()
        for reputable_domain, score in self.reputable_sources.items():
            if domain.endswith(reputable_domain) or reputable_domain in source_text:
                return score
        if any(token in domain for token in ("gov", "edu")):
            return 0.9
        if any(token in source_text for token in ("official", "government", "university", "institute")):
            return 0.82
        return 0.5

    def _extract_query(self, text: str) -> str:
        """Extract a compact keyword query from text using NLP with a regex fallback."""
        cleaned = re.sub(r'[^\w\s\'"-]', ' ', text)
        try:
            import nltk
            from nltk.corpus import stopwords

            for resource, path in [
                ('corpora/stopwords', 'stopwords'),
                ('tokenizers/punkt', 'punkt'),
                ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ]:
                try:
                    nltk.data.find(resource)
                except LookupError:
                    nltk.download(path, quiet=True)

            tokens = nltk.word_tokenize(cleaned)
            stop_words = set(stopwords.words('english'))
            try:
                tagged = nltk.pos_tag(tokens)
                keywords = [
                    word for word, tag in tagged
                    if (tag.startswith('NN') or tag.startswith('JJ') or tag.startswith('VB'))
                    and word.lower() not in stop_words
                    and len(word) > 2
                ]
            except Exception as exc:
                logger.warning(f"POS tagging failed: {exc}")
                keywords = [w for w in tokens if w.lower() not in stop_words and len(w) > 3]
            query = " ".join(keywords[:8])
            return self._normalize_whitespace(query if query else cleaned[:120])
        except Exception as exc:
            logger.error(f"Query extraction error: {exc}")
            words = [w for w in cleaned.split() if len(w) > 3]
            return self._normalize_whitespace(" ".join(words[:8]) or cleaned[:120])

    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        for phrase in re.findall(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,2})\b', text):
            phrase = self._normalize_whitespace(phrase)
            if len(phrase) > 2 and phrase.lower() not in {"The", "This", "That"}:
                entities.append(phrase)
        seen = set()
        unique = []
        for entity in entities:
            key = entity.lower()
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        return unique[:4]

    def _extract_quoted_phrase(self, text: str) -> Optional[str]:
        for quoted in re.findall(r'"([^"]{8,80})"|\'([^\']{8,80})\'', text):
            phrase = quoted[0] or quoted[1]
            phrase = self._normalize_whitespace(phrase)
            if phrase:
                return phrase
        tokens = text.split()
        if len(tokens) >= 6:
            return self._normalize_whitespace(" ".join(tokens[:8]))
        return None

    def _build_query_variants(self, text: str) -> List[str]:
        base = self._extract_query(text)
        variants = [base]
        entities = self._extract_entities(text)
        if entities:
            variants.append(self._normalize_whitespace(" ".join(entities[:3])))
            variants.append(self._normalize_whitespace(f"{base} {' '.join(entities[:2])}"))
        quoted = self._extract_quoted_phrase(text)
        if quoted:
            variants.append(f'"{quoted}"')
            if entities:
                variants.append(self._normalize_whitespace(f'"{quoted}" {" ".join(entities[:2])}'))
        shortened = " ".join(base.split()[:4]).strip()
        if shortened and shortened != base:
            variants.append(shortened)
        seen = set()
        deduped = []
        for query in variants:
            normalized = self._normalize_whitespace(query)
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                deduped.append(normalized)
        return deduped[:5]

    def _search_news_api(self, query: str) -> List[Dict]:
        """Search news using NewsAPI.org (requires API key)."""
        api_key = os.getenv("NEWS_API_KEY") or self.api_key
        if not api_key:
            return []
        try:
            logger.info(f"Searching NewsAPI.org: {query}")
            url = "https://newsapi.org/v2/everything"
            response = self.session.get(url, params={
                "q": query,
                "sortBy": "relevancy",
                "pageSize": self.max_results_per_query,
                "language": "en",
                "apiKey": api_key,
            }, timeout=10)
            data = response.json()
            if data.get('status') == 'ok':
                return [
                    {
                        'title': a.get('title', ''),
                        'url': a.get('url', ''),
                        'pub_date': a.get('publishedAt', ''),
                        'source': (a.get('source') or {}).get('name', ''),
                    }
                    for a in data.get('articles', [])
                    if a.get('title') and a.get('url')
                ]
        except Exception as exc:
            logger.error(f"NewsAPI error: {exc}")
        return []

    def _search_news_rss(self, query: str) -> List[Dict]:
        """Search Google News RSS feed for a query."""
        if not BS4_AVAILABLE:
            logger.warning("RSS search skipped - BeautifulSoup4 is not installed.")
            return []
        try:
            logger.info(f"Searching Google News RSS: {query}")
            rss_url = (
                "https://news.google.com/rss/search"
                f"?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
            )
            response = self.session.get(rss_url, timeout=10)
            if response.status_code != 200:
                logger.error(f"RSS error status: {response.status_code}")
                return []
            soup = BeautifulSoup(response.content, 'xml')
            items = []
            for item in soup.find_all('item')[:self.max_results_per_query]:
                items.append({
                    'title': self._normalize_whitespace(item.title.text if item.title else ""),
                    'url': item.link.text if item.link else "",
                    'pub_date': item.pubDate.text if item.pubDate else "",
                    'source': item.source.text if item.source else "Reported Source",
                })
            return items
        except Exception as exc:
            logger.error(f"RSS search error: {exc}")
        return []

    def _fetch_article_text(self, url: str) -> str:
        if not url:
            return ""
        if url in self.article_cache:
            return self.article_cache[url]

        text = ""
        if NEWSPAPER_AVAILABLE:
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = self._normalize_whitespace(article.text)
            except Exception as exc:
                logger.debug(f"newspaper3k extraction failed for {url}: {exc}")

        if not text and BS4_AVAILABLE:
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                text = self._normalize_whitespace(" ".join(paragraphs))
            except Exception as exc:
                logger.debug(f"HTML extraction failed for {url}: {exc}")

        self.article_cache[url] = text[:12000]
        return self.article_cache[url]

    def _extract_best_matching_snippet(self, claim_text: str, article_text: str) -> str:
        if not article_text:
            return ""
        sentences = re.split(r'(?<=[.!?])\s+', article_text)
        sentences = [self._normalize_whitespace(s) for s in sentences if len(s.strip()) > 40]
        if not sentences:
            return ""
        sample = sentences[:20]
        try:
            matrix = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform([claim_text] + sample)
            sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
            best_idx = int(np.argmax(sims))
            return sample[best_idx][:280]
        except Exception:
            return sample[0][:280]

    def _semantic_similarity(self, claim_text: str, candidate_text: str) -> float:
        if not candidate_text:
            return 0.0
        try:
            matrix = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform([claim_text, candidate_text])
            return float(cosine_similarity(matrix[0:1], matrix[1:2]).flatten()[0])
        except Exception:
            return 0.0

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        deduped = {}
        for item in results:
            title = self._normalize_whitespace(item.get("title", ""))
            url = self._normalize_url(item.get("url", ""))
            key = url or title.lower()
            if not key:
                continue
            merged = dict(item)
            merged["title"] = title
            merged["url"] = url or item.get("url", "")
            if key not in deduped:
                deduped[key] = merged
                continue

            existing = deduped[key]
            if len(merged.get("source", "")) > len(existing.get("source", "")):
                existing["source"] = merged.get("source", existing.get("source", ""))
            if len(merged.get("pub_date", "")) > len(existing.get("pub_date", "")):
                existing["pub_date"] = merged.get("pub_date", existing.get("pub_date", ""))
            existing.setdefault("queries", [])
            existing["queries"].extend(item.get("queries", []))
        return list(deduped.values())

    def _search_deep(self, claim_text: str) -> Dict:
        queries = self._build_query_variants(claim_text)
        collected = []
        for query in queries:
            batch = self._search_news_api(query)
            if not batch:
                batch = self._search_news_rss(query)
            for item in batch:
                item["queries"] = [query]
                collected.append(item)
        deduped = self._deduplicate_results(collected)
        return {
            "queries": queries,
            "results": deduped[: max(self.max_sources_returned * 2, 12)],
        }

    def _score_results(self, claim_text: str, search_payload: Dict) -> List[Dict]:
        candidates = search_payload.get("results", [])
        if not candidates:
            return []

        titles = [item.get("title", "") for item in candidates]
        try:
            title_matrix = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform([claim_text] + titles)
            title_sims = cosine_similarity(title_matrix[0:1], title_matrix[1:]).flatten()
        except Exception:
            title_sims = np.zeros(len(candidates))

        scored = []
        fetch_budget = self.max_article_fetches
        for idx, item in enumerate(candidates):
            url = item.get("url", "")
            source = item.get("source", "")
            domain = self._get_domain(url)
            title_sim = float(title_sims[idx]) if idx < len(title_sims) else 0.0
            credibility = self._domain_credibility(url, source)
            freshness = self._freshness_score(item.get("pub_date", ""))

            article_text = ""
            if fetch_budget > 0 and (title_sim >= 0.08 or credibility >= 0.85):
                article_text = self._fetch_article_text(url)
                fetch_budget -= 1

            body_sim = self._semantic_similarity(claim_text, article_text[:4000]) if article_text else 0.0
            snippet = self._extract_best_matching_snippet(claim_text, article_text) if article_text else ""
            score = (
                0.38 * title_sim +
                0.32 * body_sim +
                0.18 * credibility +
                0.12 * freshness
            )
            scored.append({
                'title': item.get('title', ''),
                'url': url,
                'source': source,
                'domain': domain,
                'score': float(min(score, 1.0)),
                'pure_similarity': float(max(title_sim, body_sim)),
                'headline_similarity': float(title_sim),
                'body_similarity': float(body_sim),
                'credibility_score': float(credibility),
                'freshness_score': float(freshness),
                'published': item.get('pub_date', ''),
                'matched_query': (item.get('queries') or [""])[0],
                'evidence_snippet': snippet,
            })
        return scored

    def _compute_consensus(self, scored_sources: List[Dict]) -> Dict:
        if not scored_sources:
            return {
                "consensus_score": 0.0,
                "verdict_code": "UNVERIFIED",
                "message": "No matching real-time news found to verify this claim.",
            }

        ordered = sorted(scored_sources, key=lambda item: item["score"], reverse=True)
        top = ordered[:5]
        mean_score = float(np.mean([item["score"] for item in top]))
        credible_count = sum(1 for item in top if item["credibility_score"] >= 0.85)
        strong_match_count = sum(1 for item in top if item["pure_similarity"] >= 0.18)
        domains = [item["domain"] for item in top if item.get("domain")]
        unique_domains = len(set(domains))
        domain_counts = Counter(domains)
        dominance_penalty = 0.08 if domain_counts and max(domain_counts.values()) >= 3 else 0.0
        diversity_bonus = min(unique_domains / 5, 1.0) * 0.14
        corroboration_bonus = min((credible_count + strong_match_count) / 8, 1.0) * 0.16
        consensus = max(0.0, min(mean_score + diversity_bonus + corroboration_bonus - dominance_penalty, 1.0))

        if consensus >= 0.72 and credible_count >= 2:
            verdict = "VERIFIED_ONLINE"
            message = (
                f"Claim is supported by {credible_count} highly credible sources across "
                f"{unique_domains} domains with strong semantic overlap."
            )
        elif consensus >= 0.42:
            verdict = "PARTIALLY_SUPPORTED"
            message = (
                f"Claim has some live reporting overlap, but corroboration is limited "
                f"or details differ across sources."
            )
        else:
            verdict = "UNVERIFIED"
            message = "Mainstream reporting does not currently provide strong corroboration for this claim."

        return {
            "consensus_score": consensus,
            "verdict_code": verdict,
            "message": message,
        }

    def verify_claim(self, text: str) -> Dict:
        """
        Verify a claim against real-time news.
        Returns dict with consensus_score and matching sources.
        """
        lang = content_translator.detect_language(text)
        query_text = text
        if lang != 'en':
            query_text, _, _ = content_translator.translate_to_english(text)

        search_payload = self._search_deep(query_text)
        search_results = search_payload.get("results", [])
        queries = search_payload.get("queries", [])

        if not search_results:
            return {
                'consensus_score': 0.0,
                'status': 'NO_RESULTS',
                'sources': [],
                'message': "No matching real-time news found to verify this claim.",
                'timestamp': datetime.now().isoformat(),
                'search_query': queries[0] if queries else self._extract_query(query_text),
                'search_queries': queries,
            }

        scored_sources = self._score_results(query_text, search_payload)
        scored_sources.sort(key=lambda item: item["score"], reverse=True)
        scored_sources = scored_sources[:self.max_sources_returned]

        consensus_payload = self._compute_consensus(scored_sources)

        return {
            'consensus_score': consensus_payload['consensus_score'],
            'status': 'SUCCESS',
            'verdict_code': consensus_payload['verdict_code'],
            'sources': scored_sources,
            'message': consensus_payload['message'],
            'timestamp': datetime.now().isoformat(),
            'search_query': queries[0] if queries else self._extract_query(query_text),
            'search_queries': queries,
            'retrieved_results': len(search_results),
        }

    def extract_article_content(self, url: str) -> str:
        """Extract full article text from a URL."""
        return self._fetch_article_text(url)


class _LazyRealtimeVerifier:
    """Proxy that defers RealtimeNewsVerifier construction until first use."""
    _instance: Optional[RealtimeNewsVerifier] = None

    def _get(self) -> RealtimeNewsVerifier:
        if self._instance is None:
            object.__setattr__(self, '_instance', RealtimeNewsVerifier())
        return self._instance  # type: ignore[return-value]

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def __setattr__(self, name, value):
        if name == '_instance':
            object.__setattr__(self, name, value)
        else:
            setattr(self._get(), name, value)


realtime_verifier = _LazyRealtimeVerifier()
