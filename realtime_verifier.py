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
import ipaddress
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Set, Tuple
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
        self.session.max_redirects = 4
        self.max_results_per_query = 8
        self.max_sources_returned = 10
        self.max_article_fetches = 5
        self.max_cached_articles = 64
        self.max_article_chars = 12000
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
        self.stance_groups = {
            "outcome": (
                {"win", "wins", "won", "victory", "champion", "champions", "beat", "beats", "beating", "defeated", "triumph", "triumphed"},
                {"lose", "loses", "lost", "eliminated", "collapse", "collapsed", "runner", "runnerup", "runner-up"},
            ),
            "approval": (
                {"approve", "approves", "approved", "pass", "passes", "passed", "backed", "backs", "support", "supports", "supported"},
                {"reject", "rejects", "rejected", "block", "blocks", "blocked", "oppose", "opposes", "opposed", "veto", "vetoed"},
            ),
            "confirmation": (
                {"confirm", "confirms", "confirmed", "verify", "verifies", "verified", "true"},
                {"deny", "denies", "denied", "refute", "refutes", "refuted", "false", "fake", "hoax"},
            ),
            "trend": (
                {"increase", "increases", "increased", "rise", "rises", "rose", "higher", "up", "surge", "surged"},
                {"decrease", "decreases", "decreased", "fall", "falls", "fell", "lower", "down", "drop", "dropped"},
            ),
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

    def _is_safe_public_url(self, url: str) -> bool:
        try:
            parsed = urlparse((url or "").strip())
        except Exception:
            return False

        if parsed.scheme not in {"http", "https"}:
            return False
        if parsed.username or parsed.password:
            return False
        if parsed.port not in (None, 80, 443):
            return False

        host = (parsed.hostname or "").strip().lower().rstrip(".")
        if not host or host == "localhost":
            return False
        if host.endswith((".local", ".internal", ".localhost", ".home", ".lan")):
            return False

        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return True

        return not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )

    def _remember_article_text(self, url: str, text: str) -> str:
        normalized = text[:self.max_article_chars]
        self.article_cache[url] = normalized
        while len(self.article_cache) > self.max_cached_articles:
            oldest_key = next(iter(self.article_cache))
            self.article_cache.pop(oldest_key, None)
        return normalized

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
                    if a.get('title') and a.get('url') and self._is_safe_public_url(a.get('url', ''))
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
                item_url = item.link.text if item.link else ""
                if not self._is_safe_public_url(item_url):
                    continue
                items.append({
                    'title': self._normalize_whitespace(item.title.text if item.title else ""),
                    'url': item_url,
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
        if not self._is_safe_public_url(url):
            logger.warning("Skipping non-public article URL: %s", url)
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

        return self._remember_article_text(url, text)

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

    def _tokenize(self, text: str) -> Set[str]:
        return set(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", (text or "").lower()))

    def _negated_hits(self, text: str, terms: Set[str]) -> Set[str]:
        lowered = (text or "").lower()
        hits = set()
        for term in terms:
            if re.search(
                rf"\b(?:not|never|no|did not|didn't|does not|doesn't|was not|wasn't|is not|isn't)\b(?:\W+\w+){{0,2}}\W+{re.escape(term)}\b",
                lowered,
            ):
                hits.add(f"not_{term}")
        return hits

    def _stance_hits(self, text: str, positive_terms: Set[str], negative_terms: Set[str]) -> Tuple[Set[str], Set[str]]:
        tokens = self._tokenize(text)
        positive_hits = set(tokens & positive_terms)
        negative_hits = set(tokens & negative_terms)

        negated_positive_hits = self._negated_hits(text, positive_terms)
        if negated_positive_hits:
            positive_hits -= {term.replace("not_", "", 1) for term in negated_positive_hits}
            negative_hits |= negated_positive_hits

        negated_negative_hits = self._negated_hits(text, negative_terms)
        if negated_negative_hits:
            negative_hits -= {term.replace("not_", "", 1) for term in negated_negative_hits}
            positive_hits |= negated_negative_hits

        return positive_hits, negative_hits

    def _analyze_stance_alignment(self, claim_text: str, candidate_text: str) -> Dict[str, object]:
        contradiction_groups: List[str] = []
        support_groups: List[str] = []

        for group_name, (positive_terms, negative_terms) in self.stance_groups.items():
            claim_pos, claim_neg = self._stance_hits(claim_text, positive_terms, negative_terms)
            cand_pos, cand_neg = self._stance_hits(candidate_text, positive_terms, negative_terms)
            if not (claim_pos or claim_neg or cand_pos or cand_neg):
                continue

            if (claim_pos and cand_neg) or (claim_neg and cand_pos):
                contradiction_groups.append(group_name)
            elif (claim_pos and cand_pos) or (claim_neg and cand_neg):
                support_groups.append(group_name)

        contradiction_penalty = min(0.38 * len(contradiction_groups), 0.68)
        support_bonus = min(0.05 * len(support_groups), 0.12)
        stance_status = "neutral"
        if contradiction_groups:
            stance_status = "contradictory"
        elif support_groups:
            stance_status = "supporting"

        return {
            "stance_status": stance_status,
            "contradiction_penalty": contradiction_penalty,
            "support_bonus": support_bonus,
            "contradiction_groups": contradiction_groups,
            "support_groups": support_groups,
        }

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
            stance = self._analyze_stance_alignment(
                claim_text,
                " ".join(filter(None, [item.get("title", ""), snippet, article_text[:4000]])),
            )
            score = (
                0.38 * title_sim +
                0.32 * body_sim +
                0.18 * credibility +
                0.12 * freshness
            )
            score = max(0.0, min(score + float(stance["support_bonus"]) - float(stance["contradiction_penalty"]), 1.0))
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
                'article_preview': article_text[:1800] if article_text else "",
                'stance_status': stance["stance_status"],
                'contradiction_penalty': float(stance["contradiction_penalty"]),
                'support_bonus': float(stance["support_bonus"]),
                'contradiction_groups': stance["contradiction_groups"],
                'support_groups': stance["support_groups"],
            })
        return scored

    def _compute_consensus(self, scored_sources: List[Dict]) -> Dict:
        if not scored_sources:
            return {
                "consensus_score": 0.0,
                "verdict_code": "UNVERIFIED",
                "message": "No matching real-time news found to verify this claim.",
                "contradiction_score": 0.0,
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
        contradictory_count = sum(1 for item in top if item.get("stance_status") == "contradictory")
        supporting_count = sum(1 for item in top if item.get("stance_status") == "supporting")
        contradiction_score = float(np.mean([item.get("contradiction_penalty", 0.0) for item in top]))
        consensus = max(
            0.0,
            min(mean_score + diversity_bonus + corroboration_bonus - dominance_penalty - contradiction_score, 1.0),
        )

        if contradictory_count >= max(1, supporting_count) and contradiction_score >= 0.2:
            verdict = "CONTRADICTED_BY_SOURCES"
            message = (
                "Matched reporting appears to contradict the claim on key details, "
                "even though the same topic or entities were found."
            )
        elif consensus >= 0.72 and credible_count >= 2:
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
            "contradiction_score": contradiction_score,
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
            'contradiction_score': consensus_payload.get('contradiction_score', 0.0),
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
