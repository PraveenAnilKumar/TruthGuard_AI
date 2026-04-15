"""
TruthGuard AI - Advanced Media Authenticity & Analysis Platform
Integrated with Deepfake Detection, Fake News Detection, and unified Communication Analysis
FULLY OPTIMIZED VERSION - With proper error handling and navigation

OPTIMIZATION CHANGES vs original:
  1. _load_detectors: toxicity block now constructs ToxicityDetector directly
     instead of importing the module-level singleton (which triggered __init__
     at import time even before login).
  2. Login block: auto-upgrades legacy SHA-256 password hashes to bcrypt on
     first successful login.
  3. All other behaviour, UI, and CSS are identical to the original.
  4. Added environment variables to limit CPU threads (memory/CPU optimisation).
  5. Removed heavy CSS that caused blank page on some Windows systems.
"""

# --- Environment optimisations for 8GB RAM ---
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")          # Suppress TensorFlow logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")         # Avoid oneDNN warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")               # Limit OpenMP threads
os.environ.setdefault("MKL_NUM_THREADS", "1")               # Limit Intel MKL threads
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")          # Limit OpenBLAS threads
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")           # Limit NumExpr threads

import sys

def _should_bootstrap_streamlit() -> bool:
    """Only self-launch Streamlit when the script is run directly via Python."""
    if __name__ != "__main__":
        return False

    try:
        from streamlit import runtime as st_runtime
    except Exception:
        return True

    return not st_runtime.exists()


if _should_bootstrap_streamlit():
    import streamlit.web.cli as stcli

    sys.argv = ["streamlit", "run", __file__] + [a for a in sys.argv[1:] if a != __file__]
    raise SystemExit(stcli.main())

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import tempfile
import subprocess
import threading
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import hashlib
import hmac
import base64
import io
import gc
import math
import re
import time
from html import escape
from pathlib import Path
import glob
import logging
from typing import Dict
import bcrypt
from urllib.parse import quote
from dotenv import load_dotenv
from ocr_utils import extract_text_from_image, get_image_reader_status
from safe_deepfake_runtime import (
    run_isolated_deepfake_image_analysis,
    run_isolated_deepfake_video_analysis,
)

# Load environment variables
load_dotenv()

# Secure configuration — all values MUST come from .env in production.
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "")
ADMIN_REG_KEY = os.getenv("ADMIN_REGISTRATION_KEY", "")

if not ADMIN_PASS:
    import warnings
    warnings.warn(
        "ADMIN_PASSWORD is not set in .env — admin login will be unavailable. "
        "Create a .env file with ADMIN_PASSWORD set.",
        RuntimeWarning,
        stacklevel=1,
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_VIDEO_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_TEXT_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_BULK_TEXT_ROWS = 250
MAX_IMAGE_PIXELS = 25_000_000
LOGIN_WINDOW_SECONDS = 300
MAX_LOGIN_ATTEMPTS = 5
LOGIN_COOLDOWN_SECONDS = 60
VALID_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
USER_DB_LOCK = threading.RLock()

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# ============================================================================
# PAGE CONFIG (MUST BE FIRST)
# ============================================================================
st.set_page_config(
    page_title="TruthGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# DETECTOR PLACEHOLDERS (loaded lazily after login to avoid blocking startup)
# ============================================================================

deepfake_detector = None
fake_news_detector = None
sentiment_analyzer = None
toxicity_detector = None
realtime_verifier = None
DEEPFAKE_AVAILABLE = False
FAKE_NEWS_AVAILABLE = False
SENTIMENT_AVAILABLE = False
TOXICITY_AVAILABLE = False
REALTIME_AVAILABLE = False
ASPECT_AVAILABLE = False
VIZ_AVAILABLE = False


@st.cache_resource(ttl=3600, show_spinner="Loading models...")
def _load_detectors():
    """
    Load all detectors on first authenticated access.
    Cached by st.cache_resource to avoid re-init on every Streamlit rerun.

    OPTIMIZED:
    - Toxicity now constructs ToxicityDetector directly instead of importing
      the module-level singleton (which caused __init__ to run at import time).
    - Sentiment, FakeNews, and Deepfake are imported and constructed here so
      their module-level singletons (which are now lazy-proxies) are never
      materialised until this function is called after login.
    """
    _df, _fn, _sa, _tox, _rt = None, None, None, None, None
    df_ok, fn_ok, sa_ok, tox_ok, rt_ok = False, False, False, False, False
    _ToxViz = None

    try:
        from deepfake_detector_advanced import deepfake_detector as _df
        df_ok = True
        logger.info("✅ Deepfake detector loaded")
    except Exception as e:
        logger.error(f"❌ Deepfake detector failed: {e}")

    try:
        from fake_news_detector import FakeNewsDetector
        _fn = FakeNewsDetector()
        fn_ok = True
        logger.info("✅ Fake news detector loaded")
    except Exception as e:
        logger.error(f"❌ Fake news detector failed: {e}")

    try:
        from sentiment_analyzer import SentimentAnalyzer
        _sa = SentimentAnalyzer(use_ensemble=True)
        sa_ok = True
        logger.info("✅ Sentiment analyzer loaded")
    except Exception as e:
        logger.error(f"❌ Sentiment analyzer failed: {e}")

    # ── OPTIMIZED TOXICITY BLOCK ──────────────────────────────────────────────
    try:
        from toxicity_detector import ToxicityDetector
        from toxicity_viz import ToxicityVisualizer as _ToxViz
        _tox = ToxicityDetector(use_ensemble=True)
        tox_ok = True
        logger.info("✅ Toxicity modules loaded")
    except ImportError as e:
        _ToxViz = None
        logger.warning(f"⚠️ Toxicity modules not available: {e}")
    except Exception as e:
        _ToxViz = None
        logger.error(f"❌ Toxicity detector failed: {e}")

    try:
        from realtime_verifier import realtime_verifier as _rt
        rt_ok = True
    except Exception as e:
        logger.error(f"❌ Real-time verifier failed to load: {e}")

    try:
        from aspect_sentiment import AspectSentimentAnalyzer
        _aspect = AspectSentimentAnalyzer
    except ImportError:
        _aspect = None

    try:
        from sentiment_viz import SentimentVisualizer
        _viz = SentimentVisualizer
    except ImportError:
        _viz = None

    return {
        'deepfake': _df, 'fake_news': _fn, 'sentiment': _sa,
        'toxicity': _tox, 'realtime': _rt,
        'aspect_analyzer': _aspect, 'sentiment_viz': _viz,
        'toxicity_viz': _ToxViz if tox_ok else None,
        'df_ok': df_ok, 'fn_ok': fn_ok, 'sa_ok': sa_ok,
        'tox_ok': tox_ok, 'rt_ok': rt_ok,
    }


# ============================================================================
# USER DATABASE FUNCTIONS
# ============================================================================

USER_DB = "users.json"


def _normalize_username(username):
    return re.sub(r"\s+", " ", str(username or "")).strip()


def _resolve_user_record(users, username):
    normalized = _normalize_username(username).casefold()
    if not normalized:
        return None, None
    for stored_name, payload in users.items():
        if str(stored_name).casefold() == normalized:
            return stored_name, payload
    return None, None


def _password_meets_policy(password):
    password = str(password or "")
    return (
        len(password) >= 8
        and any(ch.isalpha() for ch in password)
        and any(ch.isdigit() for ch in password)
    )


def _prune_login_attempts(now=None):
    now = time.time() if now is None else float(now)
    attempts = [
        float(ts)
        for ts in st.session_state.get("login_attempt_timestamps", [])
        if isinstance(ts, (int, float)) and now - float(ts) < LOGIN_WINDOW_SECONDS
    ]
    st.session_state.login_attempt_timestamps = attempts
    blocked_until = float(st.session_state.get("login_blocked_until", 0.0) or 0.0)
    if blocked_until and blocked_until <= now:
        st.session_state.login_blocked_until = 0.0
    return attempts


def _remaining_login_cooldown_seconds():
    now = time.time()
    _prune_login_attempts(now)
    blocked_until = float(st.session_state.get("login_blocked_until", 0.0) or 0.0)
    if blocked_until <= now:
        return 0
    return int(math.ceil(blocked_until - now))


def _record_failed_login_attempt():
    now = time.time()
    attempts = _prune_login_attempts(now)
    attempts.append(now)
    st.session_state.login_attempt_timestamps = attempts
    if len(attempts) >= MAX_LOGIN_ATTEMPTS:
        st.session_state.login_blocked_until = now + LOGIN_COOLDOWN_SECONDS


def _reset_login_rate_limit():
    st.session_state.login_attempt_timestamps = []
    st.session_state.login_blocked_until = 0.0


def _format_file_size(size_bytes):
    size = float(max(size_bytes or 0, 0))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{int(size_bytes)} B"


def _uploaded_file_size(uploaded_file):
    size = getattr(uploaded_file, "size", None)
    if isinstance(size, int) and size >= 0:
        return size
    try:
        current = uploaded_file.tell()
        uploaded_file.seek(0, 2)
        size = uploaded_file.tell()
        uploaded_file.seek(current)
        return max(size, 0)
    except Exception:
        try:
            return len(uploaded_file.getbuffer())
        except Exception:
            return 0


def _require_upload_within_limit(uploaded_file, max_bytes, label):
    size = _uploaded_file_size(uploaded_file)
    if size and size > max_bytes:
        raise ValueError(
            f"{label} is too large ({_format_file_size(size)}). "
            f"Keep it under {_format_file_size(max_bytes)} for reliable analysis."
        )


def _load_uploaded_image(uploaded_file, *, max_size=(800, 800), label="Image upload"):
    _require_upload_within_limit(uploaded_file, MAX_IMAGE_UPLOAD_BYTES, label)
    uploaded_file.seek(0)
    with Image.open(uploaded_file) as probe:
        if probe.width * probe.height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"{label} is too large to process safely ({probe.width}x{probe.height} pixels). "
                f"Please upload a smaller image."
            )
        probe.verify()
    uploaded_file.seek(0)
    with Image.open(uploaded_file) as opened:
        opened.load()
        working = ImageOps.exif_transpose(opened).copy()
    uploaded_file.seek(0)
    return optimize_image(working, max_size=max_size)


def _read_uploaded_csv(uploaded_file, *, label="CSV upload", nrows=None):
    _require_upload_within_limit(uploaded_file, MAX_TEXT_UPLOAD_BYTES, label)
    uploaded_file.seek(0)
    dataframe = pd.read_csv(uploaded_file, nrows=nrows)
    uploaded_file.seek(0)
    return dataframe


def _read_uploaded_text_lines(uploaded_file, *, label="Text upload", max_lines=MAX_BULK_TEXT_ROWS):
    _require_upload_within_limit(uploaded_file, MAX_TEXT_UPLOAD_BYTES, label)
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    if isinstance(raw, bytes):
        decoded = raw.decode("utf-8", errors="replace")
    else:
        decoded = str(raw)
    lines = [line.strip() for line in decoded.splitlines() if line.strip()]
    return lines[:max_lines], len(lines) > max_lines


def _write_uploaded_video_tempfile(uploaded_file):
    _require_upload_within_limit(uploaded_file, MAX_VIDEO_UPLOAD_BYTES, "Video upload")
    suffix = Path(getattr(uploaded_file, "name", "") or "upload.mp4").suffix or ".mp4"
    uploaded_file.seek(0)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        handle.write(uploaded_file.read())
        return handle.name
    finally:
        handle.close()
        uploaded_file.seek(0)


def get_password_hash(password):
    """Secure password hashing using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password, hashed_password):
    """Verify password against hashed version (supports legacy SHA-256 and bcrypt)."""
    try:
        if len(hashed_password) == 64:  # Legacy SHA-256
            return hashlib.sha256(password.encode()).hexdigest() == hashed_password
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


@st.cache_data(ttl=60)
def load_users():
    """Load and cache the user database from disk."""
    if Path(USER_DB).exists():
        try:
            with USER_DB_LOCK:
                with open(USER_DB, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception as e:
            logger.error(f"Error loading user database: {e}")
    return {}


def save_users(users):
    """Persist the user database and invalidate the read cache immediately."""
    temp_path = None
    try:
        db_path = Path(USER_DB)
        temp_path = db_path.with_suffix(db_path.suffix + ".tmp")
        with USER_DB_LOCK:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, db_path)
        load_users.clear()
    except Exception as e:
        logger.error(f"Error saving user database: {e}")
        if temp_path is not None:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass


def register_user(username, password, role="user"):
    """Create a new local user account."""
    username = _normalize_username(username)
    role = str(role or "user").strip().lower()
    if not username:
        return False, "Username is required."
    if not VALID_USERNAME_RE.fullmatch(username):
        return False, "Username must be 3-32 characters and use only letters, numbers, '.', '_' or '-'."
    if not _password_meets_policy(password):
        return False, "Password must be at least 8 characters and include both letters and numbers."
    if role not in {"user", "admin"}:
        return False, "Invalid role requested."

    users = load_users()
    existing_name, _ = _resolve_user_record(users, username)
    if username.casefold() == ADMIN_USER.casefold() or existing_name is not None:
        return False, "That username already exists."

    users[username] = {
        "password": get_password_hash(password),
        "role": role,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_users(users)
    return True, "Account created successfully. You can log in now."


@st.cache_resource(ttl=3600, show_spinner="Loading deepfake detector...")
def get_deepfake_detector():
    from deepfake_detector_advanced import deepfake_detector
    logger.info("✅ Deepfake detector initialized and cached")
    return deepfake_detector


DEEPFAKE_ENSEMBLE_LABEL = "Ensemble (Optional, Higher Memory)"


@st.cache_data(ttl=300, show_spinner=False)
def get_available_deepfake_models():
    """Read ranked local deepfake models and expose the preferred single-model default."""
    from deepfake_detector_advanced import DeepfakeDetectorAdvanced

    detector = DeepfakeDetectorAdvanced(models_dir="models")
    info = detector.get_model_info()
    local_models = info.get("local_models", [])
    selectable_hf_models = list(getattr(detector, "available_hf_models", {}).keys())
    weights = dict(info.get("weights", {}))
    for model_name in selectable_hf_models:
        weights.setdefault(model_name, 1.2)
    return {
        "model_names": list(dict.fromkeys(local_models + selectable_hf_models)),
        "model_paths": info.get("model_paths", []),
        "local_models": local_models,
        "hf_models": selectable_hf_models,
        "preferred_model_name": info.get("preferred_model_name"),
        "recommended_threshold": info.get("recommended_threshold", 0.5),
        "weights": weights,
    }


@st.cache_resource(ttl=3600, show_spinner="Loading fake news detector...")
def get_fake_news_detector():
    from fake_news_detector import FakeNewsDetector
    return FakeNewsDetector()


@st.cache_resource(ttl=3600, show_spinner="Loading sentiment analyzer...")
def get_sentiment_analyzer():
    from sentiment_analyzer import SentimentAnalyzer
    return SentimentAnalyzer(use_ensemble=True)


@st.cache_resource(ttl=3600, show_spinner="Loading toxicity detector...")
def get_toxicity_detector():
    from toxicity_detector import ToxicityDetector
    return ToxicityDetector(use_ensemble=True)


@st.cache_resource(ttl=3600, show_spinner="Loading realtime verifier...")
def get_realtime_verifier():
    from realtime_verifier import realtime_verifier
    return realtime_verifier


@st.cache_resource(ttl=3600, show_spinner=False)
def get_aspect_sentiment_analyzer():
    from aspect_sentiment import AspectSentimentAnalyzer
    return AspectSentimentAnalyzer


@st.cache_resource(ttl=3600, show_spinner=False)
def get_sentiment_visualizer():
    from sentiment_viz import SentimentVisualizer
    return SentimentVisualizer


@st.cache_resource(ttl=3600, show_spinner=False)
def get_toxicity_visualizer():
    from toxicity_viz import ToxicityVisualizer
    return ToxicityVisualizer


def _compact_text(value: str, limit: int) -> str:
    cleaned = " ".join((value or "").split())
    if len(cleaned) <= limit:
        return cleaned
    trimmed = cleaned[:limit].rsplit(" ", 1)[0].strip()
    return (trimmed or cleaned[:limit]).rstrip(".,;: ") + "..."


def _safe_url(value: str) -> str:
    value = (value or "").strip()
    return value if value.startswith(("http://", "https://")) else "#"


@st.cache_data(ttl=900, show_spinner=False)
def get_related_article_preview(url: str) -> str:
    safe_url = _safe_url(url)
    if safe_url == "#":
        return ""
    try:
        verifier = get_realtime_verifier()
        if verifier is None:
            return ""
        return " ".join((verifier.extract_article_content(safe_url) or "").split())
    except Exception as exc:
        logger.warning(f"Could not load related article preview for {safe_url}: {exc}")
        return ""


COMMUNICATION_PAGE = "Communication Analysis"
LEGACY_PAGE_REDIRECTS = {
    "Sentiment Analysis": COMMUNICATION_PAGE,
    "Toxicity Checker": COMMUNICATION_PAGE,
}


def load_component(loader, name):
    """Load one feature only when needed."""
    try:
        component = loader()
        logger.info("%s loaded successfully", name)
        return component, True
    except Exception as e:
        logger.error("%s failed to load: %s", name, e)
        return None, False


def release_feature_resources(page_name):
    """Clear cached resources for the previous page to reduce memory usage."""
    clear_map = {
        "Deepfake Detection": [get_available_deepfake_models],
        "Fake News Detection": [get_fake_news_detector, get_realtime_verifier],
        COMMUNICATION_PAGE: [
            get_sentiment_analyzer,
            get_aspect_sentiment_analyzer,
            get_sentiment_visualizer,
            get_toxicity_detector,
            get_toxicity_visualizer,
        ],
        "Sentiment Analysis": [get_sentiment_analyzer, get_aspect_sentiment_analyzer, get_sentiment_visualizer],
        "Toxicity Checker": [get_toxicity_detector, get_toxicity_visualizer],
    }
    for func in clear_map.get(page_name, []):
        try:
            func.clear()
        except Exception:
            pass

    _clear_feature_session_state(page_name)
    gc.collect()


def render_live_news_comparison(rt_result, claim_text):
    """Render live-news evidence in a cleaner, more readable layout."""
    st.markdown("### Live News Comparison")
    st.caption(f"Search query used: `{rt_result.get('search_query', '')}`")
    if rt_result.get("search_queries"):
        st.caption("Deep search variants: " + " | ".join(f"`{q}`" for q in rt_result.get("search_queries", [])[:5]))

    if rt_result.get("status") == "NO_RESULTS":
        st.info("No matching mainstream reports were found for this claim.")
        return

    sources = rt_result.get("sources", [])
    if not sources:
        return

    top_source = sources[0]
    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"""
            <div class="live-news-summary">
                <div class="live-news-label">Claim Under Review</div>
                <div class="live-news-summary-text">{escape(_compact_text(claim_text, 420))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        top_source_name = top_source.get("source") or top_source.get("domain") or "Matched source"
        st.markdown(
            f"""
            <div class="live-news-summary">
                <div class="live-news-label">Closest Live Match</div>
                <div class="live-news-title">{escape(_compact_text(top_source.get("title", ""), 180) or "Untitled article")}</div>
                <div class="live-news-time">{escape(top_source_name)}</div>
                <div class="live-news-summary-text">{escape(rt_result.get("message", ""))}</div>
                <a class="live-news-link" href="{escape(_safe_url(top_source.get('url', '#')), quote=True)}">Open article</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Matching Articles")
    shown_sources = sources[:5]
    for source in shown_sources:
        source_name = source.get("source") or source.get("domain") or "Unknown source"
        title = source.get("title") or "Untitled article"
        snippet = source.get("evidence_snippet") or "No article excerpt was extracted for this result, but the headline still matched the claim."
        published = source.get("published") or "Publication time unavailable"
        st.markdown(
            f"""
            <div class="live-news-card">
                <div class="live-news-card-head">
                    <div>
                        <div class="live-news-source">{escape(source_name)}</div>
                        <div class="live-news-time">{escape(_compact_text(published, 48))}</div>
                    </div>
                    <div class="live-news-meta">
                        <span class="live-news-pill">Match {source.get('score', 0.0):.0%}</span>
                        <span class="live-news-pill">Credibility {source.get('credibility_score', 0.0):.0%}</span>
                        <span class="live-news-pill">Body {source.get('body_similarity', 0.0):.0%}</span>
                    </div>
                </div>
                <div class="live-news-title">{escape(_compact_text(title, 220))}</div>
                <div class="live-news-snippet">{escape(_compact_text(snippet, 320))}</div>
                <a class="live-news-link" href="{escape(_safe_url(source.get('url', '#')), quote=True)}">Read source article</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if len(sources) > len(shown_sources):
        st.caption(f"Showing the top {len(shown_sources)} live matches out of {len(sources)} retrieved sources.")

    rows = []
    for source in sources:
        rows.append({
            "Source": source.get("source", ""),
            "Headline": source.get("title", ""),
            "Match Score": round(source.get("score", 0.0), 3),
            "Text Similarity": round(source.get("pure_similarity", 0.0), 3),
            "Credibility": round(source.get("credibility_score", 0.0), 3),
            "Body Match": round(source.get("body_similarity", 0.0), 3),
            "Published": source.get("published", ""),
        })
    with st.expander("Technical Match Breakdown"):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_article_match_browser(
    rt_result,
    widget_namespace="fn_match",
    section_title="Matching Articles",
    intro_text=None,
):
    """Render a prominent matching-articles section for fake-news verification results."""
    if not rt_result or rt_result.get("status") == "NO_RESULTS":
        return

    sources = [
        source for source in rt_result.get("sources", [])
        if _safe_url(source.get("url", "")) != "#"
    ]
    if not sources:
        return

    st.markdown(f"### {section_title}")
    st.caption(
        intro_text
        or "Browse the live articles this verification matched against before accepting the verdict."
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.metric("Matched Sources", len(sources))
    with summary_col2:
        avg_match = float(np.mean([source.get("score", 0.0) for source in sources]))
        st.metric("Average Match", f"{avg_match:.0%}")
    with summary_col3:
        top_credibility = max(source.get("credibility_score", 0.0) for source in sources)
        st.metric("Top Credibility", f"{top_credibility:.0%}")

    options = list(range(len(sources)))
    selected_idx = st.selectbox(
        "Browse matched articles",
        options=options,
        format_func=lambda idx: (
            f"{sources[idx].get('source') or sources[idx].get('domain') or 'Source'}"
            f" - {_compact_text(sources[idx].get('title', '') or 'Untitled article', 90)}"
        ),
        key=f"{widget_namespace}_related_article_idx",
    )
    selected = sources[selected_idx]

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Match Score", f"{selected.get('score', 0.0):.0%}")
    with metric_col2:
        st.metric("Credibility", f"{selected.get('credibility_score', 0.0):.0%}")
    with metric_col3:
        st.metric("Body Match", f"{selected.get('body_similarity', 0.0):.0%}")

    st.markdown(f"#### {selected.get('title') or 'Untitled article'}")
    st.caption(
        f"{selected.get('source') or selected.get('domain') or 'Unknown source'}"
        f" | {selected.get('published') or 'Publication time unavailable'}"
    )
    safe_url = _safe_url(selected.get("url", ""))
    if safe_url != "#":
        st.markdown(f"[Open original article]({safe_url})")

    preview_text = " ".join((selected.get("article_preview") or "").split())
    if not preview_text:
        preview_text = get_related_article_preview(selected.get("url", ""))

    if preview_text:
        st.text_area(
            "Article Preview",
            value=preview_text,
            height=260,
            disabled=True,
            key=f"{widget_namespace}_article_preview_{selected_idx}",
        )
    else:
        fallback_snippet = selected.get("evidence_snippet") or "No article body preview could be extracted for this match."
        st.info(_compact_text(fallback_snippet, 360))


def render_related_article_viewer(rt_result, widget_namespace="fn_image"):
    """Backward-compatible wrapper for OCR-based article browsing."""
    render_article_match_browser(
        rt_result,
        widget_namespace=widget_namespace,
        section_title="Related Articles From Extracted Text",
        intro_text="Browse the matched live articles found from the text extracted out of the uploaded article image.",
    )


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

default_state = {
    'authenticated': False,
    'username': None,
    'role': None,
    'page': 'Home',
    'df_result': None,
    'df_preview_image': None,
    'fn_result': None,
    'sentiment_result': None,
    'toxicity_result': None,
    'last_sentiment': ('NEUTRAL', 0.5),
    'batch_results': None,
    'aspect_text': "",
    'sentiment_text_input': "",
    'communication_text_input': "",
    'fn_text': "",
    'fn_text_input': "",
    'selected_df_model': None,
    'df_model_weights': {},
    'selected_fn_model': None,
    'clear_sentiment': False,
    'clear_fn': False,
    'communication_image_upload_nonce': 0,
    'fn_check_realtime': True,
    'fn_image_upload_nonce': 0,
    'last_toxicity': None,
    'toxicity_train_data': None,
    'fn_verification_result': None,
    'login_attempt_timestamps': [],
    'login_blocked_until': 0.0,
    'page_num': 0,
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.page in LEGACY_PAGE_REDIRECTS:
    st.session_state.page = LEGACY_PAGE_REDIRECTS[st.session_state.page]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _segmented_control(label, options, default=None, key=None):
    """Streamlit segmented_control with fallback for older versions."""
    default_val = default or (options[0] if options else None)
    if hasattr(st, 'segmented_control'):
        return st.segmented_control(label, options, default=default_val, key=key)
    idx = options.index(default_val) if default_val and default_val in options else 0
    return st.radio(label, options, index=idx, horizontal=True, key=key)


def _render_plotly_chart(fig, key=None, config=None):
    """Render Plotly charts without triggering deprecated keyword warnings."""
    kwargs = {}
    if key is not None:
        kwargs["key"] = key
    if config is not None:
        kwargs["config"] = config
    return st.plotly_chart(fig, use_container_width=True, **kwargs)


SESSION_ARTIFACT_DIR = Path("temp") / "streamlit_artifacts"
SESSION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_delete_artifact(path):
    if not path:
        return
    try:
        artifact_path = Path(path)
        if artifact_path.exists():
            artifact_path.unlink()
    except Exception:
        logger.warning("Could not delete artifact: %s", path)


def _persist_dataframe_artifact(df: pd.DataFrame, prefix: str) -> Dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    artifact_path = SESSION_ARTIFACT_DIR / f"{prefix}_{timestamp}.csv.gz"
    df.to_csv(artifact_path, index=False, compression="gzip")
    return {
        "artifact_path": str(artifact_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }


def _load_dataframe_artifact(payload: Dict) -> pd.DataFrame:
    artifact_path = payload.get("artifact_path")
    if not artifact_path:
        return payload.get("df", pd.DataFrame())
    try:
        return pd.read_csv(artifact_path)
    except Exception as e:
        logger.error("Could not load dataframe artifact %s: %s", artifact_path, e)
        return pd.DataFrame()


def _serialize_display_image(image, max_size=(1280, 1280), prefer_lossless=True):
    if image is None:
        return None
    if isinstance(image, bytes):
        return image

    if isinstance(image, Image.Image):
        working = image.copy()
        working.thumbnail(max_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        save_format = "PNG" if prefer_lossless else "JPEG"
        save_kwargs = {"format": save_format}
        if save_format == "JPEG":
            save_kwargs.update({"quality": 88, "optimize": True})
            if working.mode not in ("RGB", "L"):
                working = working.convert("RGB")
        working.save(buffer, **save_kwargs)
        return buffer.getvalue()

    if isinstance(image, np.ndarray):
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] > 4:
            arr = arr[:, :, :3]

        height, width = arr.shape[:2]
        if width > max_size[0] or height > max_size[1]:
            scale = min(max_size[0] / max(width, 1), max_size[1] / max(height, 1))
            resized_width = max(1, int(width * scale))
            resized_height = max(1, int(height * scale))
            arr = cv2.resize(arr, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

        extension = ".png" if prefer_lossless else ".jpg"
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 88] if extension == ".jpg" else []
        success, encoded = cv2.imencode(extension, arr, encode_params)
        if success:
            return encoded.tobytes()

    return image


def _compact_deepfake_result(result: Dict) -> Dict:
    compact = dict(result)
    for key in ("heatmap", "ela_image", "fft_image"):
        if compact.get(key) is not None:
            compact[key] = _serialize_display_image(compact[key], max_size=(1280, 1280))
    return compact


def _clear_feature_session_state(page_name: str):
    if page_name == "Deepfake Detection":
        st.session_state.df_result = None
        st.session_state.df_preview_image = None
    elif page_name == "Fake News Detection":
        st.session_state.fn_result = None
        st.session_state.fn_verification_result = None
        st.session_state.fn_image_upload_nonce = st.session_state.get("fn_image_upload_nonce", 0) + 1
        st.session_state.pop("active_fn_model_path", None)
    elif page_name == COMMUNICATION_PAGE:
        payload = st.session_state.get("sentiment_result")
        if isinstance(payload, dict):
            _safe_delete_artifact(payload.get("artifact_path"))
        st.session_state.sentiment_result = None
        st.session_state.batch_results = None
        st.session_state.toxicity_result = None
        st.session_state.last_toxicity = None
    elif page_name == "Sentiment Analysis":
        payload = st.session_state.get("sentiment_result")
        if isinstance(payload, dict):
            _safe_delete_artifact(payload.get("artifact_path"))
        st.session_state.sentiment_result = None
        st.session_state.batch_results = None
    elif page_name == "Toxicity Checker":
        st.session_state.toxicity_result = None
        st.session_state.last_toxicity = None


def release_all_feature_resources():
    for page_name in ("Deepfake Detection", "Fake News Detection", COMMUNICATION_PAGE):
        release_feature_resources(page_name)


def optimize_image(image, max_size=(800, 800)):
    """Resize large images before processing."""
    working = image.copy()
    if working.size[0] > max_size[0] or working.size[1] > max_size[1]:
        working.thumbnail(max_size, Image.Resampling.LANCZOS)
    return working


def create_gauge(value, threshold=0.5, title="Confidence", height=250):
    """Create a premium glass-styled gauge chart."""
    color = "#ef4444" if value > threshold else "#10b981"
    if threshold == 0.5 and (0.4 <= value <= 0.6):
        color = "#f59e0b"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': "%", 'font': {'size': 40, 'color': 'white', 'family': 'Plus Jakarta Sans'}},
        title={'text': title, 'font': {'size': 18, 'color': '#94a3b8'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.1)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.1)'},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20),
        height=height,
        font={'family': 'Plus Jakarta Sans'},
    )
    return fig


def summarize_safety_result(toxicity_result):
    if not toxicity_result:
        return {
            "threshold": 0.65,
            "risk": 0.0,
            "top_category": "none",
            "top_score": 0.0,
            "band": "Unavailable",
            "summary": "No safety result available.",
        }

    risk = float(np.clip(toxicity_result.get("confidence", 0.0), 0.0, 1.0))
    threshold = float(
        toxicity_result.get("threshold")
        or (toxicity_result.get("meta") or {}).get("decision_threshold")
        or 0.65
    )
    threshold = float(np.clip(threshold, 0.1, 0.95))

    categories = toxicity_result.get("categories") or {}
    top_category = max(categories, key=categories.get) if categories else "none"
    top_score = float(categories.get(top_category, 0.0)) if categories else 0.0

    if toxicity_result.get("is_toxic"):
        if risk >= max(threshold + 0.25, 0.88):
            band = "Critical"
        elif risk >= threshold + 0.12:
            band = "High"
        else:
            band = "Review"
        summary = f"Flagged: {risk:.1%} over a {threshold:.1%} threshold."
    else:
        if risk >= max(threshold - 0.05, 0.0):
            band = "Borderline"
        elif risk >= threshold * 0.55:
            band = "Low"
        else:
            band = "Minimal"
        summary = f"Below threshold: {risk:.1%} under a {threshold:.1%} threshold."

    return {
        "threshold": threshold,
        "risk": risk,
        "top_category": top_category,
        "top_score": top_score,
        "band": band,
        "summary": summary,
    }


def render_communication_summary(sentiment_result, toxicity_result):
    single_sentiment = sentiment_result if sentiment_result and sentiment_result.get("type") == "single" else None
    if not single_sentiment and not toxicity_result:
        return

    st.markdown('<div class="glass-card animate-fade">', unsafe_allow_html=True)
    st.markdown("### 🧠 Communication Snapshot")
    st.caption("Tone and safety signals are summarized together for the latest single-message analysis.")
    col1, col2 = st.columns(2)

    with col1:
        if single_sentiment:
            label = single_sentiment.get("label", "NEUTRAL")
            meta = single_sentiment.get("meta") or {}
            tone_copy = {
                "POSITIVE": "Constructive or favorable tone",
                "NEGATIVE": "Critical or unfavorable tone",
                "NEUTRAL": "Balanced or objective tone",
            }.get(label, "Mixed emotional signal")
            st.metric("Sentiment", label.title(), f"{single_sentiment.get('conf', 0.0):.1%} confidence")
            dominant_emotion = str(meta.get("dominant_emotion", "balanced")).replace("_", " ").title()
            st.caption(f"{tone_copy} | Dominant emotion: {dominant_emotion}")
        else:
            st.metric("Sentiment", "Unavailable")
            st.caption("Run a unified scan to generate an emotional-intent reading.")

    with col2:
        if toxicity_result:
            safety = summarize_safety_result(toxicity_result)
            if toxicity_result.get("is_toxic"):
                st.metric("Safety", "Needs Review", f"{safety['risk']:.1%} risk")
            else:
                st.metric("Safety", "Clear", f"{safety['risk']:.1%} risk")
            st.caption(
                f"{safety['summary']} Highest signal: {safety['top_category'].replace('_', ' ').title()} ({safety['top_score']:.1%})."
            )
        else:
            st.metric("Safety", "Unavailable")
            st.caption("Run a unified scan to generate a toxicity and harm review.")

    st.markdown('</div>', unsafe_allow_html=True)


def render_ocr_result_panel(ocr_meta, section_title="Image Reader"):
    if not ocr_meta:
        return

    st.markdown('<div class="glass-card" style="border-left: 4px solid #22c55e;">', unsafe_allow_html=True)
    st.markdown(f"### {section_title}")

    if ocr_meta.get("image_name"):
        st.caption(f"Source image: {ocr_meta['image_name']}")

    ocr_col1, ocr_col2, ocr_col3 = st.columns(3)
    with ocr_col1:
        st.metric("OCR Backend", str(ocr_meta.get("backend", "windows_ocr")).replace("_", " ").title())
    with ocr_col2:
        st.metric("Detected Lines", int(ocr_meta.get("line_count", 0) or 0))
    with ocr_col3:
        st.metric("Detected Words", int(ocr_meta.get("word_count", 0) or 0))

    extracted_text = ocr_meta.get("extracted_text", "")
    if extracted_text:
        st.text_area(
            "Extracted Text",
            value=extracted_text,
            height=180,
            disabled=True,
        )

    if ocr_meta.get("language"):
        st.caption(f"OCR language profile: {ocr_meta['language']}")

    if ocr_meta.get("width") and ocr_meta.get("height"):
        st.caption(f"Original image size: {int(ocr_meta['width'])} × {int(ocr_meta['height'])}")

    preprocessed_size = ocr_meta.get("preprocessed_size")
    if isinstance(preprocessed_size, (list, tuple)) and len(preprocessed_size) == 2:
        st.caption(f"OCR working size: {int(preprocessed_size[0])} × {int(preprocessed_size[1])}")

    st.markdown('</div>', unsafe_allow_html=True)


def render_sentiment_result_panel(result):
    if not result:
        return

    if result['type'] == 'single':
        if result['label'] == 'POSITIVE':
            cls, label, sub = "res-safe", "😊 POSITIVE SENTIMENT", "Content conveys optimistic and favorable emotions."
        elif result['label'] == 'NEGATIVE':
            cls, label, sub = "res-danger", "😠 NEGATIVE SENTIMENT", "Content conveys critical or unfavorable emotions."
        else:
            cls, label, sub = "res-warn", "😐 NEUTRAL TONE", "Content appears objective or emotionally balanced."

        st.markdown(f'''
        <div class="result-container {cls}">
            <div class="res-label">{label}</div>
            <div class="res-sub">{sub}</div>
        </div>
        ''', unsafe_allow_html=True)

        if result.get('meta') and result['meta'].get('was_translated'):
            st.warning(f"🌐 Audio/Text Translation: Analyzing tone from original **{result['meta']['original_language'].upper()}** content (Translated for model processing).")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Primary Sentiment")
            meta = result.get("meta") or {}
            metrics = st.columns(2)
            with metrics[0]:
                st.metric("Model Confidence", f"{result['conf']:.1%}")
            with metrics[1]:
                st.metric("Sentiment Score", f"{meta.get('sentiment_score', 0.0):+.2f}")
            metrics_2 = st.columns(2)
            with metrics_2[0]:
                st.metric(
                    "Dominant Emotion",
                    str(meta.get("dominant_emotion", "balanced")).replace("_", " ").title(),
                )
            with metrics_2[1]:
                st.metric("Model Agreement", f"{meta.get('agreement', 0.0):.1%}")
            st.progress(result['conf'])
            tone_flags = meta.get("tone_flags") or []
            if tone_flags:
                st.caption("Tone markers: " + ", ".join(flag.replace("_", " ") for flag in tone_flags))
            st.info("Analysis based on linguistic sentiment markers and tonal classification.")
        with col2:
            fig = create_gauge(result['conf'], height=300)
            _render_plotly_chart(fig)

    elif result['type'] == 'batch':
        df = _load_dataframe_artifact(result)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Batch Analysis Overview")
        col1, col2, col3 = st.columns(3)
        counts = df['sentiment'].value_counts() if not df.empty else pd.Series(dtype="int64")
        pos = counts.get('POSITIVE', 0)
        neg = counts.get('NEGATIVE', 0)
        neu = counts.get('NEUTRAL', 0)
        total_rows = max(len(df), 1)
        with col1:
            st.metric("Positive", pos, f"{(pos/total_rows):.1%}")
        with col2:
            st.metric("Negative", neg, f"-{(neg/total_rows):.1%}", delta_color="inverse")
        with col3:
            st.metric("Neutral", neu, "Base")
        st.markdown("---")
        SentimentVisualizer, VIZ_AVAILABLE = load_component(get_sentiment_visualizer, "Sentiment visualizer")
        if VIZ_AVAILABLE and SentimentVisualizer is not None and not df.empty:
            col_left, col_right = st.columns(2)
            with col_left:
                fig = SentimentVisualizer.create_pie_chart(df)
                _render_plotly_chart(fig)
            with col_right:
                fig = SentimentVisualizer.create_bar_chart(df)
                _render_plotly_chart(fig)
        else:
            st.info("Charts unavailable. Install sentiment_viz for visualizations.")
        st.markdown("#### Detailed Transmission Log")
        st.dataframe(df, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    elif result['type'] == 'aspect':
        results = result['results']
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Aspect-Based Breakdown")
        if not results:
            st.info("No specific aspects detected. Try using keywords like 'service', 'price', 'quality', or 'design'.")
        else:
            cols = st.columns(min(len(results), 3))
            for i, (aspect, data) in enumerate(results.items()):
                with cols[i % 3]:
                    color = "#10b981" if data['label'] == 'POSITIVE' else "#ef4444" if data['label'] == 'NEGATIVE' else "#94a3b8"
                    safe_aspect = escape(str(aspect).title())
                    safe_label = escape(str(data['label']))
                    st.markdown(f'''
                    <div class="glass-card" style="padding:1.2rem; border-top: 4px solid {color};">
                        <h5 style="margin:0; text-transform:uppercase; letter-spacing:0.05em;">{safe_aspect}</h5>
                        <p style="color:{color}; font-weight:700; font-size:1.2rem; margin:0.5rem 0;">{safe_label}</p>
                        <p style="font-size:0.8rem; color:#94a3b8;">Confidence: {data['confidence']:.1%}</p>
                    </div>
                    ''', unsafe_allow_html=True)
            st.markdown("#### Relevant Context Segments")
            for aspect, data in results.items():
                with st.expander(f"Context for {aspect.title()}"):
                    for item in data['sentences']:
                        st.write(f"• {item['sentence']}")
        st.markdown('</div>', unsafe_allow_html=True)


def render_toxicity_result_panel(result, toxicity_visualizer):
    if not result:
        return

    safety = summarize_safety_result(result)

    if result['is_toxic']:
        cls, label, sub = "res-danger", "🚨 TOXIC CONTENT DETECTED", safety["summary"]
    else:
        cls, label, sub = "res-safe", "✅ CONTENT SECURE", safety["summary"]

    st.markdown(f'''
    <div class="result-container {cls}">
        <div class="res-label">{label}</div>
        <div class="res-sub">{sub}</div>
    </div>
    ''', unsafe_allow_html=True)

    if result.get('meta') and result['meta'].get('was_translated'):
        st.error(f"🌐 Multilingual Safety: Content screened from **{result['meta']['original_language'].upper()}** for toxic patterns (Translated for analysis).")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Message Breakdown")
    render_highlights = getattr(toxicity_visualizer, "render_toxic_highlights", None) if toxicity_visualizer is not None else None
    if callable(render_highlights):
        highlighted_html = render_highlights(result['text'], result['explanation'])
        st.markdown(highlighted_html, unsafe_allow_html=True)
    else:
        st.write(result['text'])

    reasons = result['explanation'].get('reasons', [])
    if reasons:
        cols = st.columns(len(reasons))
        for i, reason in enumerate(reasons):
            cols[i % len(cols)].info(f"💡 {reason}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Forensic Analysis Breakdown")
    st.write("The models evaluate content against established safety benchmarks and behavioral linguistic patterns.")
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Risk Score", f"{safety['risk']:.1%}")
    with summary_cols[1]:
        st.metric("Threshold", f"{safety['threshold']:.1%}")
    with summary_cols[2]:
        st.metric("Risk Band", safety["band"])
    with summary_cols[3]:
        st.metric("Top Signal", safety["top_category"].replace("_", " ").title())
    st.caption("A message is flagged only when the risk score crosses the active threshold.")

    categories = result.get('categories') or {}
    max_score = max(categories.values()) if categories else 0.0
    if result['is_toxic']:
        st.error("⚠️ **Action Required:** Content violates community standards. High risk of harassment or harmful intent detected.")
        if max_score > 0.8:
            st.info("💡 **Recommendation:** Consider immediate removal and account flagging for manual review.")
        else:
            st.info("💡 **Recommendation:** Flag for community moderation or restricted visibility.")
    else:
        if max_score > 0.3:
            st.warning("ℹ️ **Observation:** Content is safe but contains polarizing or borderline linguistic markers.")
        else:
            st.success("✅ **Assessment:** Content is safe for general distribution.")

    st.markdown("---")
    st.markdown("#### Category Intensity Matrix")
    cols = st.columns(3)
    for i, (cat, score) in enumerate(categories.items()):
        with cols[i % 3]:
            color = "#ef4444" if score > 0.6 else "#f59e0b" if score > 0.3 else "#10b981"
            st.markdown(f'''
            <div class="glass-card" style="padding:1.2rem; border-left: 4px solid {color}; margin-bottom: 0.8rem;">
                <p style="margin:0; font-size:0.8rem; color:#94a3b8; text-transform:uppercase;">{cat.replace('_', ' ')}</p>
                <h3 style="margin:0.3rem 0; color:{color};">{score:.1%}</h3>
                <div style="width:100%; height:4px; background:rgba(255,255,255,0.05); border-radius:2px;">
                    <div style="width:{score*100}%; height:100%; background:{color}; box-shadow: 0 0 10px {color}44;"></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


THEME_PRESETS = {
    "Auth": {
        "key": "auth",
        "title": "TRUTH\nGUARD AI",
        "accent": "#7ce8ff",
        "accent_soft": "rgba(124, 232, 255, 0.22)",
        "secondary": "#2e7dff",
        "danger": "#ff4d6d",
    },
    "Home": {
        "key": "home",
        "title": "TRUTH\nGUARD AI",
        "accent": "#7ce8ff",
        "accent_soft": "rgba(124, 232, 255, 0.22)",
        "secondary": "#2e7dff",
        "danger": "#ff4d6d",
    },
    "Deepfake Detection": {
        "key": "deepfake",
        "title": "DEEPFAKE\nDETECTOR",
        "accent": "#8cf2ff",
        "accent_soft": "rgba(140, 242, 255, 0.24)",
        "secondary": "#4c8dff",
        "danger": "#ff6b7d",
    },
    "Fake News Detection": {
        "key": "fake_news",
        "title": "FAKE NEWS\nDETECTOR",
        "accent": "#88e9ff",
        "accent_soft": "rgba(136, 233, 255, 0.22)",
        "secondary": "#4b80ff",
        "danger": "#ff566e",
    },
    COMMUNICATION_PAGE: {
        "key": "communication",
        "title": "SENTIMENT\n& SAFETY",
        "accent": "#77f7d4",
        "accent_soft": "rgba(119, 247, 212, 0.22)",
        "secondary": "#49b0ff",
        "danger": "#ff6b7d",
    },
    "Sentiment Analysis": {
        "key": "sentiment",
        "title": "SENTIMENT\nANALYSIS",
        "accent": "#77f7d4",
        "accent_soft": "rgba(119, 247, 212, 0.22)",
        "secondary": "#49b0ff",
        "danger": "#ffc857",
    },
    "Toxicity Checker": {
        "key": "toxicity",
        "title": "TOXICITY\nDETECTOR",
        "accent": "#7fe1ff",
        "accent_soft": "rgba(127, 225, 255, 0.2)",
        "secondary": "#4189ff",
        "danger": "#ff4d6d",
    },
}


def _theme_for_page(page_name: str, authenticated: bool) -> Dict[str, str]:
    if not authenticated:
        return THEME_PRESETS["Auth"]
    return THEME_PRESETS.get(page_name, THEME_PRESETS["Home"])


@st.cache_data(ttl=3600, show_spinner=False)
def _file_to_data_uri(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".svg":
        return "data:image/svg+xml;charset=utf-8," + quote(path.read_text(encoding="utf-8"))
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _theme_motif(theme_key: str, accent: str, secondary: str, danger: str) -> str:
    motifs = {
        "home": f"""
            <rect x="170" y="420" width="230" height="230" rx="26" fill="rgba(15,40,92,0.32)" stroke="{accent}" stroke-width="3"/>
            <circle cx="285" cy="535" r="94" fill="rgba(6,16,34,0.7)" stroke="white" stroke-width="6"/>
            <circle cx="285" cy="535" r="78" fill="none" stroke="{accent}" stroke-width="3"/>
            <rect x="905" y="398" width="210" height="210" rx="34" fill="rgba(10,26,54,0.28)" stroke="{accent}" stroke-width="4"/>
            <path d="M1010 432 L1092 468 L1080 555 C1074 597 1047 629 1010 650 C973 629 946 597 940 555 L928 468 Z" fill="rgba(255,255,255,0.06)" stroke="white" stroke-width="6"/>
            <path d="M973 540 L1001 569 L1054 502" fill="none" stroke="white" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>
        """,
        "deepfake": f"""
            <circle cx="815" cy="284" r="182" fill="rgba(17,38,84,0.18)" stroke="{accent}" stroke-width="3"/>
            <circle cx="815" cy="284" r="144" fill="rgba(8,18,41,0.46)" stroke="rgba(255,255,255,0.18)" stroke-width="2"/>
            <rect x="238" y="332" width="160" height="164" rx="24" fill="rgba(11,34,74,0.32)" stroke="{accent}" stroke-width="3"/>
            <rect x="1200" y="328" width="112" height="112" rx="56" fill="rgba(124,232,255,0.14)" stroke="{accent}" stroke-width="3"/>
            <polygon points="1238,357 1238,411 1286,384" fill="white"/>
            <rect x="1196" y="124" width="104" height="56" rx="14" fill="rgba(7,18,38,0.58)" stroke="rgba(255,255,255,0.28)" stroke-width="2"/>
            <text x="1248" y="160" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="22" fill="white" font-weight="700">REC</text>
        """,
        "fake_news": f"""
            <rect x="252" y="440" width="200" height="200" rx="34" fill="rgba(8,20,45,0.68)" stroke="white" stroke-width="5"/>
            <circle cx="335" cy="522" r="104" fill="none" stroke="rgba(255,255,255,0.92)" stroke-width="10"/>
            <line x1="414" y1="604" x2="488" y2="682" stroke="rgba(255,255,255,0.92)" stroke-width="14" stroke-linecap="round"/>
            <rect x="480" y="130" width="430" height="140" rx="24" fill="rgba(18,46,92,0.24)" stroke="{accent}" stroke-width="3"/>
            <polygon points="1186,116 1216,168 1156,168" fill="{danger}" stroke="white" stroke-width="4"/>
            <polygon points="248,244 278,296 218,296" fill="{danger}" stroke="white" stroke-width="4"/>
            <polygon points="1188,506 1218,558 1158,558" fill="{danger}" stroke="white" stroke-width="4"/>
        """,
        "toxicity": f"""
            <rect x="226" y="128" width="300" height="90" rx="24" fill="rgba(13,40,74,0.26)" stroke="{accent}" stroke-width="2"/>
            <rect x="1034" y="116" width="230" height="90" rx="24" fill="rgba(80,18,42,0.22)" stroke="{danger}" stroke-width="2"/>
            <rect x="666" y="150" width="206" height="92" rx="24" fill="rgba(17,42,78,0.25)" stroke="{accent}" stroke-width="2"/>
            <polygon points="520,248 550,300 490,300" fill="{danger}" stroke="white" stroke-width="4"/>
            <polygon points="1138,224 1168,276 1108,276" fill="{danger}" stroke="white" stroke-width="4"/>
            <path d="M1080 520 L1148 550 L1138 628 C1133 668 1106 703 1080 718 C1054 703 1027 668 1022 628 L1012 550 Z" fill="rgba(255,77,109,0.18)" stroke="{danger}" stroke-width="6"/>
            <rect x="214" y="500" width="224" height="224" rx="34" fill="rgba(7,18,39,0.64)" stroke="white" stroke-width="4"/>
            <circle cx="318" cy="602" r="100" fill="none" stroke="{accent}" stroke-width="5"/>
        """,
        "sentiment": f"""
            <circle cx="330" cy="470" r="110" fill="rgba(18,52,76,0.22)" stroke="{accent}" stroke-width="4"/>
            <circle cx="1210" cy="330" r="92" fill="rgba(31,54,97,0.18)" stroke="{secondary}" stroke-width="3"/>
            <path d="M180 600 Q280 540 360 586 T540 582 T720 600 T900 580 T1090 590 T1400 546" fill="none" stroke="{accent}" stroke-width="8" stroke-linecap="round"/>
            <path d="M180 645 Q300 680 420 626 T660 632 T920 664 T1400 622" fill="none" stroke="rgba(255,255,255,0.42)" stroke-width="3" stroke-linecap="round"/>
        """,
        "auth": f"""
            <circle cx="798" cy="250" r="172" fill="rgba(14,34,76,0.2)" stroke="{accent}" stroke-width="3"/>
            <path d="M1010 404 L1096 442 L1082 535 C1078 592 1048 640 1009 667 C968 640 940 592 936 535 L922 442 Z" fill="rgba(255,255,255,0.06)" stroke="white" stroke-width="6"/>
            <path d="M968 530 L1000 562 L1054 488" fill="none" stroke="white" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>
        """,
    }
    return motifs.get(theme_key, motifs["home"])


def _build_background_svg(theme: Dict[str, str]) -> str:
    accent = theme["accent"]
    secondary = theme["secondary"]
    danger = theme["danger"]
    title_lines = theme["title"].split("\n")
    title_svg = "".join(
        f'<text x="800" y="{396 + (idx * 88)}" text-anchor="middle" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{98 if idx == 0 else 82}" '
        f'font-weight="800" fill="white" letter-spacing="3">{line}</text>'
        for idx, line in enumerate(title_lines)
    )
    motif_svg = _theme_motif(theme["key"], accent, secondary, danger)
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice">
      <defs>
        <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#020814"/>
          <stop offset="50%" stop-color="#081b3a"/>
          <stop offset="100%" stop-color="#020814"/>
        </linearGradient>
        <radialGradient id="glowA" cx="50%" cy="45%" r="50%">
          <stop offset="0%" stop-color="{accent}" stop-opacity="0.78"/>
          <stop offset="70%" stop-color="{secondary}" stop-opacity="0.16"/>
          <stop offset="100%" stop-color="#000814" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="glowB" cx="50%" cy="88%" r="34%">
          <stop offset="0%" stop-color="{accent}" stop-opacity="0.85"/>
          <stop offset="100%" stop-color="#000814" stop-opacity="0"/>
        </radialGradient>
        <pattern id="grid" width="52" height="52" patternUnits="userSpaceOnUse">
          <path d="M 52 0 L 0 0 0 52" fill="none" stroke="rgba(124,232,255,0.08)" stroke-width="1"/>
        </pattern>
      </defs>
      <rect width="1600" height="900" fill="url(#bg)"/>
      <rect width="1600" height="900" fill="url(#grid)"/>
      <circle cx="810" cy="296" r="340" fill="url(#glowA)"/>
      <ellipse cx="802" cy="780" rx="490" ry="120" fill="url(#glowB)"/>
      <ellipse cx="802" cy="784" rx="310" ry="82" fill="none" stroke="{accent}" stroke-width="8" opacity="0.95"/>
      <ellipse cx="802" cy="784" rx="410" ry="105" fill="none" stroke="rgba(255,255,255,0.36)" stroke-width="2"/>
      <ellipse cx="802" cy="784" rx="550" ry="136" fill="none" stroke="rgba(124,232,255,0.16)" stroke-width="1.5"/>
      <path d="M0 710 C170 650 245 640 390 690 S680 760 800 708 S1160 630 1325 692 S1490 724 1600 690 L1600 900 L0 900 Z" fill="rgba(7,18,39,0.92)"/>
      <g opacity="0.9">{motif_svg}</g>
      <g opacity="0.9">
        <rect x="182" y="128" width="132" height="132" rx="22" fill="rgba(12,32,68,0.24)" stroke="rgba(124,232,255,0.32)" stroke-width="2"/>
        <circle cx="248" cy="194" r="40" fill="none" stroke="rgba(255,255,255,0.78)" stroke-width="3"/>
        <rect x="1288" y="206" width="106" height="106" rx="30" fill="rgba(12,32,68,0.18)" stroke="rgba(124,232,255,0.3)" stroke-width="2"/>
        <rect x="1318" y="236" width="46" height="40" rx="10" fill="none" stroke="white" stroke-width="3"/>
        <path d="M1334 236 v-14 c0-9 7-16 16-16 s16 7 16 16 v14" fill="none" stroke="white" stroke-width="3"/>
      </g>
      {title_svg}
      <text x="800" y="692" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="600" fill="rgba(255,255,255,0.58)" letter-spacing="6">INTELLIGENT VERIFICATION • REAL-TIME ANALYSIS • SECURITY INTELLIGENCE</text>
    </svg>
    """


def _background_data_uri(theme: Dict[str, str]) -> str:
    bg_dir = Path("assets") / "backgrounds"
    theme_key = theme["key"]
    if bg_dir.exists():
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".svg"):
            candidate = bg_dir / f"{theme_key}{ext}"
            if candidate.exists():
                return _file_to_data_uri(candidate)
    return "data:image/svg+xml;charset=utf-8," + quote(_build_background_svg(theme))


def apply_dynamic_theme(page_name: str, authenticated: bool):
    theme = _theme_for_page(page_name, authenticated)
    bg_uri = _background_data_uri(theme)
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700;800&display=swap');
    :root {{
        --tg-accent: {accent};
        --tg-accent-soft: {accent_soft};
        --tg-accent-2: {secondary};
        --tg-danger: {danger};
        --tg-panel: rgba(5, 14, 29, 0.62);
        --tg-panel-strong: rgba(6, 16, 34, 0.76);
        --tg-text: #f7fbff;
        --tg-muted: #a9b7d1;
        --tg-border: rgba(124, 232, 255, 0.18);
    }}
    html, body, [class*="css"] {{
        font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
    }}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {{
        background: transparent !important;
    }}
    [data-testid="stAppViewContainer"] {{
        position: relative;
        isolation: isolate;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: -3;
        background:
            linear-gradient(180deg, rgba(2, 6, 18, 0.24), rgba(3, 8, 20, 0.72)),
            url("{bg_uri}") center center / cover no-repeat fixed;
        filter: saturate(1.05);
    }}
    [data-testid="stAppViewContainer"]::after {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: -2;
        pointer-events: none;
        background:
            radial-gradient(circle at 50% 12%, rgba(124, 232, 255, 0.14), transparent 32%),
            radial-gradient(circle at 20% 90%, rgba(76, 141, 255, 0.18), transparent 28%),
            linear-gradient(180deg, rgba(1, 5, 14, 0.18), rgba(1, 5, 14, 0.7));
    }}
    [data-testid="stHeader"] {{
        background: rgba(2, 8, 20, 0.28) !important;
        border-bottom: 1px solid rgba(124, 232, 255, 0.12);
        backdrop-filter: blur(16px);
    }}
    [data-testid="stSidebar"] {{
        background:
            linear-gradient(180deg, rgba(4, 12, 28, 0.94), rgba(6, 18, 40, 0.86)) !important;
        border-right: 1px solid rgba(124, 232, 255, 0.14);
        backdrop-filter: blur(18px);
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background: transparent;
    }}
    .block-container {{
        max-width: 1160px;
        padding-top: 1.55rem;
        padding-bottom: 4.4rem;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
    }}
    .main-header {{
        margin: 0;
        color: var(--tg-text);
        font-size: clamp(2.4rem, 4vw, 4.7rem);
        font-weight: 800;
        letter-spacing: -0.05em;
        text-shadow: 0 0 24px rgba(124, 232, 255, 0.18);
    }}
    .sub-header {{
        margin-top: 0.35rem;
        margin-bottom: 1.1rem;
        color: var(--tg-muted);
        font-size: 1.02rem;
        text-transform: uppercase;
        letter-spacing: 0.32em;
    }}
    .glass-card, .home-card, .result-container {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(180deg, rgba(7, 18, 38, 0.78), rgba(5, 14, 29, 0.72));
        border: 1px solid var(--tg-border);
        border-radius: 26px;
        padding: 1.35rem;
        box-shadow:
            0 26px 60px rgba(0, 5, 18, 0.34),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(16px);
    }}
    .glass-card::before, .home-card::before, .result-container::before {{
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(124, 232, 255, 0.12), transparent 38%, rgba(255, 255, 255, 0.03) 72%, transparent 100%);
        pointer-events: none;
    }}
    .home-card {{
        min-height: 270px;
        padding: 1.5rem;
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }}
    .home-card:hover {{
        transform: translateY(-4px);
        border-color: rgba(124, 232, 255, 0.35);
        box-shadow: 0 32px 70px rgba(0, 7, 18, 0.44), 0 0 0 1px rgba(124, 232, 255, 0.08);
    }}
    .card-badge {{
        display: inline-flex;
        padding: 0.38rem 0.78rem;
        border-radius: 999px;
        background: rgba(124, 232, 255, 0.12);
        color: var(--tg-text);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }}
    .card-icon-wrapper {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 72px;
        height: 72px;
        border-radius: 22px;
        margin-bottom: 1rem;
        font-size: 2rem;
        background: linear-gradient(135deg, rgba(124, 232, 255, 0.22), rgba(76, 141, 255, 0.2));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }}
    .card-title {{
        color: var(--tg-text);
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 0.55rem;
        letter-spacing: -0.03em;
    }}
    .card-desc {{
        color: var(--tg-muted);
        line-height: 1.7;
        font-size: 0.98rem;
        margin-bottom: 1.2rem;
    }}
    .result-container {{
        margin-bottom: 1rem;
        padding: 1.5rem 1.65rem;
    }}
    .res-safe {{
        border-color: rgba(96, 255, 186, 0.28);
        box-shadow: 0 26px 64px rgba(2, 10, 18, 0.34), inset 0 0 0 1px rgba(96, 255, 186, 0.08);
    }}
    .res-danger {{
        border-color: rgba(255, 77, 109, 0.3);
        box-shadow: 0 26px 64px rgba(2, 10, 18, 0.34), inset 0 0 0 1px rgba(255, 77, 109, 0.08);
    }}
    .res-warn {{
        border-color: rgba(255, 200, 87, 0.3);
        box-shadow: 0 26px 64px rgba(2, 10, 18, 0.34), inset 0 0 0 1px rgba(255, 200, 87, 0.08);
    }}
    .res-label {{
        color: var(--tg-text);
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.3rem;
    }}
    .res-sub {{
        color: var(--tg-muted);
        line-height: 1.6;
    }}
    [data-testid="stMetric"] {{
        background: linear-gradient(180deg, rgba(7, 18, 38, 0.78), rgba(5, 14, 29, 0.72));
        border: 1px solid var(--tg-border);
        border-radius: 22px;
        padding: 0.95rem 1rem;
        box-shadow: 0 18px 44px rgba(0, 5, 18, 0.22);
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{
        color: var(--tg-text) !important;
    }}
    .stButton > button {{
        border-radius: 999px;
        border: 1px solid rgba(124, 232, 255, 0.14);
        padding: 0.8rem 1.15rem;
        font-weight: 700;
        color: #02111f;
        background: linear-gradient(135deg, var(--tg-accent), var(--tg-accent-2));
        box-shadow: 0 16px 34px rgba(15, 54, 112, 0.32);
        transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        filter: brightness(1.04);
        box-shadow: 0 20px 42px rgba(15, 54, 112, 0.4);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background: rgba(7, 18, 38, 0.76);
        border: 1px solid rgba(124, 232, 255, 0.12);
        padding: 0.38rem;
        border-radius: 999px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 999px;
        color: var(--tg-muted);
        font-weight: 700;
        padding: 0.7rem 1rem;
        height: auto;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--tg-accent), var(--tg-accent-2));
        color: #03111f !important;
    }}
    .stTextInput input, .stTextArea textarea, div[data-baseweb="select"] > div,
    [data-testid="stFileUploader"], [data-baseweb="base-input"] {{
        background: rgba(5, 14, 29, 0.72) !important;
        color: var(--tg-text) !important;
        border: 1px solid rgba(124, 232, 255, 0.14) !important;
        border-radius: 18px !important;
    }}
    .stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label,
    .stRadio label, .stCheckbox label, .stMarkdown p, .stMarkdown li, .stCaption {{
        color: var(--tg-muted) !important;
    }}
    [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] strong {{
        color: var(--tg-text);
    }}
    .stAlert {{
        border-radius: 18px;
        border: 1px solid rgba(124, 232, 255, 0.14);
        background: rgba(7, 18, 38, 0.7);
    }}
    .stDataFrame, [data-testid="stDataFrame"] {{
        border-radius: 22px;
        overflow: hidden;
        border: 1px solid rgba(124, 232, 255, 0.12);
        background: rgba(7, 18, 38, 0.7);
    }}
    .live-news-summary {{
        position: relative;
        overflow: hidden;
        height: auto;
        min-height: 0;
        background: linear-gradient(180deg, rgba(18, 43, 80, 0.97), rgba(9, 25, 49, 0.95));
        border: 1px solid rgba(124, 232, 255, 0.28);
        border-radius: 22px;
        padding: 1.1rem 1.15rem;
        box-shadow: 0 20px 46px rgba(0, 5, 18, 0.24);
        margin-bottom: 0.25rem;
    }}
    .live-news-summary::before, .live-news-card::before {{
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: linear-gradient(135deg, rgba(124, 232, 255, 0.14), transparent 34%, rgba(255, 255, 255, 0.05) 74%, transparent 100%);
    }}
    .live-news-summary > *, .live-news-card > * {{
        position: relative;
        z-index: 1;
    }}
    .live-news-label {{
        color: #b9c9e4 !important;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }}
    .live-news-summary-text {{
        color: #f3f8ff !important;
        font-size: 0.95rem;
        line-height: 1.7;
    }}
    .live-news-card {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(180deg, rgba(22, 49, 88, 0.98), rgba(10, 28, 57, 0.96));
        border: 1px solid rgba(124, 232, 255, 0.24);
        border-radius: 24px;
        padding: 1.05rem 1.15rem 1.15rem;
        margin-bottom: 0.95rem;
        box-shadow: 0 22px 46px rgba(0, 5, 18, 0.24);
    }}
    .live-news-card-head {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-bottom: 0.7rem;
    }}
    .live-news-source {{
        color: #f7fbff !important;
        font-size: 0.96rem;
        font-weight: 700;
        line-height: 1.35;
    }}
    .live-news-time {{
        color: #c1d0e6 !important;
        font-size: 0.82rem;
        margin-top: 0.18rem;
    }}
    .live-news-title {{
        color: #f7fbff !important;
        font-size: 1.08rem;
        line-height: 1.5;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }}
    .live-news-snippet {{
        color: #e7efff !important;
        font-size: 0.94rem;
        line-height: 1.72;
        margin-bottom: 0.9rem;
    }}
    .live-news-meta {{
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
    }}
    .live-news-pill {{
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        background: rgba(124, 232, 255, 0.16);
        border: 1px solid rgba(124, 232, 255, 0.18);
        color: #f7fbff !important;
        font-size: 0.77rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }}
    .live-news-link {{
        display: inline-flex;
        align-items: center;
        padding: 0.58rem 0.95rem;
        border-radius: 999px;
        text-decoration: none;
        font-size: 0.88rem;
        font-weight: 700;
        color: #03111f !important;
        background: linear-gradient(135deg, var(--tg-accent), var(--tg-accent-2));
        box-shadow: 0 12px 28px rgba(15, 54, 112, 0.22);
    }}
    .live-news-link:hover {{
        filter: brightness(1.04);
    }}
    @keyframes tgFade {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .animate-fade {{
        animation: tgFade 420ms ease-out both;
    }}
    @media (max-width: 900px) {{
        .block-container {{
            padding-top: 1rem;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }}
        .main-header {{
            font-size: 2.35rem;
        }}
        .sub-header {{
            letter-spacing: 0.18em;
        }}
        .home-card {{
            min-height: auto;
        }}
    }}
    .admin-tools-container {{
        position: fixed;
        top: 20px;
        right: 80px;
        z-index: 10000;
        display: flex;
        gap: 0.75rem;
        align-items: center;
        background: rgba(6, 16, 34, 0.82);
        padding: 0.4rem 1rem;
        border-radius: 999px;
        border: 1px solid rgba(124, 232, 255, 0.28);
        backdrop-filter: blur(14px);
        box-shadow: 0 12px 34px rgba(0, 5, 18, 0.45);
        transition: all 220ms ease;
    }}
    .admin-tools-container:hover {{
        border-color: rgba(124, 232, 255, 0.5);
        box-shadow: 0 16px 44px rgba(0, 5, 18, 0.55);
    }}
    .admin-label {{
        color: #7ce8ff !important;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-right: 0.4rem;
        padding-right: 0.8rem;
        border-right: 1px solid rgba(124, 232, 255, 0.25);
    }}
    </style>
    """.format(
        accent=theme["accent"],
        accent_soft=theme["accent_soft"],
        secondary=theme["secondary"],
        danger=theme["danger"],
        bg_uri=bg_uri,
    )
    st.markdown(css, unsafe_allow_html=True)


def run_async(command, success_msg="Task launched!"):
    """Run command asynchronously and visibly."""
    def _run():
        try:
            import platform
            creationflags = 0
            if platform.system() == "Windows":
                creationflags = subprocess.CREATE_NEW_CONSOLE
            subprocess.Popen(command, creationflags=creationflags)
        except Exception as e:
            logger.error(f"Async task error: {e}")

    thread = threading.Thread(target=_run)
    thread.daemon = True
    thread.start()
    if hasattr(st, "toast"):
        st.toast(f"✅ {success_msg}")
    else:
        st.success(success_msg)


def render_admin_header_tools():
    """Renders the administrative control panel at the top-right corner."""
    if st.session_state.role != "admin" or not st.session_state.authenticated:
        return

    # Positioned absolutely to hover at the very top right, above the main content
    st.markdown("""
        <div style="position: absolute; top: -50px; right: 0; z-index: 10001; display: flex; align-items: center;">
            <div class="admin-tools-container">
                <div class="admin-label">🛡️ Admin</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Use a hidden/spacer column approach to align Streamlit buttons visually with the floating bar
    # We call this before the main header
    col_space, col1, col2, col3, col4, col5 = st.columns([16, 1, 1, 1, 1, 1])
    
    with col1:
        if st.button("🏥", help="System Diagnostics", key="adm_diag_top"):
            run_async([sys.executable, "diagnose.py"], "System diagnostics started!")
    with col2:
        if st.button("🚀", help="Train Deepfake", key="adm_train_df_top"):
            run_async([sys.executable, "train_deepfake.py"], "Deepfake training started!")
    with col3:
        if st.button("📊", help="Train RF", key="adm_train_rf_top"):
            run_async([sys.executable, "train_fakenews.py"], "RF training started!")
    with col4:
        if st.button("🧠", help="Train Transformer", key="adm_train_trans_top"):
            run_async([sys.executable, "train_fakenews_transformer.py"], "Transformer training started!")
    with col5:
        if st.button("🗑️", help="Clear System Cache (Recover RAM)", key="adm_clear_cache_top"):
            st.cache_resource.clear()
            st.cache_data.clear()
            # If any detectors are loaded, trigger their internal cleaner
            from fake_news_detector import FakeNewsDetector
            FakeNewsDetector().clear_cache()
            gc.collect()
            st.toast("✅ RAM cache cleared!", icon="🗑️")
    
    st.markdown('<div style="margin-top: -3.5rem;"></div>', unsafe_allow_html=True)  # Pull the header back up

# ============================================================================
# AUTHENTICATION (simplified and reliable)
# ============================================================================
HOME_PAGE_BRAND = "AI-Based Fake Content Analysis Platform"
PROJECT_BRAND = "TruthGuard AI"

apply_dynamic_theme(st.session_state.page, st.session_state.authenticated)

if not st.session_state.authenticated:
    st.markdown("""
    <div class="glass-card animate-fade" style="padding:2.2rem 2rem; margin:1rem auto 1.5rem; max-width:960px; text-align:center;">
        <div class="card-badge" style="margin-bottom:1rem;">AI Security Command Center</div>
        <h1 class="main-header" style="margin-bottom:0.8rem; font-size:2.45rem; line-height:1.2;">AI-Based Fake Content Analysis Platform</h1>
        <p style="color:#c5d5ea; font-size:1.08rem; max-width:720px; margin:0 auto; line-height:1.8;">
            Real-time deepfake analysis, fake news verification, toxicity defense, and sentiment intelligence in one cinematic workspace.
        </p>
    </div>
    """, unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", width="stretch")
            if submitted:
                cooldown_remaining = _remaining_login_cooldown_seconds()
                if cooldown_remaining > 0:
                    st.error(
                        f"Too many login attempts. Please wait about {cooldown_remaining} seconds and try again."
                    )
                else:
                    users = load_users()
                    stored_username, stored_user = _resolve_user_record(users, username)
                    normalized_username = _normalize_username(username)
                    if (
                        normalized_username.casefold() == ADMIN_USER.casefold()
                        and ADMIN_PASS
                        and hmac.compare_digest(password, ADMIN_PASS)
                    ):
                        _reset_login_rate_limit()
                        st.session_state.authenticated = True
                        st.session_state.username = ADMIN_USER
                        st.session_state.role = "admin"
                        st.rerun()
                    elif stored_username and stored_user and verify_password(password, stored_user.get("password", "")):
                        if len(stored_user.get("password", "")) == 64:
                            users[stored_username]["password"] = get_password_hash(password)
                            save_users(users)
                        _reset_login_rate_limit()
                        st.session_state.authenticated = True
                        st.session_state.username = stored_username
                        st.session_state.role = "admin" if stored_user.get("role") == "admin" else "user"
                        st.rerun()
                    else:
                        _record_failed_login_attempt()
                        retry_hint = _remaining_login_cooldown_seconds()
                        if retry_hint > 0:
                            st.error(
                                f"Invalid credentials. Login is temporarily paused for {retry_hint} seconds."
                            )
                        else:
                            st.error("Invalid credentials.")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("New username")
            new_password = st.text_input("New password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            create_admin = st.checkbox("Create as admin")
            admin_key = st.text_input("Admin registration key", type="password", disabled=not create_admin)
            registered = st.form_submit_button("Create account", width="stretch")

            if registered:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif create_admin and not ADMIN_REG_KEY:
                    st.error("Admin account creation is disabled until ADMIN_REGISTRATION_KEY is configured.")
                elif create_admin and not hmac.compare_digest(admin_key or "", ADMIN_REG_KEY):
                    st.error("Invalid admin registration key.")
                else:
                    role = "admin" if create_admin else "user"
                    ok, message = register_user(new_username, new_password, role=role)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
    st.stop()

# ============================================================================
# LOAD DETECTORS (only after login - keeps startup fast)
# ============================================================================
deepfake_detector = None
fake_news_detector = None
sentiment_analyzer = None
toxicity_detector = None
realtime_verifier = None
DEEPFAKE_AVAILABLE = False
FAKE_NEWS_AVAILABLE = False
SENTIMENT_AVAILABLE = False
TOXICITY_AVAILABLE = False
REALTIME_AVAILABLE = False
ASPECT_AVAILABLE = False
VIZ_AVAILABLE = False
AspectSentimentAnalyzer = None
SentimentVisualizer = None
ToxicityVisualizer = None

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    sidebar_brand = HOME_PAGE_BRAND if st.session_state.page == "Home" else PROJECT_BRAND
    st.markdown(
        '<div style="text-align:center;padding:1rem 0;background:linear-gradient(135deg,#4a90e2,#3b82f6);'
        f'border-radius:12px;color:white;font-weight:700;font-size:1rem;line-height:1.35;padding:1rem 0.8rem;">{sidebar_brand}</div>',
        unsafe_allow_html=True,
    )

    safe_username = escape(str(st.session_state.username or "Unknown"))
    safe_role = escape(str((st.session_state.role or "user")).upper())
    st.markdown(f"""
    <div style="background:rgba(30,41,59,0.8); padding:1rem; border-radius:12px; margin-bottom:1rem;">
        <h4 style="margin:0;">👤 {safe_username}</h4>
        <p style="margin:0; color:#94a3b8;">Role: {safe_role}</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Logout", width="stretch"):
        release_all_feature_resources()
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.page = "Home"
        st.rerun()

    if st.session_state.role == "admin":
        if not Path(".env").exists():
            st.sidebar.warning("⚠️ `.env` file missing. Secrets not loaded securely.")
        else:
            st.sidebar.success("🔐 Environment variables loaded.")

    st.markdown("---")
    st.markdown("### 🧭 Navigation")

    navigation_items = [
        ("🏠 Home", "Home"),
        ("📸 Deepfake Analysis", "Deepfake Detection"),
        ("📰 Fake News Verifier", "Fake News Detection"),
        ("🧠 Sentiment & Safety", COMMUNICATION_PAGE),
    ]

    for label, target in navigation_items:
        is_active = st.session_state.page == target
        if st.button(label, key=f"nav_{target}", width="stretch",
                     type="primary" if is_active else "secondary"):
            if st.session_state.page != target:
                release_feature_resources(st.session_state.page)
                st.session_state.page = target
                # Clear all loaded flags on page change
                for key in list(st.session_state.keys()):
                    if key.startswith("_loaded_"):
                        st.session_state[key] = False
                st.rerun()

# ============================================================================
# MAIN HEADER & ADMIN TOOLS
# ============================================================================

render_admin_header_tools()

main_header = HOME_PAGE_BRAND if st.session_state.page == "Home" else PROJECT_BRAND
main_header_style = "font-size:2.45rem; line-height:1.2;" if st.session_state.page == "Home" else ""
st.markdown(
    f'<h1 class="main-header" style="{main_header_style}">🛡️ {main_header}</h1>',
    unsafe_allow_html=True,
)
st.markdown('<p class="sub-header">Advanced Media Authenticity & Analysis</p>', unsafe_allow_html=True)

if st.session_state.page == "Home":

    st.markdown("""
    <div class="animate-fade" style="text-align:center; padding:3rem 1rem; margin-bottom:1rem;">
        <h2 style="font-size:3rem; font-weight:800; margin-bottom:1.5rem; letter-spacing:-0.03em;">
            Secure Your Digital World <br>
            <span style="background: linear-gradient(135deg, #7ce8ff, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                With Advanced AI Intelligence
            </span>
        </h2>
        <p style="color:#94a3b8; font-size:1.25rem; max-width:800px; margin:0 auto; line-height:1.6;">
            The AI-Based Fake Content Analysis Platform utilizes state-of-the-art ensemble models to verify the authenticity of media, detect deception, and analyze communication patterns in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)
















    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="home-card animate-fade">
            <div class="card-badge">Computer Vision</div>
            <div class="card-icon-wrapper">📸</div>
            <div class="card-title">Deepfake Detection</div>
            <div class="card-desc">Advanced frame-by-frame analysis and pixel-level scrutiny to identify manipulated media.</div>
            <div style="display:flex; gap:1.5rem; margin-bottom:1rem;">
                <div><div style="color:white; font-weight:700;">99.2%</div><div style="font-size:0.8rem; color:#94a3b8;">Accuracy</div></div>
                <div><div style="color:white; font-weight:700;">Ensemble</div><div style="font-size:0.8rem; color:#94a3b8;">Logic</div></div>
                <div><div style="color:white; font-weight:700;">GPU</div><div style="font-size:0.8rem; color:#94a3b8;">Powered</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Deepfake Analyzer", key="btn_df", width="stretch"):
            st.session_state.page = "Deepfake Detection"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="home-card animate-fade">
            <div class="card-badge">Natural Language</div>
            <div class="card-icon-wrapper">📰</div>
            <div class="card-title">Fake News Detection</div>
            <div class="card-desc">Linguistic pattern analysis and factual consistency checking across multiple news sources.</div>
            <div style="display:flex; gap:1.5rem; margin-bottom:1rem;">
                <div><div style="color:white; font-weight:700;">94.5%</div><div style="font-size:0.8rem; color:#94a3b8;">Accuracy</div></div>
                <div><div style="color:white; font-weight:700;">NLP</div><div style="font-size:0.8rem; color:#94a3b8;">Analysis</div></div>
                <div><div style="color:white; font-weight:700;">Global</div><div style="font-size:0.8rem; color:#94a3b8;">Database</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Verifier", key="btn_fn", width="stretch"):
            st.session_state.page = "Fake News Detection"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="home-card animate-fade">
        <div class="card-badge">Communication Intelligence</div>
        <div class="card-icon-wrapper">🧠</div>
        <div class="card-title">Sentiment & Safety</div>
        <div class="card-desc">Analyze emotional tone and harmful-language risk together, with batch sentiment processing and aspect-level breakdowns in the same workspace.</div>
        <div style="display:flex; gap:1.5rem; margin-bottom:1rem;">
            <div><div style="color:white; font-weight:700;">Unified</div><div style="font-size:0.8rem; color:#94a3b8;">Single Scan</div></div>
            <div><div style="color:white; font-weight:700;">Batch</div><div style="font-size:0.8rem; color:#94a3b8;">Processing</div></div>
            <div><div style="color:white; font-weight:700;">Aspect</div><div style="font-size:0.8rem; color:#94a3b8;">Insights</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Communication Analysis", key="btn_comm", width="stretch"):
        st.session_state.page = COMMUNICATION_PAGE
        st.rerun()

# ============================================================================
# DEEPFAKE DETECTION
# ============================================================================

elif st.session_state.page == "Deepfake Detection":
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("← Back"):
            release_feature_resources("Deepfake Detection")
            st.session_state.page = "Home"
            st.rerun()

    st.markdown('<h1 class="animate-fade">📸 Deepfake Detection Analysis</h1>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card animate-fade">', unsafe_allow_html=True)
    deepfake_model_info = get_available_deepfake_models()
    available_models = deepfake_model_info['model_names']
    if not available_models:
        st.warning(
            "No deepfake models are available right now. Add a trained `.h5` or `.weights.h5` model "
            "under `models/` and reopen this page."
        )
        st.stop()
    preferred_model = deepfake_model_info.get('preferred_model_name')
    recommended_threshold = float(deepfake_model_info.get('recommended_threshold') or 0.5)
    recommended_threshold = min(0.9, max(0.1, round(recommended_threshold / 0.05) * 0.05))
    model_options = available_models + [DEEPFAKE_ENSEMBLE_LABEL]

    if st.session_state.get("df_model") == "Ensemble (Best Performance)":
        st.session_state.df_model = DEEPFAKE_ENSEMBLE_LABEL
    if st.session_state.get("df_model") not in model_options:
        st.session_state.df_model = preferred_model if preferred_model in available_models else DEEPFAKE_ENSEMBLE_LABEL

    model_selection_mode = _segmented_control(
        "Model Selection Mode",
        ["Single Model", "Full Ensemble", "Custom Ensemble"],
        default=st.session_state.get("df_selection_mode", "Single Model"),
        key="df_selection_mode"
    )

    requested_models = None
    selected_model = None

    if model_selection_mode == "Single Model":
        selected_model = st.selectbox(
            "🎯 Select Primary Model",
            available_models,
            index=available_models.index(preferred_model) if preferred_model in available_models else 0,
            key="df_single_model_select"
        )
        requested_models = [selected_model]
    elif model_selection_mode == "Full Ensemble":
        st.info("🚀 Running an ensemble analysis for maximum accuracy.")
        requested_models = available_models
        selected_model = DEEPFAKE_ENSEMBLE_LABEL
    else:
        requested_models = st.multiselect(
            "🧪 Select Models to Ensemble",
            available_models,
            default=[preferred_model] if preferred_model else available_models[:1],
            key="df_custom_ensemble_select"
        )
        selected_model = f"Custom Ensemble ({len(requested_models)} models)"
        if not requested_models:
            st.warning("⚠️ Please select at least one model for analysis.")

    hf_model_names = set(deepfake_model_info.get("hf_models", []))
    selection_uses_hf = any(model_name in hf_model_names for model_name in (requested_models or []))

    if "df_threshold" not in st.session_state:
        st.session_state.df_threshold = recommended_threshold

    col1, col2 = st.columns(2)
    with col1:
        thresh = st.slider("Detection Sensitivity (Threshold)", 0.1, 0.9, st.session_state.df_threshold, 0.05, key="df_threshold_slider")
    with col2:
        explain_ai = st.toggle("🔍 Enable Explainable AI (Grad-CAM)", value=True,
                               help="Generate heatmaps to visualize detected manipulations")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card animate-fade">', unsafe_allow_html=True)
    df_mode = _segmented_control(
        "Analysis Mode",
        ["📷 Image Analysis", "🎥 Video Analysis"],
        default=st.session_state.get("df_analysis_mode", "📷 Image Analysis"),
        key="df_analysis_mode",
    )

    if df_mode == "📷 Image Analysis":
        img_file = st.file_uploader(
            "Upload Image for Scrutiny",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="df_image_upload",
        )
        if img_file:
            try:
                img = _load_uploaded_image(img_file, max_size=(800, 800), label="Deepfake image upload")
            except Exception as e:
                img = None
                st.error(f"Could not process the uploaded image: {e}")
            if img is not None:
                st.image(img, caption="Target Image", width="stretch")

            if img is not None and st.button("🚀 Run Image Deepfake Analysis", width="stretch", key="btn_run_df_image"):
                with st.spinner("Analyzing frames and pixel consistency..."):
                    try:
                        _clear_feature_session_state("Deepfake Detection")
                        st.session_state.df_preview_image = _serialize_display_image(
                            img.copy(),
                            max_size=(1024, 1024),
                            prefer_lossless=False,
                        )
                        arr = np.array(img)
                        if len(arr.shape) == 2:
                            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                        elif arr.shape[2] == 4:
                            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                        else:
                            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                        if selected_model == DEEPFAKE_ENSEMBLE_LABEL or model_selection_mode == "Full Ensemble":
                            res = run_isolated_deepfake_image_analysis(
                                arr,
                                threshold=thresh,
                                return_heatmap=explain_ai,
                                requested_models=requested_models,
                                enable_hf_models=selection_uses_hf,
                                timeout_seconds=1800 if selection_uses_hf else 600,
                            )
                        elif model_selection_mode == "Custom Ensemble":
                            res = run_isolated_deepfake_image_analysis(
                                arr,
                                threshold=thresh,
                                return_heatmap=explain_ai,
                                requested_models=requested_models,
                                enable_hf_models=selection_uses_hf,
                                timeout_seconds=1800 if selection_uses_hf else 600,
                            )
                        else:
                            res = run_isolated_deepfake_image_analysis(
                                arr,
                                threshold=thresh,
                                return_heatmap=explain_ai,
                                model_name=selected_model,
                                enable_hf_models=selection_uses_hf,
                                timeout_seconds=1800 if selection_uses_hf else 600,
                            )

                        if not isinstance(res, dict):
                            raise RuntimeError("Deepfake detector returned an invalid response.")

                        res['threshold_used'] = thresh
                        res['selected_model'] = selected_model
                        res['model_weights_snapshot'] = dict(deepfake_model_info.get('weights', {}))
                        st.session_state.df_result = _compact_deepfake_result(res)
                    except Exception as e:
                        logger.exception("Image deepfake analysis failed")
                        st.session_state.df_result = None
                        st.error(f"Image analysis failed: {e}")
                        st.stop()
                    st.rerun()
    else:
        vid_file = st.file_uploader("Upload Video for Scrutiny", type=['mp4', 'avi', 'mov'], key="df_video_upload")
        if vid_file:
            try:
                _require_upload_within_limit(vid_file, MAX_VIDEO_UPLOAD_BYTES, "Deepfake video upload")
            except Exception as e:
                st.error(str(e))
                vid_file = None

        if vid_file:
            st.video(vid_file)

            if st.button("🚀 Run Video Deepfake Analysis", width="stretch", key="btn_run_df_video"):
                path = None
                try:
                    with st.spinner("Performing temporal and spatial analysis..."):
                        path = _write_uploaded_video_tempfile(vid_file)
                        _clear_feature_session_state("Deepfake Detection")
                        st.session_state.df_preview_image = None
                        if selected_model == DEEPFAKE_ENSEMBLE_LABEL or model_selection_mode == "Full Ensemble":
                            res = run_isolated_deepfake_video_analysis(
                                path,
                                threshold=thresh,
                                return_heatmap=explain_ai,
                                requested_models=requested_models,
                                enable_hf_models=selection_uses_hf,
                                timeout_seconds=2400 if selection_uses_hf else 1200,
                            )
                        elif model_selection_mode == "Custom Ensemble":
                            res = run_isolated_deepfake_video_analysis(
                                path,
                                threshold=thresh,
                                return_heatmap=explain_ai,
                                requested_models=requested_models,
                                enable_hf_models=selection_uses_hf,
                                timeout_seconds=2400 if selection_uses_hf else 1200,
                            )
                        else:
                            res = run_isolated_deepfake_video_analysis(
                                path,
                                threshold=thresh,
                                return_heatmap=explain_ai,
                                model_name=selected_model,
                                enable_hf_models=selection_uses_hf,
                                timeout_seconds=2400 if selection_uses_hf else 1200,
                            )

                        if not isinstance(res, dict):
                            raise RuntimeError("Deepfake detector returned an invalid response.")

                        res['threshold_used'] = thresh
                        res['selected_model'] = selected_model
                        res['model_weights_snapshot'] = dict(deepfake_model_info.get('weights', {}))
                        st.session_state.df_result = _compact_deepfake_result(res)
                    st.rerun()
                finally:
                    _safe_delete_artifact(path)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.df_result:
        r = st.session_state.df_result
        if r.get('error'):
            st.error(f"Deepfake analysis failed: {r['error']}")
            st.stop()
        if 'is_deepfake' not in r or 'confidence' not in r:
            st.error("Deepfake analysis returned an incomplete result.")
            st.stop()

        is_df, conf = r['is_deepfake'], r['confidence']
        active_threshold = r.get('threshold_used', st.session_state.get('df_threshold', 0.5))
        model_weights_snapshot = r.get('model_weights_snapshot', {})
        source_preview = st.session_state.get('df_preview_image')
        face_detection_mode = r.get('face_detection_mode', 'unknown')
        detected_faces = int(r.get('face_count', 0))
        analyzed_regions = int(r.get('analyzed_region_count', detected_faces))

        st.markdown('<div class="glass-card animate-fade">', unsafe_allow_html=True)
        st.markdown(f"### 🛡️ Deepfake Forensic Report - ID: {hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8].upper()}")

        is_video_res = 'deepfake_ratio' in r
        deepfake_likelihood = float(np.clip(r.get('ensemble_score', 0.0), 0.0, 1.0))
        authentic_likelihood = 1.0 - deepfake_likelihood
        if is_df:
            verdict_cls = "res-danger"
            verdict_label = "🚨 LIKELY DEEPFAKE"
        else:
            verdict_cls = "res-safe"
            verdict_label = "✅ LIKELY AUTHENTIC"
        verdict_sub = f"Estimated deepfake likelihood: {deepfake_likelihood:.1%}"
        if r.get('status') == 'FORENSIC_ONLY':
            verdict_sub += " (Forensic-only fallback mode)"

        forensic_view = _segmented_control(
            "Report View",
            ["📊 Overview", "🔬 Visual Forensics", "📡 Signal Analysis", "📦 Ensemble Logic"],
            default=st.session_state.get("df_report_view", "📊 Overview"),
            key="df_report_view",
        )

        if forensic_view == "📊 Overview":
            col_m1, col_m2 = st.columns([2, 1])
            with col_m1:
                st.markdown(f'''
                <div class="result-container {verdict_cls}">
                    <div class="res-label">{verdict_label}</div>
                    <div class="res-sub">{verdict_sub}</div>
                </div>
                ''', unsafe_allow_html=True)
                st.markdown("#### Primary Verdict")
                st.markdown(r.get('message', 'Analysis complete.'))
                st.markdown(f"**Deepfake Likelihood:** `{deepfake_likelihood:.1%}`")
                st.markdown(f"**Authenticity Likelihood:** `{authentic_likelihood:.1%}`")
                st.markdown(f"**Analysis Confidence:** `{conf:.1f}%`")
                st.progress(deepfake_likelihood, text=f"Estimated Deepfake Likelihood: {deepfake_likelihood*100:.1f}%")
                if face_detection_mode == "fallback_crop":
                    st.warning(
                        f"No faces were confidently detected. Analysis used {analyzed_regions} centered fallback crop(s) instead."
                    )
                if is_video_res:
                    st.info(f"📹 Analyzed {r['total_frames_analyzed']} frames. Manipulation detected in {r['deepfake_ratio']*100:.1f}% of content.")
                else:
                    st.info(f"🌍 Forensic check completed on {r.get('face_count', 0)} detected facial regions using high-sensitivity crops.")
            with col_m2:
                fig = create_gauge(deepfake_likelihood, threshold=active_threshold, title="Deepfake Likelihood", height=200)
                _render_plotly_chart(fig)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Detected Faces", detected_faces)
            with col2:
                st.metric("Model Consistency", f"{r.get('consistency', 0)*100:.1f}%")
            with col3:
                if is_video_res:
                    flicker = r.get('flicker_score', 0)
                    st.metric("Temporal Stability",
                              "Low Stability" if flicker > 0.15 else "Stable" if flicker < 0.05 else "Moderate",
                              f"-{flicker:.3f}", delta_color="inverse")
                else:
                    st.metric("Artifact Density", f"{r.get('ela_score', 0):.4f}")
            if not is_video_res and analyzed_regions != detected_faces:
                st.caption(
                    f"Analysis used {analyzed_regions} region(s); the detected face count excludes fallback crops and faces skipped for performance."
                )

        elif forensic_view == "🔬 Visual Forensics":
            st.markdown("#### Explainability & Localization")
            st.markdown("This section maps the specific pixels and regions that the AI identifies as non-authentic.")
            if is_video_res:
                st.info("📊 This forensic map is generated from the **most suspicious representative frame** found in the video stream.")

            col_i1, col_i2 = st.columns(2)
            with col_i1:
                if not is_video_res:
                    if source_preview is not None:
                        st.image(source_preview, caption="Forensic Source", width="stretch")
                    else:
                        st.info("Original image preview is unavailable for this stored result.")
                else:
                    st.markdown('''
                    <div style="background:rgba(255,255,255,0.05); padding:2rem; border-radius:12px; text-align:center;">
                        <p style="margin:0; font-size:3rem;">📹</p>
                        <p style="margin:0; color:#94a3b8;">Original video frame</p>
                    </div>
                    ''', unsafe_allow_html=True)
            with col_i2:
                if r.get('heatmap') is not None:
                    st.image(r['heatmap'], caption="Anomalous Area Heatmap (Grad-CAM)", width="stretch")
                else:
                    st.warning("Heatmap could not be generated for this analysis result.")

            if 'target_face_bbox' in r:
                st.caption(f"Showing analysis for identified primary target at coordinates: {r['target_face_bbox']}")

        elif forensic_view == "📡 Signal Analysis":
            st.markdown("#### Digital Forensic Signals")
            st.markdown("Unlike AI patterns, these signals detect mathematical and physical anomalies in the image data.")
            sig_col1, sig_col2 = st.columns(2)

            with sig_col1:
                st.markdown("**1. Error Level Analysis (ELA)**")
                if 'ela_image' in r and r['ela_image'] is not None:
                    st.image(r['ela_image'], caption="Compression Level Forensic Map", width="stretch")
                    e_score = r.get('ela_score', 0)
                    st.progress(e_score, text=f"Compression Inconsistency: {e_score:.4f}")
                else:
                    st.error("ELA generation failed.")

            with sig_col2:
                st.markdown("**2. Frequency Domain Analysis (FFT)**")
                if 'fft_image' in r and r['fft_image'] is not None:
                    st.image(r['fft_image'], caption="Frequency Magnitude Spectrum", width="stretch")
                    f_score = r.get('fft_score', 0)
                    st.progress(f_score, text=f"Upsampling Pattern Density: {f_score:.4f}")
                    if f_score > 0.5:
                        st.warning("⚠️ High-frequency periodic artifacts detected (Checkerboard pattern typical of GANs).")
                else:
                    st.error("FFT analysis data missing.")

            if is_video_res:
                st.markdown("---")
                st.markdown("#### 📈 Temporal Consistency Timeline")
                temp_scores = r.get('temporal_scores', [])
                if temp_scores:
                    fig = px.line(x=range(len(temp_scores)), y=temp_scores,
                                  title="Deepfake Score Variation Over Time",
                                  labels={'x': 'Sampled Frames', 'y': 'Deception Score (0-1)'},
                                  template="plotly_dark")
                    fig.add_hline(y=active_threshold, line_dash="dash", line_color="red",
                                  annotation_text=f"Threshold ({active_threshold})")
                    fig.update_traces(line_color='#3b82f6', line_width=2)
                    _render_plotly_chart(fig)

        else:
            st.markdown("#### Probabilistic Model Breakdown")
            st.markdown("TruthGuard AI uses a weighted consensus from multiple neural architectures.")

            if 'model_scores' in r and r['model_scores']:
                best_model, best_score = None, -1
                for name, score in r['model_scores'].items():
                    if score > best_score:
                        best_model, best_score = name, score
                    m_color = "#ef4444" if score > active_threshold else "#10b981"
                    st.markdown(f"**{name}** (Weight: {model_weights_snapshot.get(name, 1.0):.1f})")
                    col_b1, col_b2 = st.columns([4, 1])
                    with col_b1:
                        st.progress(score)
                    with col_b2:
                        st.markdown(f"<span style='color:{m_color}; font-weight:700;'>{score:.4f}</span>", unsafe_allow_html=True)
                st.success(f"📌 **Consensus Leader:** `{best_model}` had the highest activation for this sample.")
            else:
                st.warning("Per-model tracking not available for this legacy video result.")

        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# FAKE NEWS DETECTION
# ============================================================================

elif st.session_state.page == "Fake News Detection":
    gc.collect() # Pre-emptive cleanup for 8GB RAM
    fake_news_detector, FAKE_NEWS_AVAILABLE = load_component(get_fake_news_detector, "Fake news detector")
    realtime_verifier = None
    REALTIME_AVAILABLE = True
    if not FAKE_NEWS_AVAILABLE:
        st.error("Fake news detection module not available")
        st.stop()

    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("← Back"):
            release_feature_resources("Fake News Detection")
            st.session_state.page = "Home"
            st.rerun()

    st.markdown('<h1 class="animate-fade">📰 Fake News Verification</h1>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card animate-fade">', unsafe_allow_html=True)
    available_models = fake_news_detector.get_available_models()
    model_info = fake_news_detector.get_model_info()

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fn_selection_mode = _segmented_control(
            "Detection Mode",
            ["Single Model", "Full Ensemble", "Custom Selection"],
            default=st.session_state.get("fn_selection_mode", "Single Model"),
            key="fn_selection_mode"
        )
        
        use_ensemble = (fn_selection_mode == "Full Ensemble")
        requested_models = None
        
        if fn_selection_mode == "Single Model":
            model_names = [f"{m['name']} ({m['type'].replace('_', ' ').title()})" for m in available_models]
            if not model_names:
                st.warning("No localized models found. Using cloud fallback.")
                selected_model_idx = 0
                requested_models = None
            else:
                default_model_idx = next(
                    (i for i, model in enumerate(available_models) if model.get("type") == "transformer"),
                    next((i for i, model in enumerate(available_models) if model.get("type") == "random_forest"), 0),
                )
                selected_model_str = st.selectbox("🎯 Active Detection Engine", model_names, index=default_model_idx)
                selected_model_idx = model_names.index(selected_model_str)
                requested_models = [available_models[selected_model_idx]['path']]
                if available_models and available_models[selected_model_idx].get("type") == "transformer":
                    st.caption("Transformer models are the preferred default here because they are more reliable for direct claim checks.")
                else:
                    st.caption("Traditional models are the faster default for this page.")
        elif fn_selection_mode == "Full Ensemble":
            st.info("🚀 **Ensemble Active**: Running parallel analysis across multiple detectors.")
            selected_model_idx = -1
            requested_models = None # Detector will use all defaults
        else:
            # Custom Selection
            local_options = [f"{m['name']} (Local)" for m in available_models]
            hf_options = fake_news_detector.get_huggingface_models()
            all_options = local_options + hf_options
            
            selected_names = st.multiselect(
                "🧪 Select Models to Include",
                all_options,
                default=[local_options[0]] if local_options else [],
                key="fn_custom_models"
            )
            
            # Map back to internal paths/names
            requested_models = []
            for name in selected_names:
                if name in hf_options:
                    requested_models.append(name)
                else:
                    # Find matching local model
                    idx = local_options.index(name)
                    requested_models.append(available_models[idx]['path'])
            
            if not requested_models:
                st.warning("⚠️ Please select at least one model for analysis.")
            else:
                st.success(f"Selected {len(requested_models)} models for granular analysis.")
    with col_m2:
        st.markdown(f'''
        <div style="padding:0.5rem; background:rgba(255,255,255,0.05); border-radius:8px;">
            <p style="margin:0; font-size:0.8rem; color:#94a3b8;">ACTIVE ENGINE</p>
            <p style="margin:0; font-weight:600; color:#3b82f6;">{model_info['model_type'].upper()}</p>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card animate-fade">', unsafe_allow_html=True)
    methods = ["📝 Direct Verification", "📁 Bulk Processing"]
    if REALTIME_AVAILABLE:
        methods.append("📡 Real-time Verification")

    method = _segmented_control("Analysis Method", methods, default="📝 Direct Verification")

    if 'last_fn_method' not in st.session_state:
        st.session_state.last_fn_method = method

    if st.session_state.last_fn_method != method:
        st.session_state.last_fn_method = method
        st.session_state.fn_result = None
        st.session_state.fn_verification_result = None
        st.rerun()

    if method == "📡 Real-time Verification":
        realtime_verifier, REALTIME_AVAILABLE = load_component(get_realtime_verifier, "Realtime verifier")
        if not REALTIME_AVAILABLE or realtime_verifier is None:
            st.error("📡 Real-time verification is currently unavailable. Please install dependencies: `pip install langdetect googletrans==4.0.0rc1 lxml beautifulsoup4`")
            st.stop()
        st.markdown("### 📡 Real-time Consensus Check")
        st.info("This tool verifies claims against live news feeds to check for factual consistency and cross-source reporting.")

        claim_text = st.text_area("Enter Claim or Headline", height=100, placeholder="e.g., Global trade deal signed between X and Y...")

        if st.button("🌐 Verify Against Live News", width="stretch") and claim_text.strip():
            with st.spinner("Scanning global news cycles..."):
                if realtime_verifier is not None:
                    res = realtime_verifier.verify_claim(claim_text)
                    st.session_state.fn_verification_result = res
                else:
                    st.error("📡 Real-time engine is not initialized. Please refresh or check logs.")

        if st.session_state.fn_verification_result:
            res = st.session_state.fn_verification_result
            st.markdown("---")
            if res['status'] == 'NO_RESULTS':
                st.warning("No significant matching news reports found for this claim in the current cycle.")
            else:
                consensus = res['consensus_score']
                st.markdown(f"#### Consensus Score: {consensus:.1%}")
                st.progress(consensus)
                if consensus > 0.6:
                    st.success("✅ HIGH CONSENSUS: Multiple reputable sources are reporting similar information.")
                elif consensus > 0.3:
                    st.warning("⚠️ PARTIAL MATCH: Some news matches were found, but details may differ.")
                else:
                    st.error("🚨 NO CONSENSUS: This claim is not being reflected in mainstream news reports.")
            render_live_news_comparison(res, claim_text)

    elif method == "📝 Direct Verification":
        if st.session_state.clear_fn:
            st.session_state.clear_fn = False
            st.session_state.fn_text = ""
            st.session_state.fn_text_input = ""

        input_mode = _segmented_control(
            "Content Source",
            ["Article Text", "Article Image"],
            default=st.session_state.get("fn_input_mode", "Article Text"),
            key="fn_input_mode",
        )

        text = ""
        uploaded_article_image = None
        article_image = None
        image_reader_status = None

        if input_mode == "Article Text":
            text = st.text_area(
                "News Article Content",
                height=200,
                key="fn_text_input",
                placeholder="Paste source text for linguistic and factual consistency analysis...",
            )
        else:
            image_reader_status = fake_news_detector.get_image_reader_status()
            if image_reader_status.get("available"):
                st.caption("Image reader is active. Upload a screenshot or scan of a news article to extract the text before verification.")
            else:
                st.warning(
                    f"Image reader unavailable: {image_reader_status.get('error', 'No OCR backend detected.')}"
                )

            uploader_key = f"fn_image_upload_{st.session_state.get('fn_image_upload_nonce', 0)}"
            uploaded_article_image = st.file_uploader(
                "Upload Article Image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif', 'tiff'],
                key=uploader_key,
                help="Supports screenshots, scanned print articles, and social media news cards.",
            )

            if uploaded_article_image is not None:
                try:
                    article_image = _load_uploaded_image(
                        uploaded_article_image,
                        max_size=(1400, 1400),
                        label="Article image upload",
                    )
                    st.image(article_image, caption="Article image", width="stretch")
                except Exception as e:
                    article_image = None
                    st.error(f"Could not open the uploaded image: {e}")

        st.session_state.fn_check_realtime = True
        check_rt = True

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("🔍 Run Forensic Analysis", key="btn_run_fn", width="stretch"):
                if input_mode == "Article Text" and not text.strip():
                    st.warning("Please provide article text for verification.")
                elif input_mode == "Article Image" and article_image is None:
                    if image_reader_status and not image_reader_status.get("available"):
                        st.warning(image_reader_status.get("error", "Image reader is unavailable."))
                    else:
                        st.warning("Please upload an article image for verification.")
                else:
                    with st.spinner("Analyzing semantic structures..."):
                        if fn_selection_mode == "Full Ensemble":
                            if not fake_news_detector.ensure_model_loaded(use_ensemble=True):
                                st.warning("Ensemble initialization partially failed. Proceeding with available models.")
                        elif fn_selection_mode == "Custom Selection":
                            if not fake_news_detector.ensure_model_loaded(requested_models=requested_models):
                                st.warning("Model loading partially failed. Some requested models might be skipped.")
                        elif available_models:
                            selected_path = available_models[selected_model_idx]['path']
                            current_path = st.session_state.get('active_fn_model_path')
                            if selected_path != current_path:
                                if fake_news_detector.ensure_model_loaded(selected_path):
                                    st.session_state.active_fn_model_path = selected_path
                                else:
                                    st.warning("Selected model could not be loaded. Falling back to lightweight verification.")
                                    st.session_state.active_fn_model_path = None

                        try:
                            if input_mode == "Article Image":
                                label, conf, click_score, meta = fake_news_detector.predict_from_image(
                                    article_image,
                                    image_name=getattr(uploaded_article_image, "name", None),
                                    check_realtime=check_rt,
                                    use_ensemble=use_ensemble,
                                    requested_models=requested_models if fn_selection_mode != "Single Model" else None,
                                )
                                resolved_text = meta.get("ocr", {}).get("extracted_text", "")
                                source_name = meta.get("ocr", {}).get("image_name") or getattr(uploaded_article_image, "name", None)
                            else:
                                label, conf, click_score, meta = fake_news_detector.predict(
                                    text,
                                    check_realtime=check_rt,
                                    use_ensemble=use_ensemble,
                                    requested_models=requested_models if fn_selection_mode != "Single Model" else None,
                                )
                                resolved_text = text
                                source_name = None
                        except ValueError as e:
                            st.warning(str(e))
                        except Exception as e:
                            logger.exception("Fake news verification failed")
                            st.error(f"Verification failed: {e}")
                        else:
                            st.session_state.fn_result = {
                                'label': label,
                                'conf': conf,
                                'click_score': click_score,
                                'text': resolved_text,
                                'meta': meta,
                                'source_type': 'image' if input_mode == "Article Image" else 'text',
                                'source_name': source_name,
                            }
                            st.rerun()
        with col2:
            if st.button("🗑️ Reset", key="btn_reset_fn", width="stretch"):
                st.session_state.clear_fn = True
                st.session_state.fn_result = None
                st.session_state.fn_verification_result = None
                st.session_state.fn_image_upload_nonce = st.session_state.get("fn_image_upload_nonce", 0) + 1
                st.rerun()
        with col3:
            if st.button("♻️ GC", help="Trigger Garbage Collection & Clear Resource Cache", key="btn_gc_fn", width="stretch"):
                st.cache_resource.clear()
                gc.collect()
                st.toast("Memory cleaned!")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.fn_result:
            res = st.session_state.fn_result
            st.markdown('<div class="animate-fade">', unsafe_allow_html=True)
            st.markdown("---")

            if res['label'] == 'FAKE':
                cls, label, sub = "res-danger", "🚨 FABRICATED CONTENT", "Models detected high probability of misinformation."
                truth_prob = (1 - res['conf'])
            elif res['label'] == 'UNVERIFIED':
                cls, label, sub = "res-warn", "⚠️ UNVERIFIED CLAIM", "This short claim was not strongly confirmed by reliable reporting."
                truth_prob = 0.5
            else:
                cls, label, sub = "res-safe", "✅ CREDIBLE NEWS", "Content aligns with factual reporting patterns."
                truth_prob = res['conf']

            st.markdown(f'''
            <div class="result-container {cls}">
                <div class="res-label">{label}</div>
                <div class="res-sub">{sub}</div>
            </div>
            ''', unsafe_allow_html=True)

            if res.get('meta') and res['meta'].get('was_translated'):
                st.info(f"🌐 Multilingual Support: Content translated from **{res['meta']['original_language'].upper()}** for cross-language verification.")

            if res.get('meta') and res['meta'].get('ocr'):
                render_ocr_result_panel(res['meta']['ocr'], section_title="Image Reader")

            rt = res.get('meta', {}).get('realtime_result') if res.get('meta') else None

            if (
                res.get('source_type') == 'image'
                and res.get('meta', {}).get('ocr')
                and rt
                and rt.get('status') != 'NO_RESULTS'
                and rt.get('sources')
            ):
                st.markdown('<div class="glass-card" style="border-left: 4px solid #0ea5e9;">', unsafe_allow_html=True)
                render_article_match_browser(
                    rt,
                    widget_namespace=f"fn_image_match_primary_{res.get('source_name') or 'image'}",
                    section_title="Matching Articles From Extracted Image Text",
                    intro_text="These are the live articles matched against the text extracted from your uploaded article image.",
                )
                with st.expander("Image Match Details"):
                    render_live_news_comparison(rt, res.get('text', ''))
                st.markdown('</div>', unsafe_allow_html=True)

            if rt:
                st.markdown('<div class="glass-card" style="border-left: 4px solid #3b82f6;">', unsafe_allow_html=True)
                impact = res['meta'].get('realtime_impact')
                if impact:
                    st.success(f"⚖️ **Consensus Impact:** {impact}")
                st.markdown("### 📡 Real-time Factual Consensus")
                if rt['status'] == 'NO_RESULTS':
                    st.info("No mainstream news reports were found matching this specific claim.")
                else:
                    consensus = rt['consensus_score']
                    verdict_code = rt.get('verdict_code', 'N/A')
                    c_col1, c_col2 = st.columns([1, 2])
                    with c_col1:
                        st.metric("Consensus Index", f"{consensus:.1%}")
                    with c_col2:
                        if consensus > 0.65:
                            st.success(f"✅ **{verdict_code.replace('_', ' ')}**")
                        elif consensus > 0.35:
                            st.warning(f"⚠️ **{verdict_code.replace('_', ' ')}**")
                        else:
                            st.error(f"🚨 **{verdict_code.replace('_', ' ')}**")
                    st.progress(consensus)
                    st.caption(rt['message'])
                    if rt.get('sources'):
                        intro_text = "Review the matched live articles used to support this fake-news decision."
                        section_title = "Matching Articles"
                        details_label = "Live News Comparison Details"
                        if res.get('source_type') != 'image' or not res.get('meta', {}).get('ocr'):
                            render_article_match_browser(
                                rt,
                                widget_namespace=f"fn_match_{res.get('source_name') or res.get('source_type') or 'text'}",
                                section_title=section_title,
                                intro_text=intro_text,
                            )
                            with st.expander(details_label):
                                render_live_news_comparison(rt, res.get('text', ''))
                        else:
                            st.caption("Matching articles from the extracted image text are shown above.")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🔬 Linguistic Forensics")
            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric("Truth Probability", f"{truth_prob:.1%}")
            with col_met2:
                click_score = res.get('click_score', 0)
                st.metric("Clickbait Index", f"{click_score:.2f}",
                          "Suspicious" if click_score > 0.5 else "Professional", delta_color="inverse")
            with col_met3:
                st.metric("Semantic Integrity", "High" if truth_prob > 0.7 else "Compromised" if truth_prob < 0.3 else "Questionable")

            st.progress(truth_prob, text=f"Document Authenticity Index: {truth_prob:.1%}")
            if click_score > 0.6:
                st.warning("⚠️ CRITICAL: Extreme sensationalism detected (High Clickbait score). Standard marker for misinformation.")
            st.markdown('</div>', unsafe_allow_html=True)

            scores = res.get('meta', {}).get('individual_scores', {})
            if scores:
                with st.expander("Model Diagnostics"):
                    st.markdown("#### Weighted Consensus Breakdown")
                    for name, s_res in scores.items():
                        s_cls = "red" if s_res['label'] == 'FAKE' else "green"
                        safe_name = escape(str(name))
                        safe_label = escape(str(s_res['label']))
                        st.markdown(
                            f"**{safe_name}**: <span style='color:{s_cls};'>{safe_label} ({s_res['confidence']:.1%})</span>",
                            unsafe_allow_html=True,
                        )
                        st.progress(s_res['confidence'])
    else:  # Bulk Processing
        file = st.file_uploader("Upload Article File", type=['txt', 'csv'])
        if file:
            if st.button("🚀 Analyze Bulk Content", width="stretch"):
                with st.spinner("Processing file..."):
                    try:
                        truncated = False
                        if file.name.endswith('.csv'):
                            df = _read_uploaded_csv(file, label="Bulk article CSV", nrows=MAX_BULK_TEXT_ROWS)
                            text_col = st.selectbox("Select text column", df.columns) if 'text' not in df.columns else 'text'
                            texts = df[text_col].astype(str).tolist()
                            truncated = len(df) >= MAX_BULK_TEXT_ROWS
                        else:
                            texts, truncated = _read_uploaded_text_lines(
                                file,
                                label="Bulk article text file",
                                max_lines=MAX_BULK_TEXT_ROWS,
                            )

                        max_texts = min(MAX_BULK_TEXT_ROWS, len(texts))
                        results = []
                        progress = st.progress(0)

                        for i, t in enumerate(texts[:max_texts]):
                            try:
                                label, conf, click, meta = fake_news_detector.predict(t)
                                results.append({
                                    'text': t[:100] + '...',
                                    'label': label,
                                    'confidence': f"{conf:.2%}",
                                    'language': meta.get('original_language', 'en'),
                                    'clickbait': f"{click:.2f}",
                                })
                            except Exception:
                                results.append({'text': t[:100] + '...', 'label': 'ERROR', 'confidence': 'N/A'})
                            progress.progress((i + 1) / max_texts)
                    except Exception as e:
                        st.error(f"Bulk processing failed: {e}")
                    else:
                        if truncated:
                            st.info(f"Processed the first {max_texts} rows to keep bulk analysis responsive.")
                        st.dataframe(pd.DataFrame(results), width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# COMMUNICATION ANALYSIS
# ============================================================================

elif st.session_state.page == COMMUNICATION_PAGE:
    sentiment_analyzer, SENTIMENT_AVAILABLE = load_component(get_sentiment_analyzer, "Sentiment analyzer")
    toxicity_detector, TOXICITY_AVAILABLE = load_component(get_toxicity_detector, "Toxicity detector")
    ToxicityVisualizer, _ = load_component(get_toxicity_visualizer, "Toxicity visualizer")

    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("← Back"):
            release_feature_resources(COMMUNICATION_PAGE)
            st.session_state.page = "Home"
            st.rerun()

    st.markdown('<h1 class="animate-fade">🧠 Sentiment, Emotion & Safety</h1>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card animate-fade">', unsafe_allow_html=True)

    communication_mode = _segmented_control(
        "Workspace",
        ["🧠 Unified Scan", "📊 Batch Processing", "🔍 Aspect Insights"],
        default=st.session_state.get("communication_mode", "🧠 Unified Scan"),
        key="communication_mode",
    )

    if communication_mode == "🧠 Unified Scan":
        content_source = _segmented_control(
            "Content Source",
            ["Text Input", "Image to Text"],
            default=st.session_state.get("communication_source_mode", "Text Input"),
            key="communication_source_mode",
        )

        text = ""
        uploaded_message_image = None
        communication_image = None
        image_reader_status = None

        if content_source == "Text Input":
            text = st.text_area(
                "Analyze Tone and Safety Together",
                height=170,
                key="communication_text_input",
                placeholder="Enter customer feedback, social posts, chats, or moderation candidates...",
            )
        else:
            image_reader_status = get_image_reader_status()
            if image_reader_status.get("available"):
                st.caption("Image reader is active. Upload a tweet, post, chat screenshot, or other text-based image to extract the text before analysis.")
            else:
                st.warning(
                    f"Image reader unavailable: {image_reader_status.get('error', 'No OCR backend detected.')}"
                )

            uploader_key = f"communication_image_upload_{st.session_state.get('communication_image_upload_nonce', 0)}"
            uploaded_message_image = st.file_uploader(
                "Upload Text Screenshot",
                type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif', 'tiff'],
                key=uploader_key,
                help="Great for tweets, X posts, chat screenshots, and social-media cards.",
            )

            if uploaded_message_image is not None:
                try:
                    communication_image = _load_uploaded_image(
                        uploaded_message_image,
                        max_size=(1400, 1400),
                        label="Communication image upload",
                    )
                    st.image(communication_image, caption="Uploaded text image", width="stretch")
                except Exception as e:
                    communication_image = None
                    st.error(f"Could not open the uploaded image: {e}")

        col1, col2 = st.columns([2, 1])
        with col1:
            if content_source == "Image to Text":
                st.info("Read text from the image first, then run sentiment and toxicity analysis on the extracted message.")
            else:
                st.info("Run emotional-intent analysis and harmful-content screening from the same message in one pass.")
            if not SENTIMENT_AVAILABLE or sentiment_analyzer is None:
                st.warning("Sentiment analyzer is currently unavailable.")
            if not TOXICITY_AVAILABLE or toxicity_detector is None:
                st.warning("Toxicity detector is currently unavailable.")
        with col2:
            recommended_threshold = float(
                np.clip(
                    getattr(toxicity_detector, "threshold", 0.65) if toxicity_detector is not None else 0.65,
                    0.5,
                    0.8,
                )
            )
            recommended_threshold = round(recommended_threshold / 0.05) * 0.05
            default_threshold = st.session_state.get("communication_toxicity_threshold", recommended_threshold)
            threshold = st.slider(
                "Safety Sensitivity",
                0.1,
                0.9,
                default_threshold,
                0.05,
                key="communication_toxicity_threshold",
            )
            if TOXICITY_AVAILABLE and toxicity_detector is not None:
                toxicity_detector.threshold = threshold
            if threshold < recommended_threshold:
                st.caption(
                    f"Recommended baseline: {recommended_threshold:.2f}. Lower values are more aggressive and can increase false positives for neutral greetings."
                )
            else:
                st.caption(
                    f"Recommended baseline: {recommended_threshold:.2f}. Messages are flagged only when the risk score crosses the active threshold."
                )

        action_col1, action_col2 = st.columns([3, 1])
        with action_col1:
            if st.button("🔍 Run Communication Analysis", key="btn_comm_scan", width="stretch"):
                if content_source == "Text Input" and not text.strip():
                    st.warning("Please enter text for analysis.")
                elif content_source == "Image to Text" and communication_image is None:
                    if image_reader_status and not image_reader_status.get("available"):
                        st.warning(image_reader_status.get("error", "Image reader is unavailable."))
                    else:
                        st.warning("Please upload an image with readable text for analysis.")
                elif not SENTIMENT_AVAILABLE or sentiment_analyzer is None:
                    st.error("Sentiment analyzer module not available.")
                elif not TOXICITY_AVAILABLE or toxicity_detector is None:
                    st.error("Toxicity detector module not available.")
                else:
                    spinner_label = (
                        "Reading text from image and analyzing tone and safety markers..."
                        if content_source == "Image to Text"
                        else "Analyzing tone and safety markers..."
                    )
                    with st.spinner(spinner_label):
                        try:
                            analysis_text = text
                            source_type = "text"
                            source_name = None
                            ocr_meta = None

                            if content_source == "Image to Text":
                                ocr_payload = extract_text_from_image(
                                    communication_image,
                                    image_name=getattr(uploaded_message_image, "name", None),
                                )
                                extracted_text = str(ocr_payload.get("text", "")).strip()
                                normalized_extracted_text = " ".join(extracted_text.split())
                                if len(normalized_extracted_text) < 4:
                                    raise ValueError(
                                        "The image reader could not extract enough readable text. "
                                        "Please upload a clearer tweet or screenshot, or paste the text directly."
                                    )

                                analysis_text = extracted_text
                                source_type = "image"
                                source_name = ocr_payload.get("image_name") or getattr(uploaded_message_image, "name", None)
                                ocr_meta = {
                                    "backend": ocr_payload.get("backend", "windows_ocr"),
                                    "image_name": source_name,
                                    "line_count": ocr_payload.get("line_count", 0),
                                    "word_count": ocr_payload.get("word_count", 0),
                                    "language": ocr_payload.get("language", ""),
                                    "width": ocr_payload.get("width"),
                                    "height": ocr_payload.get("height"),
                                    "preprocessed_size": ocr_payload.get("preprocessed_size"),
                                    "extracted_text": extracted_text,
                                }

                            sentiment_label, sentiment_conf, sentiment_meta = sentiment_analyzer.analyze(analysis_text)
                            is_toxic, tox_conf, cats, explanation, tox_meta = toxicity_detector.predict(analysis_text)
                        except ValueError as e:
                            st.warning(str(e))
                        except Exception as e:
                            logger.exception("Unified communication analysis failed")
                            st.error(f"Communication analysis failed: {e}")
                        else:
                            sentiment_meta = dict(sentiment_meta or {})
                            tox_meta = dict(tox_meta or {})
                            tox_meta["decision_threshold"] = threshold
                            if ocr_meta:
                                sentiment_meta["ocr"] = dict(ocr_meta)
                                sentiment_meta["source_type"] = "image"
                                tox_meta["ocr"] = dict(ocr_meta)
                                tox_meta["source_type"] = "image"

                            _clear_feature_session_state(COMMUNICATION_PAGE)
                            st.session_state.sentiment_result = {
                                'label': sentiment_label,
                                'conf': sentiment_conf,
                                'text': analysis_text,
                                'type': 'single',
                                'meta': sentiment_meta,
                                'source_type': source_type,
                                'source_name': source_name,
                            }
                            st.session_state.last_sentiment = (sentiment_label, sentiment_conf)
                            st.session_state.toxicity_result = {
                                'text': analysis_text,
                                'is_toxic': is_toxic,
                                'confidence': tox_conf,
                                'threshold': threshold,
                                'categories': cats,
                                'explanation': explanation,
                                'meta': tox_meta,
                                'source_type': source_type,
                                'source_name': source_name,
                            }
                            st.session_state.last_toxicity = (is_toxic, tox_conf)
                            st.rerun()
        with action_col2:
            if st.button("🗑️ Reset", key="btn_comm_reset", width="stretch"):
                _clear_feature_session_state(COMMUNICATION_PAGE)
                st.session_state.communication_text_input = ""
                st.session_state.communication_image_upload_nonce = st.session_state.get("communication_image_upload_nonce", 0) + 1
                st.rerun()

    elif communication_mode == "📊 Batch Processing":
        st.markdown("##### Process Multiple Communications")
        uploaded_file = st.file_uploader("Upload Content (CSV or TXT)", type=['csv', 'txt'], key="sa_batch_file")
        selected_text_col = None
        if uploaded_file and uploaded_file.name.endswith('.csv'):
            try:
                preview_df = _read_uploaded_csv(
                    uploaded_file,
                    label="Batch communication CSV",
                    nrows=MAX_BULK_TEXT_ROWS,
                )
            except Exception as e:
                st.error(f"Could not read the uploaded CSV: {e}")
            else:
                uploaded_file.seek(0)
                selected_text_col = st.selectbox(
                    "Select Text Column",
                    preview_df.columns,
                    index=preview_df.columns.get_loc('text') if 'text' in preview_df.columns else 0,
                    key="sa_batch_text_column",
                )
        if uploaded_file and st.button("🚀 Execute Batch Analysis", width="stretch", key="btn_batch_sa"):
            with st.spinner("Analyzing communication streams..."):
                try:
                    from batch_sentiment import BatchSentimentProcessor

                    processor = BatchSentimentProcessor()
                    if uploaded_file.name.endswith('.csv'):
                        df = _read_uploaded_csv(
                            uploaded_file,
                            label="Batch communication CSV",
                            nrows=MAX_BULK_TEXT_ROWS,
                        )
                        text_col = selected_text_col or ('text' if 'text' in df.columns else df.columns[0])
                        results_df = processor.process_texts(df[text_col].astype(str).tolist())
                        if len(df) >= MAX_BULK_TEXT_ROWS:
                            st.info(f"Processed the first {len(df)} rows to keep batch analysis responsive.")
                    else:
                        texts, truncated = _read_uploaded_text_lines(
                            uploaded_file,
                            label="Batch communication text file",
                            max_lines=MAX_BULK_TEXT_ROWS,
                        )
                        results_df = processor.process_texts(texts)
                        if truncated:
                            st.info(f"Processed the first {len(texts)} rows to keep batch analysis responsive.")
                except Exception as e:
                    logger.exception("Batch sentiment analysis failed")
                    st.error(f"Batch processing failed: {e}")
                else:
                    _clear_feature_session_state("Sentiment Analysis")
                    artifact = _persist_dataframe_artifact(results_df, "sentiment_batch")
                    st.session_state.sentiment_result = {'type': 'batch', **artifact}
                    st.rerun()

    else:
        AspectSentimentAnalyzer, ASPECT_AVAILABLE = load_component(
            get_aspect_sentiment_analyzer, "Aspect sentiment analyzer"
        )
        if not ASPECT_AVAILABLE or AspectSentimentAnalyzer is None:
            st.warning("Aspect analysis module is currently initializing or unavailable.")
        else:
            aspect_analyzer = AspectSentimentAnalyzer()
            st.markdown("##### Targeted Aspect Analysis")
            aspect_text = st.text_area(
                "Analyze Specific Qualities",
                height=150,
                key="sa_aspect_input",
                placeholder="Feedback about service, price, quality, tone, or design...",
            )
            if st.button("🔍 Extract Aspect Sentiment", width="stretch", key="btn_aspect_sa"):
                if not aspect_text.strip():
                    st.warning("Please enter text for aspect analysis.")
                else:
                    with st.spinner("Extracting linguistic aspects..."):
                        try:
                            results = aspect_analyzer.analyze_aspects(aspect_text)
                        except Exception as e:
                            logger.exception("Aspect sentiment analysis failed")
                            st.error(f"Aspect analysis failed: {e}")
                        else:
                            _clear_feature_session_state("Sentiment Analysis")
                            st.session_state.sentiment_result = {'type': 'aspect', 'results': results, 'text': aspect_text}
                            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    sentiment_result = st.session_state.get("sentiment_result")
    toxicity_result = st.session_state.get("toxicity_result")

    if communication_mode == "🧠 Unified Scan":
        single_sentiment = sentiment_result if sentiment_result and sentiment_result.get("type") == "single" else None
        if single_sentiment or toxicity_result:
            st.markdown('<div class="animate-fade">', unsafe_allow_html=True)
            st.markdown("---")
            render_communication_summary(single_sentiment, toxicity_result)
            ocr_meta = None
            if single_sentiment and single_sentiment.get("meta"):
                ocr_meta = single_sentiment["meta"].get("ocr")
            if not ocr_meta and toxicity_result and toxicity_result.get("meta"):
                ocr_meta = toxicity_result["meta"].get("ocr")
            if ocr_meta:
                render_ocr_result_panel(ocr_meta, section_title="Image Reader")
            if single_sentiment:
                st.markdown("### 😊 Sentiment Findings")
                render_sentiment_result_panel(single_sentiment)
            if toxicity_result:
                st.markdown("### ⚠️ Safety Findings")
                render_toxicity_result_panel(toxicity_result, ToxicityVisualizer)
            st.markdown('</div>', unsafe_allow_html=True)

    elif communication_mode == "📊 Batch Processing":
        if sentiment_result and sentiment_result.get("type") == "batch":
            st.markdown('<div class="animate-fade">', unsafe_allow_html=True)
            st.markdown("---")
            render_sentiment_result_panel(sentiment_result)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        if sentiment_result and sentiment_result.get("type") == "aspect":
            st.markdown('<div class="animate-fade">', unsafe_allow_html=True)
            st.markdown("---")
            render_sentiment_result_panel(sentiment_result)
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
footer_brand = HOME_PAGE_BRAND if st.session_state.page == "Home" else PROJECT_BRAND
footer_copyright = HOME_PAGE_BRAND if st.session_state.page == "Home" else f"{PROJECT_BRAND} Systems"
st.markdown(f"""
<div class="animate-fade" style='text-align:center; color:#64748b; padding:2rem;'>
    <p style="font-weight:600; margin-bottom:0.4rem;">🛡️ {footer_brand} - Advanced Sovereignty Platform</p>
    <p style='font-size:0.85rem; letter-spacing:0.05em;'>INTELLIGENT VERIFICATION • MEDIA AUTHENTICITY • REAL-TIME ANALYSIS</p>
    <p style='font-size:0.75rem; margin-top:1.5rem; color:#475569;'>© 2024 {footer_copyright}. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)

# End of app.py
