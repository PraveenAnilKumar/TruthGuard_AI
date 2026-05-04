"""
Project-wide Python startup hardening for local tools and tests.

Python imports this module automatically when the project root is on sys.path.
Keeping the settings here makes CLI scripts behave like the Streamlit app:
quiet ML logs, modest thread usage, and Unicode-safe output on Windows.
"""

import os
import sys


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _reconfigure_stream(stream):
    try:
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


_reconfigure_stream(getattr(sys, "stdout", None))
_reconfigure_stream(getattr(sys, "stderr", None))
