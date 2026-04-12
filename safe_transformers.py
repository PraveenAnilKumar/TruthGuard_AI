"""
Helpers for running transformer inference in an isolated subprocess.

This keeps risky torch/transformers imports out of the Streamlit process on
Windows setups where importing them after other ML libraries can trigger a
fatal access violation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

_WORKER_PATH = Path(__file__).resolve().parent / "scripts" / "isolated_transformer_worker.py"
_WINDOWS_FATAL_EXIT_CODES = {-1073741819, 3221225477}


def _trim_output(value: str, limit: int = 1200) -> str:
    value = (value or "").strip()
    if len(value) <= limit:
        return value
    return value[-limit:]


def _build_worker_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    env.setdefault("TRANSFORMERS_NO_FLAX", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return env


def _run_worker(payload: Dict[str, Any], timeout_seconds: int = 180) -> Dict[str, Any]:
    if not _WORKER_PATH.exists():
        return {
            "ok": False,
            "error": f"Transformer worker not found at {_WORKER_PATH}",
        }

    command = [sys.executable, str(_WORKER_PATH)]

    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            cwd=str(_WORKER_PATH.parent.parent),
            env=_build_worker_env(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"Isolated transformer worker timed out after {timeout_seconds}s.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Failed to start isolated transformer worker: {exc}",
        }

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    if completed.returncode != 0:
        error = f"Transformer worker exited with code {completed.returncode}."
        if completed.returncode in _WINDOWS_FATAL_EXIT_CODES:
            error = (
                "Transformer worker exited after a fatal Windows access violation. "
                "The app stayed alive and will fall back safely."
            )
        details = []
        if stderr:
            details.append(_trim_output(stderr))
        if stdout:
            details.append(_trim_output(stdout))
        if details:
            error = f"{error} Details: {' | '.join(details)}"
        return {
            "ok": False,
            "error": error,
            "returncode": completed.returncode,
        }

    if not stdout:
        return {
            "ok": False,
            "error": "Transformer worker returned no output.",
        }

    try:
        result = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": f"Transformer worker returned invalid JSON: {_trim_output(stdout)}",
        }

    if stderr:
        result.setdefault("stderr", _trim_output(stderr))
    return result


def run_isolated_text_classification(
    model_ref: str,
    text: Any,
    tokenizer_ref: Optional[str] = None,
    *,
    local_files_only: bool = False,
    max_length: int = 512,
    timeout_seconds: int = 180,
) -> Dict[str, Any]:
    """Run sequence classification in a fresh Python process."""
    payload = {
        "action": "predict_text_classification",
        "model": model_ref,
        "tokenizer": tokenizer_ref or model_ref,
        "text": text,
        "local_files_only": local_files_only,
        "max_length": max_length,
    }
    return _run_worker(payload, timeout_seconds=timeout_seconds)
