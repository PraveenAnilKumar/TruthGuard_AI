"""
Run deepfake analysis in an isolated subprocess.

TensorFlow imports can crash the shared Streamlit process on some Windows
installations after other ML libraries have been loaded. This module keeps the
deepfake execution path out-of-process so the app can survive those failures.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)

_WORKER_PATH = Path(__file__).resolve().parent / "scripts" / "isolated_deepfake_worker.py"
_WINDOWS_FATAL_EXIT_CODES = {-1073741819, 3221225477}


def _build_worker_env(*, enable_hf_models: bool = False) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("KERAS_BACKEND", "tensorflow")
    env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    env.setdefault("TRUTHGUARD_DEEPFAKE_WORKER", "1")
    env.setdefault("TRUTHGUARD_ENABLE_ADVANCED_FACE_DETECTORS", "1")
    env["TRUTHGUARD_ENABLE_HF_DEEPFAKE_MODELS"] = "1" if enable_hf_models else "0"
    return env


def _decode_payload(value: Any) -> Any:
    if isinstance(value, dict):
        marker = value.get("__type__")
        if marker == "image_bytes":
            return base64.b64decode(value.get("data", ""))
        return {key: _decode_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_payload(item) for item in value]
    return value


def _run_worker(payload: Dict[str, Any], timeout_seconds: int) -> Dict[str, Any]:
    if not _WORKER_PATH.exists():
        return {"error": f"Deepfake worker not found at {_WORKER_PATH}"}

    try:
        completed = subprocess.run(
            [sys.executable, str(_WORKER_PATH)],
            input=json.dumps(payload),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
            cwd=str(_WORKER_PATH.parent.parent),
            env=_build_worker_env(enable_hf_models=bool(payload.get("enable_hf_models"))),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Deepfake worker timed out after {timeout_seconds}s."}
    except Exception as exc:
        return {"error": f"Failed to start deepfake worker: {exc}"}

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()

    if completed.returncode != 0:
        error = f"Deepfake worker exited with code {completed.returncode}."
        if completed.returncode in _WINDOWS_FATAL_EXIT_CODES:
            error = (
                "Deepfake worker hit a fatal TensorFlow/runtime crash. "
                "The Streamlit app stayed alive and can recover."
            )
        if stderr:
            error = f"{error} Details: {stderr[-1200:]}"
        elif stdout:
            error = f"{error} Details: {stdout[-1200:]}"
        return {"error": error}

    if not stdout:
        return {"error": "Deepfake worker returned no output."}

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return {"error": f"Deepfake worker returned invalid JSON: {stdout[-1200:]}"}

    if not payload.get("ok"):
        return {"error": payload.get("error", "Deepfake worker failed.")}

    result = _decode_payload(payload.get("result", {}))
    if stderr:
        result.setdefault("worker_stderr", stderr[-1200:])
    return result


def _write_temp_image(image_array: np.ndarray) -> str:
    arr = np.asarray(image_array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    handle.close()
    if not cv2.imwrite(handle.name, arr):
        raise RuntimeError("Could not write temporary image for deepfake worker.")
    return handle.name


def run_isolated_deepfake_image_analysis(
    image_array: np.ndarray,
    *,
    threshold: float,
    return_heatmap: bool = False,
    model_name: Optional[str] = None,
    requested_models: Optional[List[str]] = None,
    enable_hf_models: bool = False,
    timeout_seconds: int = 600,
) -> Dict[str, Any]:
    image_path = _write_temp_image(image_array)
    try:
        payload = {
            "action": "image_single" if model_name else "image_ensemble",
            "image_path": image_path,
            "threshold": threshold,
            "return_heatmap": return_heatmap,
            "model_name": model_name,
            "requested_models": requested_models or [],
            "enable_hf_models": bool(enable_hf_models),
            "models_dir": "models",
        }
        return _run_worker(payload, timeout_seconds=timeout_seconds)
    finally:
        try:
            os.unlink(image_path)
        except OSError:
            pass


def run_isolated_deepfake_video_analysis(
    video_path: str,
    *,
    threshold: float,
    return_heatmap: bool = False,
    model_name: Optional[str] = None,
    requested_models: Optional[List[str]] = None,
    enable_hf_models: bool = False,
    timeout_seconds: int = 1200,
) -> Dict[str, Any]:
    payload = {
        "action": "video_single" if model_name else "video_ensemble",
        "video_path": video_path,
        "threshold": threshold,
        "return_heatmap": return_heatmap,
        "model_name": model_name,
        "requested_models": requested_models or [],
        "enable_hf_models": bool(enable_hf_models),
        "models_dir": "models",
    }
    return _run_worker(payload, timeout_seconds=timeout_seconds)
