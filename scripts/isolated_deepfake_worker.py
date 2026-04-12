"""
Standalone worker for deepfake analysis.

This process can safely import TensorFlow without taking down the main
Streamlit app if the runtime is unstable.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

import cv2
import numpy as np


for key, value in {
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "KERAS_BACKEND": "tensorflow",
    "CUDA_VISIBLE_DEVICES": "-1",
    "TRUTHGUARD_DEEPFAKE_WORKER": "1",
    "TRUTHGUARD_ENABLE_ADVANCED_FACE_DETECTORS": "1",
    "TRUTHGUARD_ENABLE_HF_DEEPFAKE_MODELS": "0",
}.items():
    os.environ.setdefault(key, value)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def _serialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.ndim >= 2:
            arr = value
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            success, encoded = cv2.imencode(".png", arr)
            if success:
                return {
                    "__type__": "image_bytes",
                    "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
                }
            return arr.tolist()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {
            "__type__": "image_bytes",
            "data": base64.b64encode(value).decode("ascii"),
        }
    return value


def _load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image at {path}")
    return image


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception as exc:
        _emit({"ok": False, "error": f"Invalid worker payload: {exc}"})
        return 0

    try:
        from deepfake_detector_advanced import DeepfakeDetectorAdvanced

        threshold = float(payload.get("threshold", 0.5))
        detector = DeepfakeDetectorAdvanced(threshold=threshold, models_dir=payload.get("models_dir", "models"))
        detector.threshold = threshold
        requested_models = payload.get("requested_models") or None
        action = payload.get("action")

        if action == "image_single":
            result = detector.detect_with_single_model(
                _load_image(payload["image_path"]),
                payload["model_name"],
                return_heatmap=bool(payload.get("return_heatmap", False)),
            )
        elif action == "image_ensemble":
            result = detector.detect_deepfake_ensemble(
                _load_image(payload["image_path"]),
                return_heatmap=bool(payload.get("return_heatmap", False)),
                requested_models=requested_models,
            )
        elif action == "video_single":
            result = detector.detect_video_with_single_model(
                payload["video_path"],
                payload["model_name"],
                return_heatmap=bool(payload.get("return_heatmap", False)),
            )
        elif action == "video_ensemble":
            result = detector.detect_deepfake_video_advanced(
                payload["video_path"],
                requested_models=requested_models,
                return_heatmap=bool(payload.get("return_heatmap", False)),
            )
        else:
            result = {"error": f"Unsupported deepfake worker action: {action}"}

        _emit({"ok": True, "result": _serialize_value(result)})
    except Exception as exc:
        _emit({"ok": False, "error": f"Deepfake worker error: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
