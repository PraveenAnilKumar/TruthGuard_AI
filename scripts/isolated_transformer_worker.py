"""
Small standalone worker for isolated transformer inference.

The parent process passes a JSON payload on stdin and receives a JSON response
on stdout.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List


for key, value in {
    "TRANSFORMERS_NO_TF": "1",
    "TRANSFORMERS_NO_FLAX": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}.items():
    os.environ.setdefault(key, value)


def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(_coerce_text(item) for item in value if item is not None)
    return str(value)


def _labels_from_mapping(label_map: Dict[Any, Any], count: int) -> List[str]:
    return [str(label_map.get(index, f"LABEL_{index}")) for index in range(count)]


def _predict_text_classification(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_ref = payload.get("model")
    tokenizer_ref = payload.get("tokenizer") or model_ref
    local_files_only = bool(payload.get("local_files_only", False))
    max_length = int(payload.get("max_length") or 512)
    text = _coerce_text(payload.get("text"))

    if not model_ref:
        return {"ok": False, "error": "Missing model reference."}

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_ref,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        outputs = model(**encoded)

    logits = outputs.logits[0]
    label_map = getattr(model.config, "id2label", {}) or {}

    if len(logits.shape) == 0 or logits.shape[-1] == 1:
        positive_score = float(torch.sigmoid(logits.reshape(-1)[0]).item())
        scores = [float(1.0 - positive_score), positive_score]
        labels = _labels_from_mapping(label_map, 2)
        top_index = 1 if positive_score >= 0.5 else 0
    else:
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        scores = [float(value) for value in probabilities.tolist()]
        labels = _labels_from_mapping(label_map, len(scores))
        top_index = max(range(len(scores)), key=lambda idx: scores[idx])

    return {
        "ok": True,
        "result": {
            "label": labels[top_index],
            "score": float(scores[top_index]),
            "scores": scores,
            "labels": labels,
        },
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception as exc:
        _emit({"ok": False, "error": f"Invalid JSON payload: {exc}"})
        return 0

    action = payload.get("action")

    try:
        if action == "predict_text_classification":
            result = _predict_text_classification(payload)
        else:
            result = {"ok": False, "error": f"Unsupported action: {action}"}
    except Exception as exc:
        result = {"ok": False, "error": f"Worker error: {exc}"}

    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
