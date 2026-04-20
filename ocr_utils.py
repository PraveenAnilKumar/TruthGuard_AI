"""
OCR helpers for reading article text from uploaded images.

The implementation uses the built-in Windows OCR runtime so the fake-news
detector can verify screenshots, social cards, or scanned clippings without
requiring a heavyweight third-party OCR dependency.
"""

import io
import json
import logging
import os
import re
import subprocess
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
OCR_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "windows_ocr.ps1"
OCR_TEMP_DIR = PROJECT_ROOT / "temp" / "ocr_inputs"


def _safe_image_stem(name: str) -> str:
    stem = Path(name or "article_image").stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return (stem or "article_image")[:60]


def _load_image(image_source: Any, image_name: Optional[str] = None) -> Tuple[Image.Image, str]:
    if isinstance(image_source, Image.Image):
        return image_source.copy(), image_name or "article_image.png"

    if isinstance(image_source, (str, Path)):
        path = Path(image_source)
        with Image.open(path) as image:
            return image.copy(), image_name or path.name

    if isinstance(image_source, (bytes, bytearray)):
        with Image.open(io.BytesIO(image_source)) as image:
            return image.copy(), image_name or "article_image.png"

    if hasattr(image_source, "read"):
        payload = image_source.read()
        if hasattr(image_source, "seek"):
            try:
                image_source.seek(0)
            except Exception:
                pass
        with Image.open(io.BytesIO(payload)) as image:
            source_name = getattr(image_source, "name", None) or image_name or "article_image.png"
            return image.copy(), source_name

    raise TypeError(
        "Unsupported image source. Use a PIL image, file-like object, bytes, or a filesystem path."
    )


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Normalize an uploaded article image before OCR with aggressive contrast forcing."""
    working = ImageOps.exif_transpose(image)
    if working.mode not in ("RGB", "L"):
        working = working.convert("RGB")

    max_dimension = max(working.size) if working.size else 0
    # Dynamically target up to 2400px to ensure tiny standard news fonts become legible
    if max_dimension and max_dimension < 2400:
        scale = min(3.5, 2400 / float(max_dimension))
        resized = (
            max(1, int(working.width * scale)),
            max(1, int(working.height * scale)),
        )
        working = working.resize(resized, Image.Resampling.LANCZOS)

    grayscale = ImageOps.grayscale(working)
    
    # Aggressive macro-filtering to destroy grey and force binary textual layouts
    grayscale = ImageOps.autocontrast(grayscale, cutoff=5)
    grayscale = ImageEnhance.Contrast(grayscale).enhance(3.0)
    grayscale = ImageEnhance.Brightness(grayscale).enhance(1.1)
    grayscale = ImageEnhance.Sharpness(grayscale).enhance(3.5)
    
    # Heavy binarization gate (snap remaining fuzz to pure white or pure black)
    grayscale = grayscale.point(lambda p: 255 if p > 165 else 0)
    
    return grayscale


def _save_preprocessed_image(image: Image.Image, image_name: str) -> Path:
    OCR_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_image_stem(image_name)}_{uuid.uuid4().hex[:10]}.png"
    target_path = OCR_TEMP_DIR / filename
    image.save(target_path, format="PNG", optimize=True)
    return target_path


def _parse_ocr_stdout(raw_output: str) -> Dict[str, Any]:
    """Extract the OCR JSON payload even if PowerShell emits extra noise."""
    cleaned_output = (raw_output or "").replace("\x00", "").strip()
    if not cleaned_output:
        raise RuntimeError("Windows OCR returned no output.")

    decoder = json.JSONDecoder()
    candidates = []
    lines = [line.strip().lstrip("\ufeff") for line in cleaned_output.splitlines() if line.strip()]
    candidates.extend(reversed(lines))
    candidates.append(cleaned_output.lstrip("\ufeff"))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    for match in re.finditer(r"\{", cleaned_output):
        try:
            payload, _ = decoder.raw_decode(cleaned_output[match.start():].lstrip("\ufeff"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    logger.error("Invalid OCR response: %s", cleaned_output[:2000])
    raise RuntimeError("Windows OCR returned an invalid response.")


def _parse_ocr_json_file(output_path: Path) -> Dict[str, Any]:
    """Read a JSON payload written by the PowerShell OCR script."""
    try:
        raw_output = output_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError("Windows OCR did not produce a result file.") from exc
    except OSError as exc:
        raise RuntimeError(f"Windows OCR result file could not be read: {exc}") from exc

    return _parse_ocr_stdout(raw_output)


def _run_windows_ocr(image_path: Path) -> Dict[str, Any]:
    result_path = OCR_TEMP_DIR / f"{image_path.stem}_{uuid.uuid4().hex[:10]}.json"
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(OCR_SCRIPT_PATH),
        "-ImagePath",
        str(image_path),
        "-OutputPath",
        str(result_path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )

        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "Windows OCR failed."
            raise RuntimeError(message)

        if result_path.exists():
            return _parse_ocr_json_file(result_path)

        return _parse_ocr_stdout(completed.stdout)
    finally:
        try:
            result_path.unlink(missing_ok=True)
        except Exception:
            logger.debug("Could not remove OCR result file %s", result_path, exc_info=True)


@lru_cache(maxsize=1)
def get_image_reader_status() -> Dict[str, Any]:
    """Report whether the built-in image reader is available."""
    if os.name != "nt":
        return {
            "available": False,
            "backend": None,
            "error": "Image reading is currently implemented for Windows only.",
        }

    if not OCR_SCRIPT_PATH.exists():
        return {
            "available": False,
            "backend": "windows_ocr",
            "error": f"OCR script is missing at {OCR_SCRIPT_PATH}.",
        }

    probe_command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            "$ErrorActionPreference='Stop'; "
            "Add-Type -AssemblyName System.Runtime.WindowsRuntime; "
            "[Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType = WindowsRuntime] | Out-Null; "
            "Write-Output 'WINDOWS_OCR_READY'"
        ),
    ]

    try:
        completed = subprocess.run(
            probe_command,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return {
            "available": False,
            "backend": "windows_ocr",
            "error": f"Failed to probe Windows OCR: {exc}",
        }

    if completed.returncode != 0 or "WINDOWS_OCR_READY" not in completed.stdout:
        error_message = completed.stderr.strip() or completed.stdout.strip() or "Unknown OCR initialization failure."
        return {
            "available": False,
            "backend": "windows_ocr",
            "error": error_message,
        }

    return {
        "available": True,
        "backend": "windows_ocr",
        "engine": "Windows OCR",
    }


def extract_text_from_image(image_source: Any, image_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract article text from an image using the Windows OCR runtime.

    Returns metadata that can be merged into fake-news verification results.
    """
    status = get_image_reader_status()
    if not status.get("available"):
        raise RuntimeError(status.get("error") or "Image reader is unavailable.")

    loaded_image, resolved_name = _load_image(image_source, image_name=image_name)
    prepared_image = preprocess_image_for_ocr(loaded_image)
    temp_image_path = _save_preprocessed_image(prepared_image, resolved_name)

    try:
        payload = _run_windows_ocr(temp_image_path)
    finally:
        try:
            temp_image_path.unlink(missing_ok=True)
        except Exception:
            logger.debug("Could not remove temporary OCR file %s", temp_image_path, exc_info=True)

    raw_text = str(payload.get("text", ""))
    
    # Aggressive Noise Sterilization 
    sanitized_lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        # Drop absolutely empty lines or tiny 1-2 character OCR fragment artifacts
        if len(line) < 3 and not re.search(r'[A-Za-z0-9]{2,}', line):
            continue
        # Drop lines comprised solely of UI widget noise or stray border symbols
        if re.search(r'^[^a-zA-Z0-9]+$', line):
            continue
        # Drop lines where actual text makes up less than 25% of the string (hallucinations)
        if sum(c.isalnum() for c in line) < len(line) * 0.25:
            continue
        sanitized_lines.append(line)
        
    extracted_text = " ".join(sanitized_lines) # Condense blocks into smooth paragraph format
    extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()

    payload.update(
        {
            "text": extracted_text,
            "raw_text": raw_text, # Keep original in payload for forensics if needed
            "backend": status.get("backend", "windows_ocr"),
            "image_name": resolved_name,
            "preprocessed_size": list(prepared_image.size),
        }
    )
    return payload
