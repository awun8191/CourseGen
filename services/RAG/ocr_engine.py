#!/usr/bin/env python3
"""
Standalone OCR engine: Gemini (load-balanced) + EasyOCR/PaddleOCR (no Tesseract).

Usage example:
  export GEMINI_SERVICE_PATH="/home/you/your-repo/src/services/Gemini/gemini_service.py"
  export GEMINI_MODEL="gemini-2.5-flash-lite"
  # optional fallback if service import fails:
  # export GEMINI_API_KEY="..."

  python ocr_engine.py "/path/to/file.pdf" --engine gemini --dpi 300

Key env:
  GEMINI_SERVICE_PATH=/abs/path/to/src/services/Gemini/gemini_service.py
  GEMINI_MODEL=gemini-2.5-flash-lite
  GEMINI_API_KEY=...  (only used if service import fails)

  OCR_GEMINI_MAX_EDGE=2000                 # downscale long edge before Gemini (px)
  OCR_GEMINI_MAX_TOKENS=8192
  OCR_GEMINI_AUTOFALLBACK=1                # if Gemini page -> empty, use local OCR
  OCR_GEMINI_FALLBACK_ENGINE=hybrid        # easyocr|paddleocr|hybrid
  OCR_GEMINI_FALLBACK_HAND=1               # handwriting preprocessing for fallback

  OCR_HANDWRITTEN=0/1
  OCR_UPSCALE=1.6
  OCR_SAVE_DEBUG=0/1
  OCR_CONF_TH=0.35
  OCR_HANDWRITTEN_CONF_TH=0.15
  OCR_MIN_LINES=5
  OCR_MIN_AVG_CONF=0.50
  OCR_LOG_LEVEL=INFO|DEBUG

  EASYOCR_GPU=0/1
  PADDLE_GPU=0/1
  PADDLE_TEXTLINE_ORI=1
  PADDLE_LANG=en
"""

from __future__ import annotations

import io
import os
import sys
import json
import time
import logging
import importlib
import importlib.util
import types
import re
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# ----------------- Optional deps -----------------
try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import cv2
except Exception:
    cv2 = None  # type: ignore

try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore

easyocr = None  # type: ignore
PaddleOCR = None  # type: ignore
_HAS_EASYOCR = False
_HAS_PADDLE = False

# ----------------- Logging -----------------
log = logging.getLogger("ocr_engine")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(filename)s:%(lineno)d - %(message)s"))
    log.addHandler(h)
log.setLevel(os.getenv("OCR_LOG_LEVEL", "INFO").upper())

# ----------------- Data models -----------------
@dataclass
class PageMetrics:
    page: int
    engine: str
    lines: int
    avg_conf: float
    min_conf: float
    max_conf: float
    dpi: Optional[int] = None

@dataclass
class OCRResult:
    text: str
    by_page: List[PageMetrics]
    meta: Dict[str, Union[str, int, float, bool, None]]
    debug_images: Optional[List["np.ndarray"]] = None

# ----------------- Common helpers -----------------
_OCR_PROMPT_SINGLE = (
    "Transcribe the exact visible text from this page image into plain text.\n"
    "Preserve the original line breaks and spacing where obvious.\n"
    "Do not include any explanations, headings, labels, JSON, or code fences.\n"
    "Return ONLY the text content as a single plain string."
)

_OCR_PROMPT_BATCH = (
    "Transcribe the exact visible text from these page images into plain text.\n"
    "Provide the text for each page separately, labeled as Page 1:, Page 2:, etc.\n"
    "Preserve the original line breaks and spacing where obvious.\n"
    "Do not include any explanations, headings, labels, JSON, or code fences.\n"
    "Return ONLY the labeled text content."
)

def _ensure_np(img: Union["np.ndarray", "Image.Image"]) -> "np.ndarray":
    assert np is not None, "NumPy not available"
    if isinstance(img, np.ndarray):
        return img
    if Image is None:
        raise RuntimeError("Pillow missing to convert image")
    return np.array(img.convert("RGB"))

def _to_pil(img_rgb: Union["np.ndarray", "Image.Image"]) -> "Image.Image":
    if Image is None:
        raise RuntimeError("Pillow required")
    if isinstance(img_rgb, Image.Image):
        return img_rgb.convert("RGB")
    return Image.fromarray(img_rgb.astype("uint8"), mode="RGB")

def _limit_long_edge(img_pil: "Image.Image", max_edge: int = 2000) -> "Image.Image":
    if max_edge <= 0:
        return img_pil
    w, h = img_pil.size
    m = max(w, h)
    if m <= max_edge:
        return img_pil
    scale = max_edge / float(m)
    new_size = (int(w * scale), int(h * scale))
    return img_pil.resize(new_size, Image.LANCZOS)

def _snapshot(s: str, n: int = 160) -> str:
    s = (s or "").replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s

# ----------------- Handwriting preprocessing (for Easy/Paddle) -----------------
def _deskew(gray: "np.ndarray") -> "np.ndarray":
    if cv2 is None:
        return gray
    try:
        _th, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(bw)
        if coords is None or len(coords) < 200:
            return gray
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        M = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), angle, 1.0)
        return cv2.warpAffine(
            gray, M, (gray.shape[1], gray.shape[0]),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    except Exception as e:
        log.debug(f"deskew failed: {e}")
        return gray

def _remove_shadow(gray: "np.ndarray") -> "np.ndarray":
    if cv2 is None:
        return gray
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
        bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        no_shadow = cv2.subtract(gray, bg)
        return cv2.normalize(no_shadow, None, 0, 255, cv2.NORM_MINMAX)
    except Exception as e:
        log.debug(f"shadow removal failed: {e}")
        return gray

def _adaptive_binarize(gray: "np.ndarray") -> "np.ndarray":
    if cv2 is None:
        return gray
    try:
        den = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        den = clahe.apply(den)
        bin_img = cv2.adaptiveThreshold(
            den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
        )
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    except Exception as e:
        log.debug(f"binarize failed: {e}")
        return gray

def _maybe_upscale(img: "np.ndarray", factor: float) -> "np.ndarray":
    if cv2 is None or factor <= 1.01:
        return img
    try:
        h, w = img.shape[:2]
        if max(h, w) > 2400:
            return img
        return cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        log.debug(f"upscale failed: {e}")
        return img

def preprocess_for_handwriting(img_rgb: Union["np.ndarray", "Image.Image"], *, upscale: float = 1.6) -> "np.ndarray":
    img = _ensure_np(img_rgb)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if cv2 is not None else img.mean(axis=2).astype("uint8")
    else:
        gray = img
    gray = _remove_shadow(gray)
    gray = _deskew(gray)
    bin_img = _adaptive_binarize(gray)
    if len(bin_img.shape) == 2:
        bin_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB) if cv2 is not None else np.stack([bin_img] * 3, axis=-1)
    return _maybe_upscale(bin_img, upscale)

# ----------------- EasyOCR / Paddle OCR -----------------
_EASYOCR_CACHE: Dict[Tuple[str, bool, bool], "easyocr.Reader"] = {}
_PADDLE_CACHE: Dict[Tuple[str, bool, bool, bool, str], "PaddleOCR"] = {}

def _get_easyocr(lang: str = "en", handwriting: bool = False, *, gpu_env: Optional[bool] = None) -> Optional["easyocr.Reader"]:
    global easyocr, _HAS_EASYOCR
    if not _HAS_EASYOCR:
        try:
            import easyocr  # type: ignore
            _HAS_EASYOCR = True
        except Exception as e:
            log.warning(f"EasyOCR import failed: {e}")
            return None
    langs = [lang or "en"]
    gpu = bool(int(os.getenv("EASYOCR_GPU", "0"))) if gpu_env is None else gpu_env
    key = (langs[0], handwriting, gpu)
    if key in _EASYOCR_CACHE:
        return _EASYOCR_CACHE[key]
    recog = "english_g2" if lang.startswith("en") else ("latin_g2" if lang in {"fr", "es", "pt", "it"} else "english_g2")
    try:
        reader = easyocr.Reader(langs, gpu=gpu, recog_network=recog)
        _EASYOCR_CACHE[key] = reader
        log.info(f"EasyOCR ready (langs={langs}, gpu={gpu}, recog={recog})")
        return reader
    except Exception as e:
        log.warning(f"EasyOCR init failed: {e}")
        return None

def _get_paddle(lang: str = "en", *, angle: bool = False, gpu_env: Optional[bool] = None, textline: Optional[bool] = None) -> Optional["PaddleOCR"]:
    global PaddleOCR, _HAS_PADDLE
    if not _HAS_PADDLE:
        try:
            from paddleocr import PaddleOCR as _PaddleOCR
            PaddleOCR = _PaddleOCR
            _HAS_PADDLE = True
        except Exception as e:
            log.warning(f"Paddle import failed: {e}")
            return None
    gpu = bool(int(os.getenv("PADDLE_GPU", "0"))) if gpu_env is None else gpu_env
    textline = bool(int(os.getenv("PADDLE_TEXTLINE_ORI", "1"))) if textline is None else textline
    key = (lang, gpu, textline, angle, "v1")
    if key in _PADDLE_CACHE:
        return _PADDLE_CACHE[key]
    try:
        kwargs = dict(lang=lang, use_textline_orientation=textline, use_gpu=gpu, show_log=False)
        if angle:
            kwargs["use_angle_cls"] = True
        reader = PaddleOCR(**kwargs)
        _PADDLE_CACHE[key] = reader
        log.info(f"PaddleOCR ready (lang={lang}, gpu={gpu}, textline={textline}, angle={angle})")
        return reader
    except Exception as e:
        log.warning(f"Paddle init failed: {e}")
        return None

def get_paddle_ocr(lang: str = "en", use_gpu: bool = False) -> Optional["PaddleOCR"]:
    """Public warmup/helper for external modules."""
    return _get_paddle(lang=lang, angle=False, gpu_env=use_gpu, textline=None)

def _easy_read(reader: "easyocr.Reader", img_rgb: "np.ndarray", *, handwriting: bool) -> Tuple[List[str], List[float]]:
    decoder = "beamsearch" if handwriting else "greedy"
    try:
        results = reader.readtext(
            img_rgb,
            decoder=decoder,
            beamWidth=10 if handwriting else 5,
            detail=1,
            paragraph=False,
            width_ths=0.5 if handwriting else 0.7,
            height_ths=0.5 if handwriting else 0.7,
        )
    except Exception as e:
        log.warning(f"EasyOCR read failed: {e}")
        return [], []
    lines: List[Tuple[str, float, Tuple[float, float]]] = []
    for box, text, conf in results:
        try:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            lines.append((text, float(conf), (cx, cy)))
        except Exception:
            continue
    lines.sort(key=lambda t: (round(t[2][1] / 16.0), round(t[2][0] / 16.0)))
    return [t for (t, _c, _p) in lines], [float(c) for (_t, c, _p) in lines]

# ----------------- Gemini integrations -----------------
_GEMINI_SVC_SINGLETON = None
_GEMINI_SVC_LOADED = False

def _import_gemini_service_via_dotted(path_str: str):
    """
    Try normal dotted import (services.Gemini.gemini_service) after adding repo root
    to sys.path. Preserves relative imports inside gemini_service.py.
    """
    file_path = path_str.replace('.', '/') + '.py'
    p = Path(file_path).resolve()
    repo_root = None
    base_dir = None
    for anc in p.parents:
        if anc.name == "services":
            base_dir = anc
            repo_root = anc.parent
            break
    if repo_root is None or base_dir is None:
        return None, f"No 'services' directory found in ancestors of {p}"
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    dotted = ".".join(p.relative_to(repo_root).with_suffix("").parts)  # services.Gemini.gemini_service
    try:
        mod = importlib.import_module(dotted)
    except Exception as e:
        return None, f"import_module('{dotted}') failed: {e}"
    svc = getattr(mod, "GeminiService", None)
    if svc is None:
        return None, f"GeminiService class not found in module '{dotted}'"
    if not hasattr(svc, "ocr"):
        return None, f"GeminiService.ocr(...) missing in module '{dotted}'"
    return svc, None

def _import_gemini_service_via_pkgstub(path_str: str):
    """
    Robust loader: create package stubs so relative imports in the service work
    even when loading the file directly.
    """
    file_path = path_str.replace('.', '/') + '.py'
    p = Path(file_path).resolve()
    repo_root = None
    base_dir = None
    for anc in p.parents:
        if anc.name == "services":
            base_dir = anc
            repo_root = anc.parent
            break
    if repo_root is None or base_dir is None:
        return None, f"No 'services' directory found in ancestors of {p}"

    def _ensure_pkg(name: str, path_list: List[str]):
        pkg = sys.modules.get(name)
        if pkg is None:
            pkg = types.ModuleType(name)
            pkg.__path__ = path_list  # type: ignore[attr-defined]
            sys.modules[name] = pkg
        else:
            if not hasattr(pkg, "__path__"):
                pkg.__path__ = path_list  # type: ignore[attr-defined]
        return pkg

    _ensure_pkg("services", [str(base_dir)])
    _ensure_pkg("services.Gemini", [str(base_dir / "Gemini")])

    mod_name = "services.Gemini.gemini_service"
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if not spec or not spec.loader:
        return None, f"Cannot create spec for {p}"
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "services.Gemini"
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as e:
        sys.modules.pop(mod_name, None)
        return None, f"exec_module('{mod_name}') failed: {e}"

    svc = getattr(module, "GeminiService", None)
    if svc is None:
        return None, "GeminiService class not found in provided file"
    if not hasattr(svc, "ocr"):
        return None, "GeminiService.ocr(...) method not found"
    return svc, None

def _import_gemini_service_via_file(path_str: str):
    """Last resort raw import (works only if the service uses absolute imports)."""
    file_path = path_str.replace('.', '/') + '.py'
    p = Path(file_path)
    if not p.is_file():
        return None, f"Not a file: {file_path}"
    spec = importlib.util.spec_from_file_location("gemini_service_ext", str(p))
    if not spec or not spec.loader:
        return None, f"Cannot create import spec for: {file_path}"
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception as e:
        return None, f"exec_module failed: {e}"
    svc = getattr(mod, "GeminiService", None)
    if svc is None:
        return None, "GeminiService class not found in provided file"
    if not hasattr(svc, "ocr"):
        return None, "GeminiService.ocr(...) method not found"
    return svc, None

def _get_gemini_service():
    """Initialize your load-balanced GeminiService from GEMINI_SERVICE_PATH, else None."""
    global _GEMINI_SVC_SINGLETON, _GEMINI_SVC_LOADED

    # Check if we're in a multiprocessing context
    import multiprocessing
    if multiprocessing.current_process().name != 'MainProcess':
        # In worker processes, create a new instance instead of using singleton
        log.debug("Creating new GeminiService instance for worker process")
        return _create_gemini_service_instance()

    # Main process: use singleton pattern
    if _GEMINI_SVC_LOADED:
        return _GEMINI_SVC_SINGLETON
    _GEMINI_SVC_LOADED = True
    _GEMINI_SVC_SINGLETON = _create_gemini_service_instance()
    return _GEMINI_SVC_SINGLETON

def _create_gemini_service_instance():
    """Create a new GeminiService instance."""
    path_env = os.getenv("GEMINI_SERVICE_PATH", "").strip()
    candidate_paths = []
    if path_env:
        candidate_paths.append(path_env)
    # Always fall back to the built-in GeminiService inside this repository.
    candidate_paths.append("services.Gemini.gemini_service")

    svc_cls = None
    for candidate in candidate_paths:
        # 1) Try dotted import
        svc_cls, err = _import_gemini_service_via_dotted(candidate)
        if svc_cls is not None:
            path_env = candidate
            break
        log.error(f"GeminiService dotted import failed: {err}")
        # 2) Try package-stubbed import
        svc_cls, err2 = _import_gemini_service_via_pkgstub(candidate)
        if svc_cls is not None:
            path_env = candidate
            break
        log.error(f"GeminiService package-stub import failed: {err2}")
        # 3) Raw file import
        svc_cls, err3 = _import_gemini_service_via_file(candidate)
        if svc_cls is not None:
            path_env = candidate
            break
        log.error(
            f"Failed to load GeminiService from {candidate}: {err3}"
        )

    if svc_cls is None:
        log.info(
            "No GeminiService available; will use direct google-genai fallback "
            "(if GEMINI_API_KEY present)."
        )
        return None

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    try:
        instance = svc_cls(model=model)
        log.info(f"GeminiService instance ready (model={model}) from {path_env}")
        return instance
    except Exception as e:
        log.error(f"GeminiService init failed: {e}")
        return None

def _gemini_via_service(img_pils: List["Image.Image"]) -> List[str]:
    svc = _get_gemini_service()
    if svc is None:
        return ["" for _ in img_pils]

    max_edge = int(os.getenv("OCR_GEMINI_MAX_EDGE", "2000"))
    texts: List[str] = []

    for i in range(0, len(img_pils), 3):
        chunk_imgs: List[Dict[str, bytes]] = []
        for img_pil in img_pils[i:i + 3]:
            if max_edge > 0:
                img_pil = _limit_long_edge(img_pil, max_edge)
            with io.BytesIO() as buf:
                img_pil.save(buf, format="PNG")
                chunk_imgs.append({"mime_type": "image/png", "data": buf.getvalue()})
            try:
                img_pil.close()
            except Exception:
                pass

        prompt = _OCR_PROMPT_BATCH if len(chunk_imgs) > 1 else _OCR_PROMPT_SINGLE

        attempt = 0
        while attempt < 3:
            try:
                out = svc.ocr(images=chunk_imgs, prompt=prompt, response_model=None)
                if isinstance(out, dict) and "result" in out:
                    text_block = out["result"] or ""
                elif isinstance(out, str):
                    text_block = out
                else:
                    try:
                        text_block = str(out)
                    except Exception:
                        text_block = ""

                if len(chunk_imgs) == 1:
                    texts.append(text_block.strip())
                else:
                    pages: List[str] = []
                    pattern = r"Page\s*\d+\s*:\s*(.*?)\s*(?=Page\s*\d+\s*:|$)"
                    for m in re.finditer(pattern, text_block, re.DOTALL):
                        pages.append(m.group(1).strip())
                    if not pages:
                        splits = text_block.split("Page ")
                        for s in splits[1:]:
                            pages.append(s.split(":", 1)[-1].strip())
                    while len(pages) < len(chunk_imgs):
                        pages.append("")
                    texts.extend(pages[: len(chunk_imgs)])
                break
            except Exception as e:
                attempt += 1
                log.error(f"GeminiService.ocr failed (attempt {attempt}): {e}")
                time.sleep(min(2 ** attempt, 8))
        else:
            texts.extend(["" for _ in chunk_imgs])

        del chunk_imgs
        gc.collect()

    return texts

def _gemini_via_fallback(img_pil: "Image.Image") -> str:
    """Direct google-genai fallback (requires GEMINI_API_KEY)."""
    try:
        from google import genai
        from google.genai import types as gtypes
    except Exception:
        log.error("google-genai not installed. `pip install google-genai`")
        return ""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        log.error("GOOGLE_API_KEY or GEMINI_API_KEY not set and no GEMINI_SERVICE_PATH usable.")
        return ""
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    max_tokens = int(os.getenv("OCR_GEMINI_MAX_TOKENS", "8192"))
    max_edge = int(os.getenv("OCR_GEMINI_MAX_EDGE", "2000"))
    if max_edge > 0:
        img_pil = _limit_long_edge(img_pil, max_edge)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    try:
        client = genai.Client(api_key=api_key)
        image_part = gtypes.Part.from_bytes(mime_type="image/png", data=img_bytes)
        resp = client.models.generate_content(
            model=model,
            contents=[image_part, _OCR_PROMPT_SINGLE],
            config={
                "temperature": 0.0,
                "max_output_tokens": max_tokens
                },
        )
        text = (getattr(resp, "text", None) or getattr(resp, "output_text", "") or "").strip()
        if not text:
            try:
                parts_text = []
                for cand in (getattr(resp, "candidates", []) or []):
                    content = getattr(cand, "content", None)
                    if content and getattr(content, "parts", None):
                        for p in content.parts:
                            t = getattr(p, "text", None)
                            if t:
                                parts_text.append(t)
                text = "\n".join(parts_text).strip()
            except Exception:
                pass
        return text
    except Exception as e:
        log.error(f"Gemini fallback failed: {e}")
        return ""

# ----------------- Public OCR APIs -----------------
def ocr_image(
    img_rgb: Union["np.ndarray", "Image.Image"],
    *,
    lang: str = "en",
    handwriting: bool = False,
    engine: str = "hybrid",  # "easyocr" | "paddleocr" | "hybrid" | "gemini"
    dpi_hint: Optional[int] = None,
    return_debug: bool = False,
) -> OCRResult:
    assert np is not None, "numpy required"
    engine = (engine or os.getenv("OCR_ENGINE", "hybrid")).lower()
    handwriting = bool(int(os.getenv("OCR_HANDWRITTEN", "1" if handwriting else "0")))
    conf_th = float(os.getenv("OCR_HANDWRITTEN_CONF_TH" if handwriting else "OCR_CONF_TH", "0.15" if handwriting else "0.35"))
    min_lines = int(os.getenv("OCR_MIN_LINES", "5"))
    min_avg = float(os.getenv("OCR_MIN_AVG_CONF", "0.50"))
    upscale = float(os.getenv("OCR_UPSCALE", "1.6")) if handwriting else 1.0

    debug_imgs: List["np.ndarray"] = []

    # Preprocess for Easy/Paddle. For Gemini, we prefer the raw page image.
    pre_for_easy = preprocess_for_handwriting(img_rgb, upscale=upscale) if handwriting else _ensure_np(img_rgb)
    if return_debug and engine != "gemini":
        debug_imgs.append(pre_for_easy.copy())

    # --- Gemini path ---
    if engine == "gemini":
        pil_img = _to_pil(img_rgb)
        text_list = _gemini_via_service([pil_img])
        text = text_list[0] if text_list else ""
        if not text:
            text = _gemini_via_fallback(pil_img)

        if not text.strip():
            # Optional auto-fallback to a local engine
            if bool(int(os.getenv("OCR_GEMINI_AUTOFALLBACK", "1"))):
                fb_engine = os.getenv("OCR_GEMINI_FALLBACK_ENGINE", "hybrid").lower()
                fb_hand = bool(int(os.getenv("OCR_GEMINI_FALLBACK_HAND", "1")))
                if fb_engine not in ("easyocr", "paddleocr", "hybrid"):
                    fb_engine = "hybrid"
                log.warning(f"Gemini returned empty text; falling back to {fb_engine} (hand={int(fb_hand)})")
                return ocr_image(
                    img_rgb,
                    lang=lang,
                    handwriting=fb_hand,
                    engine=fb_engine,      # NOT gemini (avoid recursion)
                    dpi_hint=dpi_hint,
                    return_debug=return_debug,
                )
            # If no fallback, fail loudly
            raise RuntimeError("Gemini OCR returned empty text (check GEMINI_SERVICE_PATH or GEMINI_API_KEY).")

        lines = text.splitlines()
        metrics = [PageMetrics(page=1, engine="gemini", lines=len(lines),
                               avg_conf=0.0, min_conf=0.0, max_conf=0.0, dpi=dpi_hint)]
        return OCRResult(
            text="\n".join(lines),
            by_page=metrics,
            meta={"engine": "gemini", "handwriting": handwriting},
            debug_images=None
        )

    # --- EasyOCR first (if selected or hybrid) ---
    texts: List[str] = []
    confs: List[float] = []
    used_engine = engine

    if engine in ("easyocr", "hybrid"):
        reader = _get_easyocr(lang=lang, handwriting=handwriting)
        if reader is not None:
            t0 = time.time()
            texts, confs = _easy_read(reader, pre_for_easy, handwriting=handwriting)
            dt = time.time() - t0
            avg = (sum(confs) / len(confs)) if confs else 0.0
            log.info(f"EasyOCR lines={len(texts)} avg={avg:.2f} in {dt:.2f}s")
            used_engine = "easyocr"

    # --- Hybrid fallback with Paddle (if needed) ---
    avg = (sum(confs) / len(confs)) if confs else 0.0
    if engine in ("hybrid", "paddleocr") and (len(texts) < min_lines or avg < min_avg):
        for angle_flag in (False, True):
            ocr = _get_paddle(lang=lang, angle=angle_flag)
            if ocr is None:
                continue
            try:
                res = ocr.ocr(pre_for_easy)
                flat: List[Tuple[str, float]] = []
                for blk in res or []:
                    for line in blk or []:
                        try:
                            _box, (tx, cf) = line
                            flat.append((tx, float(cf)))
                        except Exception:
                            continue
                p_texts = [t for (t, c) in flat if t and c >= conf_th]
                p_confs = [c for (t, c) in flat if t and c >= conf_th]
                p_avg = (sum(p_confs) / len(p_confs)) if p_confs else 0.0
                if len(p_texts) > len(texts) or p_avg > avg:
                    texts, confs = p_texts, p_confs
                    used_engine = f"paddleocr(angle_cls={angle_flag})"
                    avg = p_avg
                if texts and (len(texts) >= min_lines and avg >= min_avg):
                    break
            except Exception as e:
                log.warning(f"Paddle pass failed: {e}")

    page_metrics = [PageMetrics(
        page=1, engine=used_engine, lines=len(texts),
        avg_conf=(sum(confs)/len(confs)) if confs else 0.0,
        min_conf=min(confs) if confs else 0.0,
        max_conf=max(confs) if confs else 0.0,
        dpi=dpi_hint
    )]

    return OCRResult(
        text="\n".join(texts),
        by_page=page_metrics,
        meta={"engine": used_engine, "handwriting": handwriting},
        debug_images=(debug_imgs if return_debug else None)
    )

def _open_pdf(pdf_path: Path) -> "fitz.Document":
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF OCR")
    return fitz.open(str(pdf_path))

def ocr_pdf(
    pdf_path: Union[str, Path],
    *,
    lang: str = "en",
    handwriting: bool = False,
    dpi: int = 300,
    engine: str = "hybrid",  # include "gemini"
    return_debug: bool = False,
) -> OCRResult:
    doc = _open_pdf(Path(pdf_path))

    pages_text: List[str] = []
    metrics: List[PageMetrics] = []
    debug_imgs: List["np.ndarray"] = []

    pending_imgs: List["Image.Image"] = []
    pending_idx: List[int] = []

    for i in range(len(doc)):
        pg = doc.load_page(i)
        zoom = max(1.0, dpi / 72.0)
        pm = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

        # --- Gemini path for PDF ---
        if engine == "gemini":
            if Image is None:
                raise RuntimeError("Pillow required for Gemini OCR")
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            pending_imgs.append(img)
            pending_idx.append(i)

            if len(pending_imgs) == 3 or i == len(doc) - 1:
                batch_texts = _gemini_via_service(pending_imgs)
                for j, (idx, page_text) in enumerate(zip(pending_idx, batch_texts)):
                    if not page_text.strip():
                        if bool(int(os.getenv("OCR_GEMINI_AUTOFALLBACK", "1"))):
                            fb_engine = os.getenv("OCR_GEMINI_FALLBACK_ENGINE", "hybrid").lower()
                            fb_hand = bool(int(os.getenv("OCR_GEMINI_FALLBACK_HAND", "1")))
                            if fb_engine not in ("easyocr", "paddleocr", "hybrid"):
                                fb_engine = "hybrid"
                            log.warning(f"[p{idx+1}] Gemini empty; falling back to {fb_engine} (hand={int(fb_hand)})")

                            rgb = np.array(pending_imgs[j])
                            fb = ocr_image(
                                rgb,
                                lang=lang,
                                handwriting=fb_hand,
                                engine=fb_engine,
                                dpi_hint=dpi,
                                return_debug=return_debug,
                            )
                            pages_text.append(fb.text)
                            metrics.append(PageMetrics(
                                page=idx + 1,
                                engine=f"gemini→{fb.by_page[0].engine}",
                                lines=len(fb.text.splitlines()) if fb.text else 0,
                                avg_conf=fb.by_page[0].avg_conf,
                                min_conf=fb.by_page[0].min_conf,
                                max_conf=fb.by_page[0].max_conf,
                                dpi=dpi,
                            ))
                        else:
                            raise RuntimeError("Gemini OCR returned empty text (page).")
                    else:
                        pages_text.append(page_text)
                        metrics.append(PageMetrics(
                            page=idx + 1,
                            engine="gemini",
                            lines=len(page_text.splitlines()),
                            avg_conf=0.0,
                            min_conf=0.0,
                            max_conf=0.0,
                            dpi=dpi,
                        ))
                pending_imgs = []
                pending_idx = []
                gc.collect()
            continue

        # --- Non-Gemini: route via ocr_image() for consistency ---
        if cv2 is not None:
            arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)
            if pm.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            rgb = arr
        else:
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            rgb = np.array(img)

        single = ocr_image(
            rgb, lang=lang, handwriting=handwriting,
            engine=engine, dpi_hint=dpi, return_debug=return_debug
        )
        pages_text.append(single.text)
        m = single.by_page[0]
        m.page = i + 1
        metrics.append(m)
        if return_debug and single.debug_images:
            debug_imgs.extend(single.debug_images)

    full = "\n".join(pages_text)
    meta = {"engine": engine, "handwriting": handwriting, "dpi": dpi, "pages": len(doc)}
    return OCRResult(text=full, by_page=metrics, meta=meta, debug_images=(debug_imgs if return_debug else None))

# ----------------- CLI -----------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Standalone OCR for images/PDFs (Gemini/EasyOCR/Paddle)")
    p.add_argument("inputs", nargs="+", help="File(s) or folder(s) to OCR")
    p.add_argument("-o", "--outdir", default="ocr_out", help="Output directory")
    p.add_argument("--hand", action="store_true", help="Handwriting mode (preprocessing for Easy/Paddle)")
    p.add_argument("--lang", default=os.getenv("PADDLE_LANG", "en"), help="Language (e.g., en, fr)")
    p.add_argument("--engine", default=os.getenv("OCR_ENGINE", "gemini"),
                   choices=["easyocr", "paddleocr", "hybrid", "gemini"],
                   help="OCR engine to use")
    p.add_argument("--dpi", type=int, default=300, help="PDF render DPI (also affects Gemini page raster)")
    p.add_argument("--save-debug", action="store_true", help="Save preprocessed images (non-Gemini)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dbgdir = outdir / "debug"
    if args.save_debug:
        dbgdir.mkdir(exist_ok=True)

    exts_img = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    targets: List[Path] = []
    for raw in args.inputs:
        pth = Path(raw)
        if pth.is_dir():
            targets += sorted(pth.rglob("*.pdf"))
            for ext in exts_img:
                targets += sorted(pth.rglob(f"*{ext}"))
        else:
            targets.append(pth)

    if not targets:
        raise SystemExit("No inputs found.")

    for src in targets:
        stem = src.stem
        if src.suffix.lower() == ".pdf":
            try:
                res = ocr_pdf(
                    src, lang=args.lang, handwriting=args.hand,
                    dpi=args.dpi, engine=args.engine, return_debug=args.save_debug
                )
                (outdir / f"{stem}.txt").write_text(res.text, encoding="utf-8")
                (outdir / f"{stem}.metrics.json").write_text(
                    json.dumps([m.__dict__ for m in res.by_page], ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                if args.save_debug and res.debug_images and cv2 is not None:
                    for idx, img in enumerate(res.debug_images, start=1):
                        cv2.imwrite(str(dbgdir / f"{stem}_p{idx}.png"),
                                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                log.info(f"PDF OK: {src.name}  chars={len(res.text):,}  preview='{_snapshot(res.text)}'")
            except Exception as e:
                log.error(f"PDF FAIL: {src} -> {e}")
        else:
            try:
                if Image is None:
                    raise RuntimeError("Pillow is required to open images")
                im = Image.open(src)
                res = ocr_image(
                    im, lang=args.lang, handwriting=args.hand,
                    engine=args.engine, dpi_hint=None, return_debug=args.save_debug
                )
                (outdir / f"{stem}.txt").write_text(res.text, encoding="utf-8")
                (outdir / f"{stem}.metrics.json").write_text(
                    json.dumps([m.__dict__ for m in res.by_page], ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                if args.save_debug and res.debug_images and cv2 is not None and res.debug_images:
                    cv2.imwrite(str(dbgdir / f"{stem}.png"),
                                cv2.cvtColor(res.debug_images[0], cv2.COLOR_RGB2BGR))
                log.info(f"IMG OK: {src.name}  chars={len(res.text):,}  preview='{_snapshot(res.text)}'")
            except Exception as e:
                log.error(f"IMG FAIL: {src} -> {e}")

    print(f"Saved outputs to: {outdir}")
