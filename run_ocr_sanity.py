"""Simple OCR runner using the load-balanced Gemini service."""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path
from typing import Iterable, List

# Expose repository as a package so GeminiService's relative imports resolve.
_repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_root))
_pkg = types.ModuleType("COURSEGEN")
_pkg.__path__ = [str(_repo_root)]
sys.modules.setdefault("COURSEGEN", _pkg)

from services.Gemini.gemini_service import GeminiService  # noqa: E402

_gemini = GeminiService()


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}


def _iter_sources(path: Path) -> Iterable[Path]:
    """Yield image or PDF files from *path* preserving sort order."""

    if path.is_file():
        yield path
        return

    for pth in sorted(
        p
        for p in path.iterdir()
        if p.suffix.lower() in IMG_EXTS or p.suffix.lower() in PDF_EXTS
    ):
        yield pth


def _ocr_lines(image: Path) -> List[str]:
    """Return recognized text lines for an image via Gemini OCR."""

    from PIL import Image  # Imported lazily for a friendlier error if missing

    buf = io.BytesIO()
    Image.open(image).save(buf, format="PNG")
    resp = _gemini.ocr(
        [{"mime_type": "image/png", "data": buf.getvalue()}], prompt=""
    )
    text = resp.get("result", "") if isinstance(resp, dict) else ""
    print(text)
    return text.splitlines() if text else []


def _ocr_pdf(pdf_path: Path) -> List[str]:
    """Return recognized text lines for a PDF via Gemini OCR."""

    from services.RAG.ocr_engine import (  # noqa: E402
        ocr_pdf as _ocr_pdf_engine,
    )

    res = _ocr_pdf_engine(pdf_path=pdf_path, engine="gemini")
    text = getattr(res, "text", "")
    print(text)
    return text.splitlines() if text else []


def main(target: Path) -> None:
    for src_path in _iter_sources(target):
        if src_path.suffix.lower() in IMG_EXTS:
            lines = _ocr_lines(src_path)
        elif src_path.suffix.lower() in PDF_EXTS:
            lines = _ocr_pdf(src_path)
        else:
            continue
        preview = " ".join(lines[:5])  # keep output manageable
        print(f"{src_path.name}: {preview}")


if __name__ == "__main__":
    import sys as _sys

    input_arg = (
        Path(_sys.argv[1])
        if len(_sys.argv) > 1
        else Path(
            "data/textbooks/COMPILATION/EEE/400/1/PQ/",
            "WhatsApp Image 2023-11-19 at 13.37.17_46555d24.jpg",
        )
    )
    main(input_arg)
