from __future__ import annotations
import logging
import re
from pathlib import Path

import fitz

from data_models.ocr_data_model import OCRData
from services.Gemini.gemini_service import GeminiService
from utils.Caching.cache import Cache

ACCEPTABLE_TEXT_PERCENTAGE = 0.85
CACHE_FILE = "pdf_cache.json"
OCR_PROMPT = (
    "Extract all text from this document image. "
    "Pay special attention to LaTeX and engineering notation. "
    "Use step-by-step reasoning for accuracy. "
    "Respond as JSON in the form {\"text\": \"<extracted text>\"}."
)


def open_pdf(file_path: str) -> fitz.Document:
    """Open a PDF file and return the document object."""
    return fitz.open(str(file_path))


def is_image_focused(file_path: str, text_threshold: int = 100,
                     cache: Cache | None = None) -> bool:
    """Determine if a PDF is primarily composed of image-only pages.

    ``text_threshold`` defines the minimum length of text for a page to be
    considered text-based. A page is considered "image only" when the length of
    its extracted text (``len(page.get_text().strip())``) is below this
    threshold and the page contains at least one image. If more than
    ``ACCEPTABLE_TEXT_PERCENTAGE`` (85% by default) of pages are image-only, the
    PDF is treated as requiring OCR.
    """
    cache_key = f"analysis_{Path(file_path).stem}"
    doc = open_pdf(file_path)
    total_pages = len(doc)

    image_only_pages = 0
    for page in doc:
        text_length = len(page.get_text().strip())
        has_image = bool(page.get_images(full=True))
        if text_length < text_threshold and has_image:
            image_only_pages += 1

    ratio = image_only_pages / total_pages if total_pages else 0
    result = ratio >= ACCEPTABLE_TEXT_PERCENTAGE

    if cache:
        try:
            analysis_data = {
                "file_path": str(file_path),
                "requires_ocr": result,
                "total_pages": total_pages,
                "image_only_pages": image_only_pages,
                "ratio": ratio,
                "timestamp": str(Path(file_path).stat().st_mtime),
            }
            cache.update_cache(cache_key, analysis_data)
        except Exception as e:
            logging.warning(f"Failed to cache analysis results: {e}")

    return result


def ocr_text_extraction(
    file_path: str,
    gemini_service: GeminiService,
    cache: Cache | None = None,
    *,
    combine_pages: bool = False,
) -> str:
    """Perform OCR on a PDF and cache the results.

    If ``combine_pages`` is True the returned text will contain no page break markers,
    allowing downstream paragraph-based chunking.
    """
    cache_key = Path(file_path).stem

    if cache:
        try:
            cached = cache.read_cache()
            if cache_key in cached and "pages" in cached[cache_key]:
                pages = cached[cache_key]["pages"]
                print(f"Pages: {pages}")
                if pages and all(str(i + 1) in pages for i in range(len(pages))):
                    return "\n\n--- PAGE BREAK ---\n\n".join(
                        pages[str(i + 1)]["text"] for i in range(len(pages))
                    )
        except Exception as e:
            logging.warning(f"Cache check failed for OCR: {e}")

    doc = open_pdf(file_path)
    images = [
        {"mime_type": "image/png", "data": page.get_pixmap(dpi=400).tobytes("png")}
        for page in doc
    ]

    pages_data = {}
    for idx, img in enumerate(images):
        page_num = str(idx + 1)
        print(f"Page number: {page_num}")
        text = ""
        try:
            response = gemini_service.ocr(
                [img], prompt=OCR_PROMPT, response_model=OCRData
            )
            text = response.text
        except Exception as e:
            logging.error(f"OCR failed for page {page_num}: {e}")
        pages_data[page_num] = {
            "text": text,
            "text_length": len(text),
        }

    if cache:
        try:
            cached = cache.read_cache()
            if cache_key not in cached:
                cached[cache_key] = {"metadata": {}, "pages": {}}
            cached[cache_key]["pages"].update(pages_data)
            cached[cache_key]["metadata"].update({
                "method": "ocr",
                "processing_timestamp": str(Path(file_path).stat().st_mtime),
                "total_chars": sum(len(p["text"]) for p in pages_data.values()),
                "ocr_pages_processed": len(pages_data),
            })
            cache.update_cache(cache_key, cached[cache_key])
        except Exception as e:
            logging.warning(f"Failed to cache OCR results: {e}")

    joined = "\n\n--- PAGE BREAK ---\n\n".join(
        pages_data[str(i + 1)]["text"] for i in range(len(images))
    )
    if combine_pages:
        joined = re.sub(r"\n?-+ PAGE BREAK -+\n?", "\n\n", joined)
    return joined
