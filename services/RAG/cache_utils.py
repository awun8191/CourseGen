import os
import hashlib
from pathlib import Path
from typing import Optional
from COURSEGEN.services.RAG.log_utils import get_logger

log = get_logger("cache")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _key_text(file_hash: str, lang: str) -> str:
    return f"{file_hash}.text.{lang}.txt.cache"


def _key_ocr(file_hash: str, lang: str, dpi: int, policy: str) -> str:
    return f"{file_hash}.ocr.{lang}.dpi{dpi}.{policy}.cache"


def try_read(cache_dir: Path, key: str) -> Optional[str]:
    cpath = cache_dir / key
    if cpath.exists():
        try:
            txt = cpath.read_text(encoding="utf-8", errors="ignore")
            log.info(f"[CACHE] hit key={key} size={len(txt)} path={cpath}")
            return txt
        except Exception as e:
            log.warning(f"[CACHE] read-failed key={key} err={e}")
    else:
        log.debug(f"[CACHE] miss key={key}")
    return None


def write(cache_dir: Path, key: str, text: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / key
    try:
        cpath.write_text(text, encoding="utf-8")
        log.info(f"[CACHE] wrote key={key} size={len(text)} path={cpath}")
    except Exception as e:
        log.warning(f"[CACHE] write-failed key={key} err={e}")


__all__ = [
    "sha256_file",
    "_key_text",
    "_key_ocr",
    "try_read",
    "write",
]

