import logging
import sys
import json
from typing import Optional

_LOGGER_INITIALIZED = False

def setup_logging(level: str = "DEBUG", log_file: Optional[str] = None) -> None:
    """Configure root logging with rich, descriptive format.
    Safe to call multiple times.
    """
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return

    lvl = getattr(logging, level.upper(), logging.DEBUG)

    fmt = (
        "[%(asctime)s %(levelname)s %(processName)s %(threadName)s "
        "%(name)s:%(lineno)d] %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=lvl, format=fmt, datefmt=datefmt, handlers=handlers)
    _LOGGER_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def snapshot(text: str, n: int = 240) -> str:
    """Return a compact, single-line snapshot of text for logs."""
    if text is None:
        return ""
    s = text.replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


def log_kv(logger: logging.Logger, event: str, **kwargs) -> None:
    """Emit a structured key-value log event as JSON in the message."""
    payload = {"event": event, **kwargs}
    try:
        logger.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        # Fallback to simple str if non-serializable
        logger.info(f"{event} | {kwargs}")

