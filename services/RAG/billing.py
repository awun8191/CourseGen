import os
import json
from pathlib import Path
from typing import Any, Dict, Tuple
from services.RAG.log_utils import get_logger

log = get_logger("billing")

class Billing:
    """Simple token usage and cost tracker persisted to JSON.
    - Windows-safe atomic writes with backup
    - Verbose logging on every update
    """
    def __init__(self, file: Path) -> None:
        self.file = file
        self.enabled = os.getenv("BILLING_ENABLED", "1") == "1"
        try:
            self.price_per_m = float(os.getenv("CF_PRICE_PER_M_TOKENS", "0.012"))
        except Exception:
            self.price_per_m = 0.012
        self.state: Dict[str, Any] = {
            "model": "@cf/baai/bge-m3",
            "price_per_m": self.price_per_m,
            "totals": {"tokens": 0, "cost": 0.0, "batches": 0, "files": 0},
            "files": {}
        }
        if self.file.exists():
            try:
                self.state = json.loads(self.file.read_text(encoding="utf-8"))
                log.info(f"[Billing] Loaded state from {self.file} :: totals={self.state.get('totals')}")
            except Exception as e:
                log.warning(f"[Billing] Failed to read existing state: {e}")

    def _atomic_write(self) -> None:
        try:
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
            if self.file.exists():
                backup = self.file.with_suffix(".backup")
                if backup.exists():
                    backup.unlink()
                self.file.rename(backup)
            tmp.rename(self.file)
            backup = self.file.with_suffix(".backup")
            if backup.exists():
                try:
                    backup.unlink()
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"[Billing] Failed to persist state: {e}")

    def add(self, src: str, new_tokens: int) -> Tuple[int, float]:
        if not self.enabled:
            return (0, 0.0)
        rec = self.state["files"].setdefault(
            src, {"tokens": 0, "cost": 0.0, "batches": 0}
        )
        rec["tokens"] += new_tokens
        rec["batches"] += 1
        inc_cost = (new_tokens / 1_000_000.0) * self.price_per_m
        rec["cost"] += inc_cost

        t = self.state["totals"]
        t["tokens"] += new_tokens
        t["cost"] += inc_cost
        t["batches"] += 1
        t["files"] = len(self.state["files"])

        self._atomic_write()
        log.info(f"[Billing] {Path(src).name}: +{new_tokens} tokens | totals={t} | file={rec}")
        return rec["tokens"], rec["cost"]

