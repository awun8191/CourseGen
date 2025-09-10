import os
import json
from pathlib import Path
from typing import Any, Dict
from services.RAG.log_utils import get_logger

log = get_logger("progress")


def safe_file_replace(src_path: Path, dest_path: Path) -> None:
    try:
        if dest_path.exists():
            backup_path = dest_path.with_suffix(dest_path.suffix + ".backup")
            if backup_path.exists():
                backup_path.unlink()
            dest_path.rename(backup_path)
        src_path.rename(dest_path)
        backup_path = dest_path.with_suffix(dest_path.suffix + ".backup")
        if backup_path.exists():
            try:
                backup_path.unlink()
            except Exception:
                pass
    except Exception as e:
        try:
            dest_path.write_bytes(src_path.read_bytes())
            src_path.unlink()
        except Exception as inner_e:
            log.error(f"[PROG] Failed to replace {dest_path}: {e}, fallback failed: {inner_e}")
            raise


def load_progress(p: Path) -> Dict[str, Any]:
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            log.info(f"[PROG] loaded {p} with {len(data.get('files', {}))} entries")
            return data
        except Exception as e:
            log.warning(f"[PROG] load failed {p}: {e}")
    return {"version": "simple-1", "files": {}}


def save_progress(p: Path, st: Dict[str, Any]) -> None:
    try:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
        safe_file_replace(tmp, p)
        log.info(f"[PROG] saved to {p} entries={len(st.get('files', {}))}")
    except Exception as e:
        log.warning(f"[PROG] save failed {p}: {e}")


def should_skip(rec: Dict[str, Any], cur_size: int, cur_mtime: int) -> bool:
    # Fast path: check status first (most common case)
    if rec.get("status") != "completed":
        return False
    
    # Check file hasnt changed (size and mtime)
    if rec.get("file_size") != cur_size or rec.get("file_mtime") != cur_mtime:
        return False
    
    # Check processing completed successfully
    return rec.get("jsonl_archived") and rec.get("chroma_upserted")

