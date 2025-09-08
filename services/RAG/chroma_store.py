from pathlib import Path
from typing import Any, Dict
import json
from log_utils import get_logger
from chromadb.config import Settings
import chromadb

log = get_logger("chroma")


def chroma_client(persist_dir: str):
    try:
        client = chromadb.PersistentClient(path=str(Path(persist_dir)))
        log.info(f"[Chroma] PersistentClient at {persist_dir}")
        return client
    except Exception as e:
        log.warning(f"[Chroma] PersistentClient failed: {e}; falling back to in-process client")
        return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(Path(persist_dir))))


def _sanitize_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (str, int, float, bool)) or x is None else str(x) for x in v]
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out


def chroma_upsert_jsonl(jsonl_path: Path, collection, client, batch: int = 128) -> int:
    ids, docs, metas, embs = [], [], [], []
    n_added = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            md = rec.get("metadata", {})
            if md.get("is_duplicate") or md.get("skip_index"):
                continue
            ids.append(rec["id"])
            docs.append(rec["text"])
            metas.append(_sanitize_meta(md))
            embs.append(rec["embedding"])
            if len(ids) >= batch:
                n_added += _safe_upsert(collection, ids, docs, metas, embs)
                ids, docs, metas, embs = [], [], [], []
    if ids:
        n_added += _safe_upsert(collection, ids, docs, metas, embs)
    # Persist only if supported by the client (older Chroma clients). Newer
    # chromadb PersistentClient persists automatically and exposes no 'persist'.
    try:
        if hasattr(client, "persist"):
            client.persist()  # type: ignore[attr-defined]
    except Exception as e:
        log.warning(f"[Chroma] persist failed: {e}")
    log.info(f"[Chroma] upserted={n_added} from {jsonl_path.name}")
    return n_added


def _safe_upsert(collection, ids, docs, metas, embs) -> int:
    try:
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        return len(ids)
    except Exception as e:
        log.warning(f"[Chroma] batch upsert failed ({len(ids)}): {e}; retrying halves")
        # Retry smaller batches
        n = 0
        size = max(1, len(ids) // 2)
        i = 0
        while i < len(ids):
            j = min(i + size, len(ids))
            try:
                collection.upsert(ids=ids[i:j], documents=docs[i:j], metadatas=metas[i:j], embeddings=embs[i:j])
                n += (j - i)
            except Exception as e2:
                log.error(f"[Chroma] sub-batch failed range={i}:{j} err={e2}")
            i = j
        return n

