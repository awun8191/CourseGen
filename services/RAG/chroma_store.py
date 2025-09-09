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
        log.warning(
            "[Chroma] PersistentClient failed: %s; "
            "falling back to in-process client",
            e,
        )
        return chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(Path(persist_dir)),
            )
        )


def _sanitize_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [
                (
                    x
                    if isinstance(x, (str, int, float, bool)) or x is None
                    else str(x)
                )
                for x in v
            ]
            if any(
                not isinstance(
                    x,
                    (str, int, float, bool),
                )
                and x is not None
                for x in v
            ):
                log.debug(
                    "[Chroma] metadata key '%s' list contains non-primitive "
                    "types; coerced to strings",
                    k,
                )
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
            log.debug(
                "[Chroma] metadata key '%s' serialized to JSON string",
                k,
            )
    return out


def chroma_upsert_jsonl(
    jsonl_path: Path, collection, client, batch: int = 128
) -> int:
    ids, docs, metas, embs = [], [], [], []
    n_added = 0
    n_skipped = 0
    log.info("[Chroma] loading %s batch_size=%s", jsonl_path.name, batch)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            md = rec.get("metadata", {})
            if md.get("is_duplicate") or md.get("skip_index"):
                n_skipped += 1
                log.debug(
                    "[Chroma] skipping id=%s duplicate=%s skip_index=%s",
                    rec.get("id"),
                    md.get("is_duplicate"),
                    md.get("skip_index"),
                )
                continue
            ids.append(rec["id"])
            docs.append(rec["text"])
            metas.append(_sanitize_meta(md))
            embs.append(rec["embedding"])
            if len(ids) >= batch:
                log.info(
                    "[Chroma] upserting batch size=%s line=%s",
                    len(ids),
                    line_no,
                )
                n_added += _safe_upsert(collection, ids, docs, metas, embs)
                ids, docs, metas, embs = [], [], [], []
    if ids:
        log.info("[Chroma] upserting final batch size=%s", len(ids))
        n_added += _safe_upsert(collection, ids, docs, metas, embs)
    # Persist only if supported by the client (older Chroma clients).
    # Newer chromadb PersistentClient persists automatically and exposes no
    # 'persist'.
    try:
        if hasattr(client, "persist"):
            client.persist()  # type: ignore[attr-defined]
    except Exception as e:
        log.warning("[Chroma] persist failed: %s", e)
    log.info(
        "[Chroma] upserted=%s skipped=%s from %s",
        n_added,
        n_skipped,
        jsonl_path.name,
    )
    return n_added


def _safe_upsert(collection, ids, docs, metas, embs) -> int:
    try:
        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embs,
        )
        log.debug("[Chroma] batch upsert succeeded size=%s", len(ids))
        return len(ids)
    except Exception as e:
        log.warning(
            "[Chroma] batch upsert failed (%s): %s; retrying halves",
            len(ids),
            e,
        )
        # Retry smaller batches
        n = 0
        size = max(1, len(ids) // 2)
        i = 0
        while i < len(ids):
            j = min(i + size, len(ids))
            try:
                collection.upsert(
                    ids=ids[i:j],
                    documents=docs[i:j],
                    metadatas=metas[i:j],
                    embeddings=embs[i:j],
                )
                n += j - i
            except Exception as e2:
                log.error(
                    "[Chroma] sub-batch failed range=%s:%s err=%s",
                    i,
                    j,
                    e2,
                )
            i = j
        return n
