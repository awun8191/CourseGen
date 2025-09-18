"""Rebuild a Chroma collection from progress report exports."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import chromadb


logger = logging.getLogger("chroma_revive")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROGRESS_STATE = REPO_ROOT / "OUTPUT_DATA2/progress_report/progress_state.json"
DEFAULT_PROGRESS_DIR = DEFAULT_PROGRESS_STATE.parent
DEFAULT_PERSIST_DIR = REPO_ROOT / "OUTPUT_DATA2/emdeddings"
DEFAULT_COLLECTION = "course_embeddings"


def _iter_progress_entries(progress_state: Path) -> Iterator[Dict[str, str]]:
    with progress_state.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    files = payload.get("files") or {}
    for src_path, info in files.items():
        if not isinstance(info, dict):
            continue
        jsonl_name = info.get("jsonl_name")
        if not jsonl_name:
            continue
        if info.get("status") not in {"completed", "complete", "done"}:
            continue
        yield {
            "source_path": src_path,
            "jsonl_name": jsonl_name,
            "group_key": info.get("group_key"),
            "course_code": info.get("course_code"),
        }


def _iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _chunked(seq: Sequence, size: int) -> Iterable[Sequence]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def rebuild_collection(*, batch_size: int = 64, drop_existing: bool = False) -> None:
    progress_state = DEFAULT_PROGRESS_STATE.resolve()
    progress_dir = DEFAULT_PROGRESS_DIR.resolve()
    persist_dir = DEFAULT_PERSIST_DIR.resolve()
    collection_name = DEFAULT_COLLECTION

    client = chromadb.PersistentClient(path=str(persist_dir))
    if drop_existing:
        try:
            client.delete_collection(name=collection_name)
            logger.info("Deleted existing collection '%s'", collection_name)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to delete collection '%s': %s", collection_name, exc)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info("Scanning progress entries from %s", progress_state)
    entries = list(_iter_progress_entries(progress_state))
    logger.info("Found %d JSONL exports", len(entries))

    total_chunks = 0
    for entry in entries:
        jsonl_path = progress_dir / entry["jsonl_name"]
        if not jsonl_path.exists():
            logger.warning("JSONL missing for %s (%s)", entry["source_path"], jsonl_path)
            continue

        rows = list(_iter_jsonl(jsonl_path))
        if not rows:
            logger.warning("No rows in %s", jsonl_path)
            continue

        for chunk in _chunked(rows, batch_size):
            ids: List[str] = []
            embeddings: List[List[float]] = []
            metadatas: List[Dict[str, object]] = []
            documents: List[str] = []

            for item in chunk:
                chunk_id = str(item.get("id"))
                embed = item.get("embedding")
                meta = item.get("metadata") or {}
                doc = item.get("text") or item.get("document")

                if not chunk_id or not isinstance(embed, list) or doc is None:
                    continue

                ids.append(chunk_id)
                embeddings.append(embed)
                if isinstance(meta, dict):
                    metadatas.append({str(k): v for k, v in meta.items()})
                else:
                    metadatas.append({})
                documents.append(str(doc))

            if ids:
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                total_chunks += len(ids)

        logger.info("Upserted %d chunks from %s", len(rows), jsonl_path.name)

    logger.info(
        "Rebuild finished → collection='%s' persist_dir='%s' total_chunks=%d",
        collection_name,
        persist_dir,
        total_chunks,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    rebuild_collection()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
