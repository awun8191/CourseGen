#!/usr/bin/env python3
"""
inspect_chroma.py

Inspect and export a ChromaDB collection.

- List collections in a Chroma persist directory
- Show comprehensive stats (count, embedding dim, metadata stats, etc.)
- Print a small sample
- Export full collection to JSONL (optionally with embeddings)
- Export compact metadata CSV

Examples (PowerShell):
  $PERSIST = r"/home/raregazetto/Documents/Recursive-PDF-EXTRACTION-AND-RAG/COURSEGEN/services/RAG/OUTPUT_DATA2/emdeddings"
  $COLL    = "embeddings"

  # list collections
  python inspect_chroma.py -p "$PERSIST"

  # inspect one collection, show 5 rows
  python inspect_chroma.py -p "$PERSIST" -c "$COLL" -n 5

  # export full collection to JSONL (no embeddings)
  python inspect_chroma.py -p "$PERSIST" -c "$COLL" --export-jsonl "$PERSIST\\dump.jsonl"

  # export with embeddings (big!)
  python inspect_chroma.py -p "$PERSIST" -c "$COLL" --export-jsonl "$PERSIST\\dump_with_emb.jsonl" --with-embeddings

  # CSV summary of popular metadata fields
  python inspect_chroma.py -p "$PERSIST" -c "$COLL" --export-csv "$PERSIST\\meta.csv"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import chromadb
from chromadb.config import Settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Client helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_client(persist_dir: str):
    pdir = str(Path(persist_dir))
    logger.info(f"Building ChromaDB client for persist directory: {pdir}")
    try:
        client = chromadb.PersistentClient(path=pdir)  # type: ignore[attr-defined]
        logger.info("Successfully created PersistentClient")
        return client
    except Exception as e:
        logger.warning(f"PersistentClient failed: {e}, falling back to Client with Settings")
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=pdir))
        logger.info("Successfully created fallback Client")
        return client


def list_collections(client) -> List[str]:
    logger.info("Listing collections from client")
    cols = client.list_collections()
    logger.info(f"Found {len(cols)} collections")
    names = []
    for c in cols:
        name = getattr(c, "name", None)
        if not name and isinstance(c, dict):
            name = c.get("name")
        if name:
            names.append(name)
            logger.debug(f"Collection: {name}")
    logger.info(f"Collected {len(names)} collection names")
    return names


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def safe_len(x: Any) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def embedding_dim(col, total_count: int) -> Optional[int]:
    """Infer embedding dimensionality from a single row."""
    if total_count == 0:
        return None
    r = col.get(limit=1, include=["embeddings"])  # OK to ask for embeddings
    emb = r.get("embeddings", None)
    if emb is None:
        return None

    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore

    if isinstance(emb, (list, tuple)):
        if len(emb) == 0:
            return None
        first = emb[0]
        if np is not None and isinstance(first, np.ndarray):
            return int(first.shape[-1])
        if isinstance(first, (list, tuple)):
            return len(first)
        if isinstance(first, (int, float)):
            return len(emb)
        return None

    if np is not None and isinstance(emb, np.ndarray):
        if emb.ndim == 2:
            return int(emb.shape[1])
        if emb.ndim == 1:
            return int(emb.shape[0])

    return None


def compute_statistics(col, total_count: int) -> Dict[str, Any]:
    """Compute comprehensive statistics for the collection."""
    logger.info("Computing comprehensive statistics")
    stats = {
        "total_documents": total_count,
        "embedding_dimension": embedding_dim(col, total_count),
        "metadata_keys": set(),
        "document_lengths": [],
        "unique_metadata_values": {},
    }

    if total_count == 0:
        logger.info("Collection is empty, no statistics to compute")
        # Ensure all keys are present even for empty collections
        stats.update({
            "avg_document_length": 0,
            "min_document_length": 0,
            "max_document_length": 0,
        })
        return stats

    # Sample a batch to compute stats
    batch_size = min(1000, total_count)
    r = col.get(limit=batch_size, include=["documents", "metadatas"])
    docs = r.get("documents", [])
    metas = r.get("metadatas", [])

    for doc in docs:
        if isinstance(doc, str):
            stats["document_lengths"].append(len(doc))

    for meta in metas:
        if isinstance(meta, dict):
            for key, value in meta.items():
                stats["metadata_keys"].add(key)
                if key not in stats["unique_metadata_values"]:
                    stats["unique_metadata_values"][key] = set()
                if isinstance(value, (str, int, float)):
                    stats["unique_metadata_values"][key].add(value)

    # Convert sets to counts
    stats["metadata_keys"] = list(stats["metadata_keys"])
    stats["unique_metadata_values"] = {k: len(v) for k, v in stats["unique_metadata_values"].items()}

    if stats["document_lengths"]:
        stats["avg_document_length"] = sum(stats["document_lengths"]) / len(stats["document_lengths"])
        stats["min_document_length"] = min(stats["document_lengths"])
        stats["max_document_length"] = max(stats["document_lengths"])
    else:
        stats["avg_document_length"] = 0
        stats["min_document_length"] = 0
        stats["max_document_length"] = 0

    logger.info(f"Statistics computed: {len(stats['metadata_keys'])} metadata keys, avg doc length {stats['avg_document_length']:.1f}")
    return stats


def stream_all(col, *, batch: int = 1000, include: Iterable[str] = ("metadatas", "documents")):
    """
    Yield batched results:
      dict with keys: ids (always returned), and any requested in `include`.

    NOTE: Do NOT include "ids" in `include` (Chroma will error). IDs are always present.
    """
    # ensure list (Chroma rejects tuples)
    include_list = list(include)
    offset = 0
    total = col.count()
    while True:
        r = col.get(limit=batch, offset=offset, include=include_list)
        ids = r.get("ids", [])
        n = safe_len(ids)
        if n == 0:
            break
        yield r
        offset += n
        if offset >= total:
            break


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing / sampling
# ──────────────────────────────────────────────────────────────────────────────

def print_sample(col, n: int = 5) -> None:
    # DO NOT request "ids" in include
    r = col.get(limit=n, include=["documents", "metadatas"])
    ids = r.get("ids", [])
    docs = r.get("documents", [])
    metas = r.get("metadatas", [])

    N = min(n, safe_len(ids))
    if N == 0:
        print("No sample rows to display.")
        return

    print("\nSample rows:")
    for i in range(N):
        id_ = ids[i] if i < safe_len(ids) else None
        doc = docs[i] if i < safe_len(docs) else None
        md  = metas[i] if i < safe_len(metas) else {}
        # compact text
        doc_preview = None
        if isinstance(doc, str):
            doc_preview = (doc[:280] + "…") if len(doc) > 280 else doc
        print(f"— #{i+1}")
        print(f"  id: {id_}")
        if doc_preview is not None:
            print(f"  text: {json.dumps(doc_preview, ensure_ascii=False)}")
        if isinstance(md, dict) and md:
            keys_show = [
                "path","GROUP_KEY","DEPARTMENT","COURSE_CODE","COURSE_NUMBER",
                "LEVEL","SEMESTER","CATEGORY","chunk_index","total_chunks_in_doc",
                "FILENAME","STEM"
            ]
            md_preview = {k: md.get(k) for k in keys_show if k in md}
            if not md_preview:
                for k in list(md.keys())[:8]:
                    md_preview[k] = md.get(k)
            print(f"  meta: {json.dumps(md_preview, ensure_ascii=False)}")


# ──────────────────────────────────────────────────────────────────────────────
# Exports
# ──────────────────────────────────────────────────────────────────────────────

def numpy_to_list(v: Any) -> Any:
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    return v


def export_jsonl(col, path: Path, with_embeddings: bool, batch: int = 1000) -> int:
    include = ["metadatas", "documents"]
    if with_embeddings:
        include.append("embeddings")

    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in stream_all(col, batch=batch, include=include):
            ids  = r.get("ids", [])
            docs = r.get("documents", [])
            metas = r.get("metadatas", [])
            embs = r.get("embeddings", []) if with_embeddings else [None] * safe_len(ids)

            L = safe_len(ids)
            for i in range(L):
                rec = {
                    "id": ids[i],
                    "text": (docs[i] if i < safe_len(docs) else None),
                    "metadata": (metas[i] if i < safe_len(metas) else {}),
                }
                if with_embeddings:
                    e = embs[i] if i < safe_len(embs) else None
                    rec["embedding"] = numpy_to_list(e)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


CSV_KEYS_DEFAULT = [
    "id",
    "path", "FILENAME", "STEM",
    "file_size", "file_mtime", "file_hash", "chunk_hash",
    "GROUP_KEY", "DEPARTMENT", "COURSE_CODE", "COURSE_NUMBER",
    "LEVEL", "SEMESTER", "CATEGORY", "COURSE_FOLDER", "SUBCATEGORY",
    "chunk_index", "total_chunks_in_doc",
]


def export_csv(col, path: Path, keys: Optional[List[str]] = None, batch: int = 2000) -> int:
    keys = keys or CSV_KEYS_DEFAULT
    path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()

        for r in stream_all(col, batch=batch, include=["metadatas"]):
            ids = r.get("ids", [])
            metas = r.get("metadatas", [])
            L = safe_len(ids)
            for i in range(L):
                md = metas[i] if i < safe_len(metas) else {}
                row = {k: md.get(k, "") for k in keys}
                row["id"] = ids[i]
                writer.writerow(row)
                n += 1
    return n


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Inspect/export a ChromaDB collection")
    ap.add_argument("-p", "--persist", required=True, help="Path to Chroma persist directory")
    ap.add_argument("-c", "--collection", help="Collection name to inspect")
    ap.add_argument("-n", "--sample", type=int, default=0, help="Print a sample of N rows")

    ap.add_argument("--export-jsonl", type=str, help="Export all rows to a JSONL file")
    ap.add_argument("--with-embeddings", action="store_true", help="Include embeddings in JSONL (large)")
    ap.add_argument("--export-csv", type=str, help="Export a compact metadata CSV")

    args = ap.parse_args()

    # quick diagnostic: print info about the persist dir so users see if they passed
    # a wrong path (common when using relative paths or PowerShell examples on bash)
    def _report_persist(persist_dir: str) -> None:
        logger.info(f"Reporting persist directory: {persist_dir}")
        p = Path(persist_dir)
        try:
            abs_p = p.resolve()
        except Exception:
            abs_p = p
        if not p.exists():
            logger.error(f"Persist path does not exist: {p} (resolved: {abs_p})")
            print(f"Persist path does not exist: {p} (resolved: {abs_p})")
            if p.parent.exists():
                try:
                    entries = sorted(p.parent.iterdir())
                except Exception:
                    entries = []
                if entries:
                    print("Contents of parent directory:")
                    for e in entries:
                        print(f"  - {e.name}")
            return
        # exists -- show a short listing so user can confirm this is the chroma dir
        try:
            children = list(p.iterdir())
        except Exception:
            children = []
        if children:
            print(f"Persist directory: {abs_p} (contains {len(children)} entries)")
            # show a few entries with sizes
            for e in children[:20]:
                try:
                    size = e.stat().st_size
                    print(f"  - {e.name} ({size} bytes)")
                except Exception:
                    print(f"  - {e.name}")
        else:
            logger.warning(f"Persist directory exists but is empty: {abs_p}")
            print(f"Persist directory exists but is empty: {abs_p}")

    _report_persist(args.persist)

    client = build_client(args.persist)

    if not args.collection:
        names = list_collections(client)
        if not names:
            print("No collections found.")
            print("Make sure you passed the Chroma persist directory (the folder that contains Chroma files, e.g. 'chroma_db_data') and not a collection name or an empty folder.")
            return
        print("Collections:")
        for n in names:
            print(f"  - {n}")
        print("\nTip: pass -c <name> to inspect one collection.")
        return

    try:
        col = client.get_collection(args.collection)
        logger.info(f"Retrieved existing collection: {args.collection}")
    except Exception as e:
        logger.warning(f"Collection {args.collection} not found, creating new one: {e}")
        col = client.get_or_create_collection(args.collection)

    total = col.count()
    logger.info(f"Collection count: {total}")

    # Compute comprehensive statistics
    stats = compute_statistics(col, total)

    print(f"Collection: {args.collection}")
    print(f"Count     : {total}")
    print(f"Dim       : {stats['embedding_dimension'] if stats['embedding_dimension'] is not None else 'unknown'}")
    print(f"Metadata Keys: {len(stats['metadata_keys'])}")
    if stats['metadata_keys']:
        print(f"  Keys: {', '.join(stats['metadata_keys'][:10])}{'...' if len(stats['metadata_keys']) > 10 else ''}")
    print(f"Avg Doc Length: {stats['avg_document_length']:.1f} chars")
    print(f"Min Doc Length: {stats['min_document_length']}")
    print(f"Max Doc Length: {stats['max_document_length']}")

    if total == 0:
        print("\nNote: Collection is empty. No documents or embeddings found.")
        print("This could be because:")
        print("  - The collection was just created and no data has been added yet")
        print("  - Data was deleted or the persist directory is corrupted")
        print("  - The wrong collection name was specified")
    else:
        print(f"\nUnique values per metadata key (sampled):")
        for key, count in stats['unique_metadata_values'].items():
            print(f"  {key}: {count} unique values")

    if args.sample and args.sample > 0:
        print_sample(col, n=args.sample)

    if args.export_jsonl:
        out = Path(args.export_jsonl)
        n = export_jsonl(col, out, with_embeddings=args.with_embeddings, batch=1000)
        print(f"\nExported JSONL: {out}  ({n} rows){' with embeddings' if args.with_embeddings else ''}")

    if args.export_csv:
        out = Path(args.export_csv)
        n = export_csv(col, out, keys=None, batch=2000)
        print(f"Exported CSV  : {out}  ({n} rows)")

    if not args.sample and not args.export_jsonl and not args.export_csv:
        print("\n(Use -n to print a sample, or --export-jsonl / --export-csv to export the data.)")


if __name__ == "__main__":
    main()
