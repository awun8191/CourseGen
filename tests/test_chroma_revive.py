import json
from pathlib import Path

import chromadb

from services.RAG import chroma_revive as cr


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row))
            fh.write("\n")


def test_rebuild_collection_roundtrip(tmp_path: Path, monkeypatch):
    progress_dir = tmp_path / "progress"
    progress_dir.mkdir()
    persist_dir = tmp_path / "persist"

    chunk = {
        "id": "chunk-1",
        "text": "Test document text",
        "embedding": [0.0, 1.0, 0.5],
        "metadata": {"COURSE_FOLDER": "EEE 101", "DEPARTMENT": "EEE"},
    }

    jsonl_path = progress_dir / "EEE-EEE-101__abc.jsonl"
    _write_jsonl(jsonl_path, [chunk])

    progress_state = tmp_path / "progress_state.json"
    progress_state.write_text(
        json.dumps(
            {
                "files": {
                    "/fake/src.pdf": {
                        "status": "completed",
                        "jsonl_name": jsonl_path.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cr, "DEFAULT_PROGRESS_STATE", progress_state)
    monkeypatch.setattr(cr, "DEFAULT_PROGRESS_DIR", progress_dir)
    monkeypatch.setattr(cr, "DEFAULT_PERSIST_DIR", persist_dir)
    monkeypatch.setattr(cr, "DEFAULT_COLLECTION", "test_collection")

    cr.rebuild_collection(drop_existing=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection("test_collection")
    results = collection.get(ids=["chunk-1"], include=["documents", "embeddings", "metadatas"])

    assert results["ids"] == ["chunk-1"]
    assert results["documents"] == ["Test document text"]
    assert results["metadatas"][0]["COURSE_FOLDER"] == "EEE 101"
