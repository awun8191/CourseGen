#!/usr/bin/env python3
import chromadb
from chromadb import PersistentClient

# Test ChromaDB collection with filters
CHROMA_PATH = r"/home/raregazetto/Documents/Recursive-PDF-EXTRACTION-AND-RAG/src/services/RAG/OUTPUT_DATA/chroma_db_data"
COLLECTION = "pdfs_bge_m3_cloudflare"

try:
    print("Connecting to ChromaDB...")
    client = PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(name=COLLECTION)
    print(f"Collection count: {col.count()}")

    print("Testing get with COURSE_FOLDER filter...")
    result = col.get(
        where={"COURSE_FOLDER": "EEE 511"},
        limit=5,
        include=["metadatas", "documents"]
    )
    print(f"Got {len(result.get('ids', []))} items with COURSE_FOLDER='EEE 511'")

    if result.get('metadatas'):
        for i, meta in enumerate(result['metadatas'][:3]):
            print(f"Item {i}: COURSE_FOLDER={meta.get('COURSE_FOLDER')}")

    print("Testing query with COURSE_FOLDER filter...")
    query_result = col.query(
        query_texts=["test query"],
        where={"COURSE_FOLDER": "EEE 511"},
        n_results=3,
        include=["metadatas", "documents", "distances"]
    )
    print(f"Query returned {len(query_result.get('ids', [[]])[0])} results")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()