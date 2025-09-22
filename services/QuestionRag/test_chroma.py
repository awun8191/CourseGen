#!/usr/bin/env python3
from chromadb import PersistentClient

from services.QuestionRag.utils.chromadb_query import CHROMA_COLLECTION, CHROMA_PATH

try:
    print("Connecting to ChromaDB...")
    client = PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(name=CHROMA_COLLECTION)
    print(f"Collection count: {col.count()}")

    print("Testing basic get without embeddings...")
    result = col.get(limit=1, include=["metadatas", "documents"])
    print(f"Got {len(result.get('ids', []))} items")
    if result.get('metadatas'):
        print(f"Sample metadata: {result['metadatas'][0]}")

    print("Testing query without embeddings...")
    query_result = col.query(
        query_texts=["test query"],
        n_results=1,
        include=["metadatas", "documents", "distances"]
    )
    print(f"Query returned {len(query_result.get('ids', [[]])[0])} results")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
