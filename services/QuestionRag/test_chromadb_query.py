#!/usr/bin/env python3
import os
os.environ["CLOUDFLARE_ACCOUNT_ID"] = "c1719c3cf4696ae260e6a5f57b1f3100"
os.environ["CLOUDFLARE_API_TOKEN"] = "DQNVqFCduZCgliO47GC4kAbjPmWN_oO6lHKmIwrm"
from chromadb_query import ChromaQuery, MetaData

# Test the actual ChromaQuery class
try:
    print("Testing ChromaQuery.search with COURSE_FOLDER filter...")
    cq = ChromaQuery()

    # Test the same query that's failing in gemini_question_gen.py
    result = cq.search(
        "Obtain textbooks and questions for the course EEE 511 on Digital Signal Processing for an EEE engineering student",
        where={"COURSE_FOLDER": "EEE 511"},
        show_snippet=True,
        k=10
    )
    
    
    
    print(result)

    print(f"Search returned {len(result)} results")
    if result:
        print(f"First result metadata: {result[0].get('metadata', {})}")

except Exception as e:
    print(f"Error in ChromaQuery.search: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\nTesting ChromaQuery.search_with_temperature...")
    cq = ChromaQuery()

    result = cq.search_with_temperature(
        "course outline",
        where={"COURSE_FOLDER": "EEE 313"},
        topk=50,
        tau=0.35,
        min_sim=0.65,
        seed=None,
        show_snippet=True
    )

    print(f"Temperature search returned {len(result)} results")

except Exception as e:
    print(f"Error in ChromaQuery.search_with_temperature: {e}")
    import traceback
    traceback.print_exc()