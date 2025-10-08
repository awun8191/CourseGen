from services.QuestionRag.utils.chromadb_query import ChromaQuery
cq = ChromaQuery()
result = cq.col.get(include=['metadatas'])
courses = sorted(set(m.get('COURSE_FOLDER', 'Unknown') for m in result['metadatas'] if m))
print(f'Found {len(courses)} courses:')
for c in courses:
    print(f'  - {c}')