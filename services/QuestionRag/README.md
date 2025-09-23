# QuestionRag: RAG-Powered Question Generation Pipeline

Generates 30 MCQs per subtopic (20 theory, 10 calc) using Gemini + Chroma RAG, with caching, Firestore persistence, and LaTeX for calculations. All questions grounded in retrieved docs; skips if no context.

## Usage
- Single course: `python -m services.QuestionRag.gemini_question_gen --generate-questions --course-code EEE301`
- All courses (with outlines): `python -m services.QuestionRag.gemini_question_gen --generate-questions`
  (Skips courses without outlines; processes all depts.)

- Requires pre-generated outlines in courses.json (from course_outline_generator).
- Embeddings in OUTPUT_DATA2/chroma (via services.RAG.convert_to_embeddings).
- Metadata: course code/name, topic/subtopic, level/semester from courses.json; question/options/answer/explanation/steps/rag_sources generated.

## Docker
Use root docker-compose.yml: `docker-compose run --rm coursegen python -m services.QuestionRag.gemini_question_gen ...`

## Cache & Resume
- Cache: OUTPUT_DATA2/cache (JSONL per request key: course-topic-subtopic-request).
- Resume: Loads completed batches; skips failed (marked with reason).

## Validation
- 4 unique options (A-D).
- Valid correct answer.
- Non-empty question/explanation.
- Calc: Non-empty LaTeX-wrapped steps.

See question_generator.py for config (e.g., rag_topk=30, temperature=0.25).
