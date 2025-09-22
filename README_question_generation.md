# Question Generation in CourseGen

This module generates diverse, interactive questions (MCQ, short answer, essay, true/false) for courses, leveraging RAG to ensure relevance to specific topics and learning objectives. It builds on course outlines or direct queries, using Gemini to create questions at varying difficulty levels (aligned with Bloom's taxonomy). Outputs include answers, explanations, and source references, formatted for quizzes or e-learning platforms.

## Overview
Manual question creation is labor-intensive; this pipeline automates it while maintaining pedagogical quality:
- Retrieves context-specific chunks from ChromaDB (e.g., filtered by topic "Z-Transforms").
- Synthesizes questions via structured prompts, supporting formats like MCQs with distractors.
- Validates with Pydantic (`question_model.py`) for consistency.
- Batch-capable with rate limiting and caching.
- Integrates with outlines for targeted generation (e.g., 10 questions per module).

Key benefits:
- Generates 100s of questions/hour, customizable by difficulty/depth.
- Reduces bias/hallucinations via RAG grounding.
- Tracks sources for traceability and fairness.
- Supports interactive features like adaptive difficulty.

## Architecture
1. **Input**: Course outline MD/JSON (`--input-outline`), or direct query/topic (e.g., "Explain Fourier transforms in DSP").
2. **Retrieval**: `services/QuestionRag/utils/chromadb_query.py` fetches top-k chunks, filtered by metadata (e.g., `LEVEL=400`, `CATEGORY=PQ` for past questions).
3. **Generation**: `services/QuestionRag/question_generator.py` (or `gemini_question_gen.py`):
   - Loads prompts from resources (e.g., JSONL templates for MCQ vs. essay).
   - Calls balanced Gemini client (`gemini_service.py`).
   - Generates per-topic: Questions + Answer + Explanation + Difficulty + Bloom Level + Sources.
4. **Post-Processing**: Deduplicates (`utils/Remove Duplicates/`), formats (JSONL/MD), caches responses.
5. **Output**: JSONL files (e.g., `data/questions_EEE471.jsonl`) or integrated with Firestore.
6. **Batching**: `services/QuestionRag/utils/batch_utils.py` for parallel processing across topics/courses.

Dependencies:
- Embeddings/ChromaDB.
- Gemini keys (load balanced).
- Optional: `data/courses_fallback_with_topics.json` for topic extraction.

## Usage
### Prerequisites
- Embeddings processed (see [Convert to Embeddings README](README_convert_to_embeddings.md)).
- Outlines generated (see [Course Outline README](README_course_outline_generation.md)) or use direct mode.

### CLI Command
```
python -m services.QuestionRag.question_generator [OPTIONS]
# Or: python services/QuestionRag/gemini_question_gen.py [OPTIONS]
```

#### Key Options
- `--input-outline PATH`: MD/JSON outline file. Required for outline-based mode.
- `--input-query STR`: Direct topic/query (e.g., "Z-Transforms in EEE471"). Alternative to outline.
- `--num-questions INT`: Total questions (default 20; distributes across topics).
- `--questions-per-topic INT`: If outline provided (default 5).
- `--difficulty STR`: low/medium/high (default medium; maps to Bloom: Remember/Apply/Analyze).
- `--question-types LIST`: e.g., ["mcq", "short_answer", "essay"] (default all).
- `--collection STR`: Chroma collection (default "pdfs_bge_m3_cloudflare").
- `--persist-dir PATH`: Chroma path (default "chromadb_storage").
- `--output PATH`: Save file (default "data/questions.jsonl"; supports .jsonl, .md).
- `--output-format STR`: jsonl/markdown (default jsonl).
- `--top-k INT`: Retrieval chunks per query (default 30).
- `--temperature FLOAT`: Creativity (0.1 for factual; default 0.4).
- `--max-tokens INT`: Per question (default 500).
- `--workers INT`: Parallel (default 1; requires load balancer).
- `--cache-dir PATH`: Gemini cache (default "data/gemini_cache").
- `--resume`: Skip cached questions.
- `--with-explanations`: Always include (default true).
- `--bloom-levels LIST`: e.g., ["remember", "apply"] (default all).

#### Outline-Based Example
```
python -m services.QuestionRag.question_generator \
  --input-outline utils/course_outline/course_outline_EEE471.md \
  --num-questions 30 \
  --difficulty medium \
  --question-types mcq short_answer \
  --output data/questions_EEE471.jsonl \
  --top-k 40 \
  --workers 2
```

#### Direct Query Example
```
python -m services.QuestionRag.question_generator \
  --input-query "Explain sampling theorem in digital signal processing for EEE471" \
  --num-questions 10 \
  --difficulty high \
  --output-format markdown \
  --collection pdfs_bge_m3_cloudflare
```

### Programmatic Usage
```python
from services.QuestionRag.question_generator import generate_questions

questions = generate_questions(
    input_outline="path/to/outline.md",
    num_questions=20,
    difficulty="medium",
    top_k=30
)
import json
with open("questions.jsonl", "w") as f:
    for q in questions:
        f.write(json.dumps(q) + "\n")
```

## Prompt Engineering
Templates in `services/QuestionRag/resources/` (JSONL):
```jsonl
{"role": "system", "content": "Generate {num} {type} questions on {topic} from {chunks}. Include 4 options for MCQ with one correct. Difficulty: {difficulty} (Bloom: {level}). Provide explanation and source."}
{"role": "user", "content": "Topic: {topic}\nChunks: {retrieved_text}\nOutput JSON: [{question, type, options[], answer, explanation, difficulty, bloom_level, sources[]}]"}
```
- Customize: Add distractor strategies for MCQs, word limits for essays.
- Test: Run with small `--num-questions` and inspect for accuracy.

## Output Format
### JSONL (Default)
Each line:
```json
{
  "id": "q_EEE471_ztransform_1",
  "question": "What is the z-transform of the unit step sequence u[n]?",
  "type": "mcq",
  "difficulty": "medium",
  "bloom_level": "understand",
  "options": [
    "A) 1/(1 - z^-1)",
    "B) 1/z",
    "C) z/(z-1)",
    "D) 1"
  ],
  "answer": "A",
  "explanation": "The z-transform of u[n] is the sum from n=0 to inf of z^-n = 1/(1 - z^-1) for |z| > 1.",
  "sources": [
    {"path": "EEE471_textbook.pdf", "chunk_index": 45, "page": 23}
  ],
  "topic": "Z-Transforms",
  "generated_at": "2025-09-22T10:00:00Z"
}
```

### Markdown
```
## Questions for Module 1: Z-Transforms

### Q1 (MCQ, Medium)
What is the z-transform of the unit step sequence u[n]?

A) 1/(1 - z^-1)  
B) 1/z  
C) z/(z-1)  
D) 1  

**Answer**: A  
**Explanation**: The z-transform of u[n] is ...  
**Source**: EEE471_textbook.pdf (page 23)
```

## Integration
- **With Outlines**: Auto-generate questions per module.
- **With Platforms**: Export to JSONL for Quizlet/Moodle import.
- **With Firestore**: Store for real-time quiz apps.
- **Batch from Catalog**: Use `data/courses.json` + `batch_utils.py`.
- **Customization**: Extend `question_model.py` for new types (e.g., fill-in-blank).

## Testing and Validation
- **Unit Tests**: `pytest tests/test_gemini_question_gen_cache.py`, `test_filter.py`.
- **Retrieval Tests**: `test_chromadb_query.py` for relevance.
- **Quality Checks**: Manual review; compute diversity (e.g., unique topics covered).
- **Edge Cases**: Test low-retrieval scenarios, duplicate questions.
- **Performance**: Time 100 questions; ensure <5s/question with caching.

## Troubleshooting
- **Irrelevant Questions**: Refine retrieval filters or increase `--top-k`.
- **Rate Limits**: Enable load balancer; check `rate_limit_data.py`.
- **Hallucinations**: Lower `--temperature`; verify sources match chunks.
- **Format Errors**: Validate with `question_model.py`; debug prompts.
- **Cache Misses**: Clear `data/gemini_cache/` if stale.
- **Logs**: Use `services/RAG/log_utils.py` for traces (e.g., retrieved chunks).

## Best Practices
- Balance types: 40% MCQ, 30% short answer, 30% essay for variety.
- Align with objectives: Map questions to outline LOs.
- Review batches: Spot-check 10% for accuracy.
- Scale gradually: Start with 10 questions/topic, expand.
- Track usage: Integrate with billing for cost per question set.

## Future Enhancements
- Adaptive generation: Adjust difficulty based on user performance.
- Multi-modal: Include diagram-based questions.
- Evaluation: Auto-grade sample answers with Gemini.
- Localization: Translate questions for non-English courses.

This module empowers educators to build dynamic question banks efficiently, enhancing interactive learning.