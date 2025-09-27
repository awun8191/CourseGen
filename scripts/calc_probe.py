"""Interactive calculation-only probe for Gemini question generation.

This script exercises the full prompt + parsing pipeline used by
`QuestionGenerator._call_gemini`, but scopes the run to a single calculation
request so we can inspect the raw Gemini response alongside the parsed JSON.

Example:
    python scripts/calc_probe.py \
        --course-code "AAE 331" \
        --topic "Aerodynamic & Structural Analysis" \
        --subtopic "Aircraft Components" \
        --questions 3

Environment:
- Requires Gemini credentials via the usual CourseGen mechanisms (env vars,
  config, or API key rotation helpers).
- Uses the existing Chroma embeddings through `ChromaQuery` to ground the
  prompt with RAG context.

Outputs:
- Echoes the prompt (optional) and prints both the raw Gemini response and the
  parsed JSON payload before rendering them into `Question` models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_models.gemini_config import GeminiConfig
from services.QuestionRag.pipelines.config import QuestionBatchConfig, RequestPlan
from services.QuestionRag.pipelines.models import GeminiQuestionBatch
from services.QuestionRag.pipelines.question_generator import QuestionGenerator


def _print_heading(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single calculation request probe for Gemini")
    parser.add_argument("--course-code", required=True, help="Course code as in courses.json, e.g. 'AAE 331'")
    parser.add_argument("--topic", required=True, help="Topic title to target")
    parser.add_argument("--subtopic", required=True, help="Subtopic title to target")
    parser.add_argument("--questions", type=int, default=3, help="Number of calculation questions to request (default: 3)")
    parser.add_argument(
        "--model",
        default=None,
        help="Override Gemini model (defaults to COURSEGEN_QUESTION_MODEL or pipeline default)",
    )
    parser.add_argument("--show-prompt", action="store_true", help="Print the generated prompt before calling Gemini")
    parser.add_argument("--structured", action="store_true", help="Force structured output mode")
    parser.add_argument("--no-structured", dest="structured", action="store_false", help="Disable structured output")
    parser.set_defaults(structured=None)
    parser.add_argument(
        "--rag-limit",
        type=int,
        default=None,
        help="Override number of RAG chunks fed into the prompt",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Temporary cache directory (defaults to pipeline cache)",
    )
    return parser


def main(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    generator = QuestionGenerator(
        use_structured=args.structured,
    )

    cfg_kwargs: Dict[str, Any] = {}
    if args.cache_dir is not None:
        cfg_kwargs["cache_dir"] = args.cache_dir
    if args.model is not None:
        cfg_kwargs["gemini_model_override"] = args.model
    if args.rag_limit is not None:
        cfg_kwargs["rag_context_limit_override"] = args.rag_limit

    config = QuestionBatchConfig(
        course_code=args.course_code,
        theory_questions_per_request_override=args.questions,
        calc_questions_per_request_override=args.questions,
        resume=False,
        store_firestore=False,
        request_delay_override=0.0,
        delay_jitter_override=0.0,
        request_attempts_override=1,
        **cfg_kwargs,
    )

    course = generator._load_course(config.courses_json_path, args.course_code)

    rag_results = generator._retrieve_rag_context(
        course=course,
        topic_title=args.topic,
        subtopic_title=args.subtopic,
        config=config,
    )

    if not rag_results:
        print("[ERROR] No RAG context found for the provided inputs; aborting")
        return 2

    context_text, rag_sources = generator._format_context(
        rag_results,
        limit=config.rag_context_limit,
        offset=0,
    )

    request = RequestPlan(
        name="calc-probe",
        kind="calculation",
        question_count=args.questions,
        difficulty_rank=5,
    )

    from services.QuestionRag.pipelines.prompt_utils import build_question_generation_prompt
    prompt = build_question_generation_prompt(
        course=course,
        topic_title=args.topic,
        subtopic_title=args.subtopic,
        request=request,
        context_text=context_text,
    )

    if args.show_prompt:
        _print_heading("Prompt")
        print(prompt)

    gemini_config = GeminiConfig(
        temperature=config.gemini_temperature,
        top_p=config.gemini_top_p,
        max_output_tokens=config.gemini_max_output_tokens,
    )
    if generator.use_structured:
        gemini_config.response_schema = GeminiQuestionBatch

    try:
        response = generator.gemini.generate(
            prompt,
            model=config.gemini_model,
            generation_config=gemini_config,
            response_model=GeminiQuestionBatch if generator.use_structured else None,
        )
    except Exception as exc:
        print(f"[ERROR] Gemini call failed: {exc}")
        return 4

    if isinstance(response, dict) and "result" in response:
        raw_result = response["result"]
    elif hasattr(response, "model_dump_json"):
        raw_result = response.model_dump_json(indent=2, ensure_ascii=False)
    else:
        raw_result = json.dumps(response, indent=2, ensure_ascii=False)

    _print_heading("Raw Gemini Response")
    print(raw_result)

    if hasattr(response, "model_dump"):
        parsed_batch = response
        parsed_json = response.model_dump()
    else:
        try:
            from services.QuestionRag.pipelines.json_utils import parse_batch_from_raw, dump_failed_payload
            parsed_batch = parse_batch_from_raw(raw_result)
        except Exception as exc:
            dump_failed_payload(raw_result)
            print(f"[ERROR] Failed to parse Gemini response: {exc}")
            return 3
        parsed_json = parsed_batch.model_dump()

    _print_heading("Parsed JSON")
    print(json.dumps(parsed_json, indent=2, ensure_ascii=False))

    from services.QuestionRag.pipelines.validation_utils import convert_to_questions
    questions = convert_to_questions(
        [q.model_dump() for q in parsed_batch.questions],
        course=course,
        topic_title=args.topic,
        subtopic_title=args.subtopic,
        request=request,
        rag_sources=rag_sources,
        wrap_latex=config.latex_wrap_steps,
    )

    _print_heading("Rendered Questions")
    for idx, q in enumerate(questions, start=1):
        print(f"[{idx}] {q.question}")
        for opt_idx, opt in enumerate(q.options, start=1):
            print(f"    {chr(64 + opt_idx)}. {opt}")
        print(f"    Answer: {q.correct_answer} ({q.correct_answer_text})")
        print(f"    Explanation: {q.explanation}")
        if q.solution_steps:
            print("    Steps:")
            for step in q.solution_steps:
                print(f"      - {step}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))




# python scripts/calc_probe.py --course-code "AAE 331" --topic "Aerodynamic & Structural Analysis" --subtopic "Aircraft Components" --questions 5 --model "gemini-2.5-flash-lite" --show-prompt > /tmp/calc_probe_flash.txt
