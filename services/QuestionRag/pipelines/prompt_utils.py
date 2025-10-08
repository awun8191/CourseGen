"""Prompt building utilities for question generation."""

from __future__ import annotations

import textwrap
from typing import Any, Dict

from .config import RequestPlan


def build_question_generation_prompt(
    *,
    course: Dict[str, Any],
    topic_title: str,
    subtopic_title: str,
    request: RequestPlan,
    context_text: str,
) -> str:
    """Build the prompt for question generation."""
    level = get_course_level(course)
    semester = get_course_semester(course)

    base_guidance = textwrap.dedent("""
        - Questions must be original, unambiguous, and self-contained.
        - Provide exactly four distinct options and preserve their order.
        - `correct_answer_indexes` must be a JSON array containing the zero-based index of the correct option (e.g., `[2]` when the third option is correct). Include exactly one index.
        - `correct_answer_text` must repeat the option string at that index verbatim.
        - Explanations should help students understand why the answer is correct and reference key formulas when relevant.
        - Wrap every formula or symbol in `$...$` with double-escaped commands (e.g., `$\\omega = 2\\pi f$`).
        """).strip()

    latex_guidance = textwrap.dedent("""
        - Use LaTeX for every mathematical expression.
        - Wrap inline math with `$...$` and multi-line math with `$$...$$` so the renderer treats it correctly.
        - **CRITICAL JSON ESCAPING:** Because the output is JSON, you MUST double-escape ALL backslashes in LaTeX.
        - Write `\\\\frac{a}{b}` in your JSON output (which becomes `\\frac{a}{b}` after JSON parsing).
        - Example inline in JSON: `"$\\\\frac{12}{4} = 3\\\\,\\\\text{Ohms}$"` (note the double backslashes).
        - Example with subscript: `"$\\\\epsilon_0$"` (double backslash before epsilon).
        - Example integral: `"$\\\\int_{0}^{1} x^2 \\\\, dx$"` (double backslashes throughout).
        - **EVERY LaTeX command needs double backslashes:** `\\\\text`, `\\\\frac`, `\\\\int`, `\\\\epsilon`, `\\\\omega`, etc.
        - Ensure explanations and solution steps follow the same double-escaped format.
        """).strip()

    if request.kind == "calculation":
        steps_guidance = textwrap.dedent("""
            - **🚨 CRITICAL: Generate ONLY calculation questions that require actual numerical computation.**
            - **🔥 EVERY QUESTION MUST:** (1) contain specific numeric values in the question text, AND (2) require mathematical calculation to solve, AND (3) have a numerical answer.
            - **✅ CALCULATION QUESTION EXAMPLES (GENERATE THESE):**
              - "A beam with length 5.2 m and load 1250 N deflects by how much?" (has numbers 5.2 m, 1250 N, requires calculation)
              - "If stress is 450 MPa and area is 25 mm², what is the force?" (has numbers 450 MPa, 25 mm², requires F = σ×A)
              - "A 2.5 mm thick panel erodes to 2.25 mm. What is the percentage reduction?" (has numbers 2.5 mm, 2.25 mm, requires math)
              - "During inspection, corrosion reduces shear area by 15%. Original strength 12,000 N. What is remaining strength?" (has 15%, 12,000 N, requires 12,000 × 0.85)
            - **❌ THEORY QUESTION EXAMPLES (NEVER GENERATE FOR CALCULATION REQUESTS):**
              - "What is the primary duty of an inspector?" (no numbers, conceptual)
              - "Which standard must inspectors follow?" (no numbers, conceptual)
              - "What should be done when a crack is found?" (no numbers, procedural)
              - "According to principles, what is the responsibility?" (no numbers, conceptual)
              - "What is the required standard for inspection?" (no numbers, conceptual)
            - **🚫 MANDATORY REQUIREMENTS:**
              - **Question text MUST include concrete numbers (e.g., "500 N", "25°C", "10 m/s²", "15%") - not variables like "F" or "A" or "the load".**
              - **Answer MUST be a specific numerical value with units (e.g., "10,200 N", "10.0%").**
              - **'solution_steps' MUST contain 3–5 calculation steps showing: formula, substitution with actual numbers, arithmetic, final result.**
              - **REJECT any question that asks "what", "which", "how should", "what is the responsibility", "according to", "primary duty", "required standard" - these are theory questions.**
            - **⚠️ If the context doesn't provide enough numeric data to create a calculation question, DO NOT generate a theory question instead - skip that question.**
            - **🎯 Focus exclusively on quantitative computational problems. NEVER generate conceptual, qualitative, or procedural questions for calculation requests.**
            - **📝 Remember: Calculation questions = numbers + math + numerical answer. Theory questions = concepts + procedures + no numbers.**
            """).strip()
    else:
        steps_guidance = textwrap.dedent("""
            - **CRITICAL: Generate ONLY conceptual theory questions that test understanding of principles and concepts.**
            - **🚫 FORBIDDEN: Questions with specific numbers, calculations, or mathematical computations.**
            - **✅ THEORY QUESTION EXAMPLES (GENERATE THESE):**
              - "What physical principle explains this orbital behavior?" (tests conceptual understanding)
              - "Which coordinate system is most appropriate for this motion?" (tests knowledge of concepts)
              - "What assumption is typically made when using the two-body problem?" (tests theoretical knowledge)
              - "How does the Coriolis effect influence the apparent motion?" (tests understanding of effects)
            - **❌ CALCULATION QUESTION EXAMPLES (NEVER GENERATE FOR THEORY REQUESTS):**
              - "If position is r(t) = 3t² + 2, what is velocity at t=2s?" (has numbers, requires calculation)
              - "What is the magnitude of velocity after 5 seconds?" (has specific time, requires math)
              - "Calculate the acceleration component a_r = ï + rθ̇²" (requires numerical computation)
            - **MANDATORY REQUIREMENTS FOR THEORY QUESTIONS:**
              - **Questions MUST ask "what", "which", "how", "why", "explain", "describe" - focusing on concepts, principles, or reasoning.**
              - **Questions MUST NOT contain specific numbers like "5.2 m", "1250 N", "15%", "t=2s".**
              - **Questions MUST test theoretical understanding, not computational ability.**
              - **Answers should be conceptual explanations, not numerical values.**
              - **'solution_steps' MUST be an empty JSON array [] - no calculations allowed.**
            - **Focus exclusively on conceptual understanding, physical principles, and qualitative reasoning.**
            - **If you cannot create a conceptual question, do not generate a calculation question instead.**
            """).strip()

    prompt = textwrap.dedent(f"""
Generate {request.question_count} unique, curriculum-aligned multiple-choice questions (MCQs) for the course "{course.get('title','')}" ({course.get('code','')}) on the topic "{topic_title}" (subtopic: "{subtopic_title}"). Use ONLY the provided extracts for grounding; do not quote them verbatim. Keep each question self-contained and original.

### REQUIRED OUTPUT (VALID JSON ONLY)
- Return a single JSON object and nothing else. No markdown, no comments, no surrounding prose.
- Top-level format:
  {{
    "questions": [
      {{
        "question": "<string>",
        "options": ["<string>", "<string>", "<string>", "<string>"],
        "correct_answer_indexes": [<int>],
        "correct_answer_text": "<string>",
        "explanation": "<string>",
        "solution_steps": ["<step1>", "<step2>", "..."]
      }}
    ]
  }}
- The JSON MUST parse as-is. Do not include extra keys or null placeholders.

### OPTIONS / ANSWER RULES
- Provide exactly **four** option *strings* in `options`. **Do not** prefix option strings with "A)", "B)", etc — options should be raw option text.
- Populate `correct_answer_indexes` with a single-element array holding the zero-based index of the correct option (0 for the first option, 1 for the second, etc.).
- `correct_answer_text` must be exactly equal to `options[correct_answer_indexes[0]]`.
- All option texts must be distinct and plausible. Avoid distractors that are obviously wrong (e.g., unit mismatch, off by factor of 1000).
- For numeric options, include units in the option text (e.g., "29.8 MPa").

### CONSISTENCY & NO-BACKTRACKING RULES
- Never alter the problem data to fit an answer. If a mismatch is detected, DISCARD that question and generate a new one that is internally consistent.
- Do not "work backwards from options." Compute the correct result first, then compose options (1 correct + 3 plausible distractors).
- Absolutely forbid meta-reasoning or self-correction in `solution_steps` (e.g., "recalculating", "let's assume", "there seems to be a discrepancy", "work backwards", "typo").

- **FOR CALCULATION QUESTIONS:**
  - `solution_steps` style:
    - Exactly 3–5 short lines, each starting with one of: "Formula:", "Convert:", "Substitute:", "Compute:", "Final:".
    - No extra sentences or commentary; each line ≤ 25 words.
    - The **Final** line must repeat the numeric answer with units and rounding.
  - Options policy:
    - Generate options AFTER computing the answer.
    - Ensure `correct_answer_indexes` identifies the correct option and `correct_answer_text` repeats it exactly.
    - For numeric questions, every option includes units; distractors reflect realistic slips (rounding, factor-of-10, omitted factor of 2), not nonsense.
    - Keep steps **minimal**: no repeating the same calculation in different units.
    - If SI units are already consistent, do not add conversions. Only convert when units are mismatched.
    - Do not restate or re-check the same formula more than once.

- **FOR THEORY QUESTIONS:**
  - `solution_steps` must be `[]` (empty JSON array).
  - **FORBIDDEN: Any calculation steps, formulas, or mathematical computations in solution_steps.**
  - Options policy:
    - Generate options that test conceptual understanding and common misconceptions.
    - Options should be different conceptual approaches, principles, or explanations.
    - Focus on testing theoretical knowledge, not computational ability.


### SOLUTION STEPS RULES
- If `request.kind == "calculation"`:
  - `solution_steps` must be an array of **3–5** concise calculation steps (strings).
  - Steps must show: formula identification, numeric substitution with actual numbers, step-by-step arithmetic computation, and final numeric result with units.
  - Every step must involve actual numerical computation - no conceptual explanations.
  - Round intermediate and final numeric results sensibly (2–4 significant figures) and state the rounding rule used.
  - Focus purely on the mathematical calculation process.
- If `request.kind == "theory"`:
  - `solution_steps` must be an empty JSON array [].
  - **FORBIDDEN: Any calculation steps, formulas, or mathematical computations.**
  - **FORBIDDEN: Any numeric values or units in solution steps.**
  - Theory questions test conceptual understanding - no calculations needed.
- Never put long prose in `solution_steps` — keep them terse, ordered, and actionable.

### MATHEMATICAL / LATEX FORMATTING
- Use LaTeX for all math. Wrap inline math with `$...$` and display math with `$$...$$`.
- **CRITICAL: Because the output is JSON, you MUST double-escape ALL backslashes in LaTeX commands.**
- A LaTeX fraction must look like `"\\\\frac{{a}}{{b}}"` in your JSON output (which becomes `\\frac{a}{b}` after JSON parsing).
- **CORRECT JSON EXAMPLES:**
  - Fraction: `"$\\\\frac{{12}}{{4}} = 3\\\\,\\\\text{{Ohms}}$"`
  - Subscript: `"$\\\\epsilon_0 = 8.85 \\\\times 10^{{-12}}$"`
  - Integral: `"$\\\\int_{{0}}^{{1}} x^2 \\\\, dx$"`
  - Greek letters: `"$\\\\omega = 2\\\\pi f$"`, `"$\\\\theta$"`, `"$\\\\epsilon$"`
  - Text in math: `"$R = 0.5\\\\,\\\\text{{m}}$"`
- **WRONG (will break JSON parsing):**
  - Single backslash: `"$\frac{a}{b}$"` ❌
  - Triple backslash: `"$\\\frac{a}{b}$"` ❌
  - No backslash: `"$frac{a}{b}$"` ❌
- Use exactly **two backslashes** for every LaTeX command: `\\\\frac`, `\\\\int`, `\\\\text`, `\\\\epsilon`, `\\\\omega`, etc.
- Do not expand algebra more than once; keep expressions in their simplest readable LaTeX form.
- Write units as `"\\\\text{{...}}"` immediately after the number, with a thin space if needed (e.g., `"$333.3\\\\,\\\\text{{kN}}$"`).


### CONTENT & PEDAGOGICAL GUIDELINES
- Align difficulty with Level **{level}** and Semester **{semester}**.
- Questions must be: original, unambiguous, self-contained, and solvable using the provided context + common engineering formulas.

- **FOR CALCULATION QUESTIONS (request.kind == "calculation"):**
  - **MANDATORY: Question MUST include specific numeric values (e.g., "500 N", "25°C", "10 m/s²", "15%") and require mathematical computation.**
  - **MANDATORY: Answer MUST be a specific numerical value with units (e.g., "10,200 N", "85%", "2.25 mm").**
  - **FORBIDDEN: Questions asking "what is", "which should", "what are the responsibilities", "according to principles" - these are theory questions.**
  - **FORBIDDEN: Questions without concrete numbers in the question text itself.**
  - **FORBIDDEN: Questions that start with "According to", "What is the", "Which of the following", "What should" when requesting calculation type.**
  - For calculation questions, always include units and show unit conversions in `solution_steps`.
  - Distractors: include 3 plausible distractors (e.g., common algebraic slips, rounding variants, unit conversion mistakes).
  - If the grounding context lacks enough numeric data to create a calculation question:
    - **DO NOT generate a theory question instead - generate a proper calculation question or skip it.**
    - Create self-contained numeric assumptions only if they fit naturally with the engineering context.

- **FOR THEORY QUESTIONS (request.kind == "theory"):**
  - **MANDATORY: Questions MUST test conceptual understanding, principles, or qualitative reasoning.**
  - **MANDATORY: Questions MUST ask "what", "which", "how", "why", "explain", "describe" - focusing on concepts and principles.**
  - **FORBIDDEN: Questions with specific numbers like "5.2 m", "1250 N", "15%", "t=2s" - these are calculation questions.**
  - **FORBIDDEN: Questions requiring mathematical computation or formula manipulation.**
  - **FORBIDDEN: Questions asking for numerical values, calculations, or quantitative results.**
  - Distractors: include 3 plausible conceptual distractors that test common misunderstandings.
  - Focus on understanding of physical principles, coordinate systems, assumptions, and theoretical concepts.

- **REMINDER: The request kind is "{request.kind}" - if it's "calculation", generate calculation questions; if it's "theory", generate theory questions.**
- Avoid excessive edge cases, trick wording, or ambiguous qualifiers (e.g., "usually", "often", "may").

### QUALITY & SANITY CHECKS (do these before returning JSON)
1. Confirm `options` contains exactly 4 items and `correct_answer_indexes[0]` points to one of them (0–3) with `correct_answer_text` matching that option exactly.
2. Confirm no option duplicates.
3. For numeric answers, re-calculate the result and ensure the value in `correct_answer_text` matches the `solution_steps` final line.
4. Confirm all LaTeX backslashes are double-escaped.
5. Validate that the full output is legal JSON (single parseable object).


### Numerical Consistency
- The numeric value in the "Final" solution step **must exactly match** the 'correct_answer_text'.
- Never produce mismatched results (e.g., steps yielding 5625 but correct_answer_text = 1125).
- If rounding is required, round consistently across steps, final answer, and correct_answer_text.
- Do not exaggerate or miscompute values; ensure units are realistic (e.g., MPa range for stresses, not GPa unless physically correct).


### TERMINATION
- After the final numeric result is given, stop generating steps.
- Do not continue with alternative derivations, assumptions, or repeated formulas.


### REFERENCE CONTEXT (grounding only)
Use these extracts strictly for background/context. Do not copy text verbatim; rephrase and use them to ensure curriculum alignment:
{context_text}

### FINAL INSTRUCTIONS
- Return exactly {request.question_count} questions that satisfy every rule above.
- **FOR CALCULATION QUESTIONS:** Keep `explanation` concise — one paragraph (1–3 sentences) that teaches why the correct answer is right and calls out the key formula(s) in `$...$` form.
- **FOR THEORY QUESTIONS:** Keep `explanation` concise — one paragraph (1–3 sentences) that explains the conceptual reasoning and principles involved.
- If any constraint cannot be satisfied for a candidate question:
  - For calculation requests: skip that question and generate another calculation question that is fully answerable.
  - For theory requests: skip that question and generate another theory question that tests conceptual understanding.
  - **CRITICAL: Never generate a calculation question when theory is requested, or vice versa.**
""").strip()

    return prompt


def get_course_level(course: Dict[str, Any]) -> str:
    """Extract course level from course data."""
    levels = course.get("levels")
    if isinstance(levels, list) and levels:
        return str(levels[0])
    if isinstance(levels, str) and levels.strip():
        return levels.strip()
    return "Unknown"


def get_course_semester(course: Dict[str, Any]) -> str:
    """Extract course semester from course data."""
    semesters = course.get("semesters")
    if isinstance(semesters, list) and semesters:
        return str(semesters[0])
    if isinstance(semesters, str) and semesters.strip():
        return semesters.strip()
    return "Unknown"
