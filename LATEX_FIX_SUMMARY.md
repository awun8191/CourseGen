# LaTeX Formatting Fixes for Question Generation

## Problem
The AI model was generating LaTeX expressions with single backslashes (e.g., `\frac`, `\text`, `\epsilon`) which broke JSON parsing, causing errors like:
```
[ERROR] Could not parse JSON for EEE 315 - Permittivity and Electric Fields (theory-1)
```

## Root Cause
LaTeX commands use backslashes (`\`), but JSON also uses backslashes for escape sequences. When the model generates `\text{m}` in JSON, the parser interprets `\t` as a tab character, breaking the JSON structure.

## Solution Applied

### 1. **Strengthened Prompt Instructions** (`prompt_utils.py`)
Enhanced the LaTeX formatting section with:
- Explicit examples showing double-backslash format: `\\\\frac`, `\\\\text`, `\\\\epsilon`
- Clear "CORRECT" vs "WRONG" examples
- Emphasized that **every** LaTeX command needs double backslashes in JSON output

**Key changes:**
```python
- **CRITICAL: Because the output is JSON, you MUST double-escape ALL backslashes in LaTeX commands.**
- **CORRECT JSON EXAMPLES:**
  - Fraction: `"$\\\\frac{{12}}{{4}} = 3\\\\,\\\\text{{Ohms}}$"`
  - Greek letters: `"$\\\\epsilon_0$"`, `"$\\\\omega = 2\\\\pi f$"`
- **WRONG (will break JSON parsing):**
  - Single backslash: `"$\frac{a}{b}$"` ❌
```

### 2. **Enhanced JSON Parser** (`json_utils.py`)
Added a pre-processing step that automatically fixes LaTeX within `$...$` delimiters:

```python
def preprocess_latex_in_math(text: str) -> str:
    """Pre-process LaTeX within $...$ delimiters to ensure proper escaping."""
    def fix_math_expr(match):
        content = match.group(1)
        # Double any single backslashes that aren't already doubled
        fixed = re.sub(r'(?<!\\)\\(?!\\)', r'\\\\', content)
        return f'${fixed}$'
    
    result = re.sub(r'\$([^$]+)\$', fix_math_expr, text)
    return result
```

This function:
- Finds all math expressions between `$...$`
- Automatically doubles single backslashes within them
- Preserves already-doubled backslashes
- Runs before JSON parsing attempts

### 3. **Multi-Stage Parsing Strategy**
The parser now tries multiple strategies in order:
1. Original content (in case it's already correct)
2. Pre-processed LaTeX (fixes single backslashes in math)
3. Full sanitization (handles edge cases)
4. Trailing comma removal (fixes common JSON formatting issues)
5. Truncation repair (handles incomplete responses)

## Testing
Created `test_latex_json.py` to verify the fixes handle:
- ✓ Single backslash LaTeX (auto-corrected)
- ✓ Double backslash LaTeX (preserved)
- ✓ Fractions: `\frac{a}{b}`
- ✓ Greek letters: `\epsilon`, `\omega`, `\theta`
- ✓ Text in math: `\text{m}`, `\text{Ohms}`
- ✓ Multiple expressions in one string

All tests pass.

## Impact
- **Prevention**: Clearer prompt instructions reduce the chance of the model generating incorrect format
- **Recovery**: Enhanced parser automatically fixes common LaTeX escaping issues
- **Robustness**: Multi-stage parsing handles various edge cases gracefully

## Usage
No changes needed to existing code. The fixes are transparent:
- The prompt automatically includes the enhanced LaTeX instructions
- The parser automatically applies the fixes during JSON parsing
- Failed payloads are still saved for debugging if all strategies fail

## Example
**Before (would fail):**
```json
{"question": "A sphere has radius $R = 0.5\,\text{m}$"}
```

**After (auto-corrected to):**
```json
{"question": "A sphere has radius $R = 0.5\\,\\text{m}$"}
```

## Files Modified
1. `services/QuestionRag/pipelines/prompt_utils.py` - Enhanced LaTeX instructions
2. `services/QuestionRag/pipelines/json_utils.py` - Added pre-processor and improved parsing
3. `test_latex_json.py` - Test suite for verification

## Next Steps
If you still encounter LaTeX-related JSON parsing errors:
1. Check the saved payload in the error message location
2. Run `python test_latex_json.py` to verify the parser is working
3. Look for unusual LaTeX patterns not covered by the current fixes
4. Consider adding those patterns to the test suite
