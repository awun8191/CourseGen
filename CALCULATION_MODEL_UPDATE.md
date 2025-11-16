# Calculation Question Model Update

## Overview
Modified the question generation pipeline to use a separate, more capable AI model specifically for calculation questions while keeping the existing model for theory questions.

## Changes Made

### 1. Configuration Files

**config.py** (Root)
- Added `gemini_calc_model` field with environment variable `COURSEGEN_CALC_MODEL`
- Default value: `gemini-2.5-flash` (more capable than the lite version)

**question_gen_config.py**
- Added `gemini_calc_model: str = "gemini-2.5-flash"` to `QuestionGenerationConfig`
- Updated `from_env()` method to read from centralized config

**config.py** (Pipeline)
- Added `gemini_calc_model` property to `QuestionBatchConfig` class
- Property delegates to centralized config

### 2. Question Generator

**question_generator.py**
- Modified `_call_gemini()` method to select model based on question type:
  ```python
  model = config.gemini_calc_model if request.kind == "calculation" else config.gemini_model
  ```
- All Gemini API calls now use the selected model variable

### 3. Environment Variables

**.env.example**
- Added `COURSEGEN_CALC_MODEL=gemini-2.5-flash` to model configuration section

## Usage

### Set the Environment Variable

Add to your `.env` file:
```bash
COURSEGEN_CALC_MODEL=gemini-2.5-flash
```

### Default Behavior

- **Theory questions**: Use `COURSEGEN_QUESTION_MODEL` (default: `gemini-2.5-flash-lite`)
- **Calculation questions**: Use `COURSEGEN_CALC_MODEL` (default: `gemini-2.5-flash`)

### Customization

You can set any Gemini model for calculation questions:
```bash
# Use the most capable model for calculations
COURSEGEN_CALC_MODEL=gemini-2.5-pro

# Or use the same model for both (not recommended)
COURSEGEN_CALC_MODEL=gemini-2.5-flash-lite
```

## Benefits

1. **Better Performance**: Gemini 2.5 Flash is more capable at mathematical reasoning than Flash Lite
2. **Cost Optimization**: Theory questions still use the cheaper Flash Lite model
3. **Flexibility**: Easy to experiment with different models for calculations
4. **Backward Compatible**: Existing configurations continue to work

## Testing

To verify the change is working:

1. Set `COURSEGEN_DEBUG=true` in your `.env`
2. Run question generation
3. Check logs for model selection per request type

## Rollback

To revert to using the same model for all questions:
```bash
COURSEGEN_CALC_MODEL=gemini-2.5-flash-lite
```
