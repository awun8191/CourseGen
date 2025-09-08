"""Wrapper around the Google Gemini (new SDK) client with typed configuration."""

from __future__ import annotations

import json
import os
import time
import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)
from typing import get_origin, get_args

from google import genai
from google.genai import types as gtypes
from google.genai import errors as genai_errors

# Backwards-compat stubs for old exceptions referenced in some branches
try:  # pragma: no cover
    from google.generativeai.types.generation_types import StopCandidateException  # type: ignore
    from google.api_core.exceptions import ResourceExhausted  # type: ignore
except Exception:  # pragma: no cover
    class StopCandidateException(Exception):
        pass

    class ResourceExhausted(Exception):
        pass
from pydantic import BaseModel

from COURSEGEN.data_models.gemini_config import GeminiConfig
from .api_key_manager import ApiKeyManager

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

class CourseOutline(BaseModel):
    course: str
    topics: List[str]
    description: str


class CoursesResponse(BaseModel):
    courses: List[CourseOutline]



class GeminiService:
    """Simple wrapper around the new Google GenAI client
    with typed configuration.
    """

    def __init__(
        self,
        api_keys: List[str] = None,
        model: str = DEFAULT_MODEL,
        generation_config: Optional[GeminiConfig] = None,
        api_key_manager: Optional[ApiKeyManager] = None,
    ) -> None:
        self.model = model
        self.default_config = generation_config or GeminiConfig()
        
        # Use default API keys if none provided
        if api_keys is None and api_key_manager is None:
            from .gemini_api_keys import GeminiApiKeys
            gemini_keys = GeminiApiKeys()
            api_keys = gemini_keys.get_keys()
        
        self.api_key_manager = api_key_manager or ApiKeyManager(api_keys)
        self._configure_genai()

    def _configure_genai(self, model: str = "flash"):
        """Configure the new Google GenAI client with the current API key for the given model."""
        api_key = self.api_key_manager.get_key(model)
        # Create a client instance bound to the current API key
        self.client = genai.Client(api_key=api_key)
    
    
    _GEMINI_SVC_SINGLETON = None


    def _to_generation_config(
        self, config: Optional[GeminiConfig | Dict[str, Any]]
    ) -> tuple[gtypes.GenerateContentConfig, Optional[list[dict[str, Any]]]]:
        """Convert configuration data to a GenerateContentConfig and tools."""
        if config is None:
            config = self.default_config
        elif isinstance(config, dict):
            config = GeminiConfig(**config)
        elif not isinstance(config, GeminiConfig):
            raise TypeError("generation_config must be GeminiConfig or dict")

        def _simple_type_schema(py_type: Any) -> dict[str, Any]:
            origin = get_origin(py_type)
            args = get_args(py_type)
            if origin is list or origin is List:
                item_t = args[0] if args else str
                return {"type": "array", "items": _simple_type_schema(item_t)}
            # Literal values
            try:
                from typing import Literal as _Literal
            except Exception:
                _Literal = None
            if _Literal is not None and get_origin(py_type) is _Literal:
                vals = list(get_args(py_type))
                # Assume literals are strings/numbers; default to string
                # Use enum to constrain
                return {"type": "string", "enum": [str(v) for v in vals]}
            # Nested BaseModel
            if isinstance(py_type, type) and issubclass(py_type, BaseModel):
                return _pydantic_to_simple_schema(py_type)
            # Primitives
            if py_type in (str, Any):
                return {"type": "string"}
            if py_type in (int,):
                return {"type": "integer"}
            if py_type in (float,):
                return {"type": "number"}
            if py_type in (bool,):
                return {"type": "boolean"}
            # Fallback
            return {"type": "string"}

        def _pydantic_to_simple_schema(model_cls: Type[BaseModel]) -> dict[str, Any]:
            schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            for name, field in model_cls.model_fields.items():
                # Field type resolution
                py_t = field.annotation if field.annotation is not None else str
                prop_schema = _simple_type_schema(py_t)
                if field.description:
                    prop_schema["description"] = field.description
                schema["properties"][name] = prop_schema
                # Consider it required if no default
                # Pydantic v2: field.is_required attribute
                is_required = getattr(field, "is_required", False)
                if is_required:
                    schema["required"].append(name)
            if not schema["required"]:
                schema.pop("required")
            return schema

        # Build generation config for new SDK
        gen_config = gtypes.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
            response_mime_type=("application/json" if config.response_schema else None),
        )

        tools = None
        if config.response_schema:
            try:
                if isinstance(config.response_schema, type) and issubclass(config.response_schema, BaseModel):
                    simple_schema = _pydantic_to_simple_schema(config.response_schema)
                else:
                    # If a mapping was provided, strip unsupported keys
                    raw = dict(config.response_schema)
                    def _strip(d: Any) -> Any:
                        if isinstance(d, dict):
                            out = {}
                            for k, v in d.items():
                                if k in ("type", "format", "description", "enum", "properties", "items", "required"):
                                    out[k] = _strip(v)
                            return out
                        elif isinstance(d, list):
                            return [_strip(x) for x in d]
                        else:
                            return d
                    simple_schema = _strip(raw)
                # Attach the simplified schema to generation config
                gen_config.response_schema = simple_schema  # type: ignore[attr-defined]
            except Exception:
                # If schema conversion fails, skip response_schema
                pass

        return gen_config, tools

    def _generate(
        self,
        parts: Iterable[Any],
        model: str,
        generation_config: Optional[GeminiConfig | Dict[str, Any]],
        response_model: Optional[Type[T]] = None,
    ) -> T | Dict[str, Any]:
        """Internal helper to perform a generation request."""
        try:
            print("Generation Started")
            gen_config, tools = self._to_generation_config(generation_config)
            print("Step 1")
            # Configure per-model key (flash/lite/pro)
            model_name = self._get_model_name(model)
            self._configure_genai(model_name)
            # If a response schema is provided, config already covers it.
            print("Step 2")
            response = self.client.models.generate_content(
                model=model,
                contents=list(parts),
                config=gen_config,
            )
            # Extract text robustly (response.text can be empty for non-text MIME types)
            response_text = getattr(response, "text", "") or ""
            if not response_text:
                try:
                    parts_text: list[str] = []
                    for cand in (getattr(response, "candidates", []) or []):
                        content = getattr(cand, "content", None)
                        if content and getattr(content, "parts", None):
                            for p in content.parts:
                                t = getattr(p, "text", None)
                                if t:
                                    parts_text.append(t)
                    response_text = "\n".join(parts_text).strip()
                except Exception:
                    response_text = ""
            if os.environ.get("GEMINI_DEBUG_PRINT", "0") == "1":
                try:
                    print(f"Response: {response_text}")
                except UnicodeEncodeError:
                    preview = response_text[:1000]
                    safe_preview = preview.encode('ascii', 'ignore').decode('ascii', 'ignore')
                    print(f"Response (preview, non-ascii omitted): {safe_preview} ...")

            # Update usage
            key = self.api_key_manager.get_key(model_name)
            # A simple way to estimate tokens, not perfect.
            tokens = len(response_text) / 4
            self.api_key_manager.update_usage(
                key, model_name, int(tokens)
            )

            # For plain text responses, return the text directly
            if generation_config and generation_config.response_schema is None:
                return {"result": response_text.strip()}

            # For structured responses, handle JSON/function-call parsing
            # 1) Prefer tool/function_call args when a response schema is set.
            def _extract_function_args(resp) -> Optional[Dict[str, Any]]:
                try:
                    candidates = getattr(resp, "candidates", None) or []
                    for cand in candidates:
                        content = getattr(cand, "content", None)
                        if not content:
                            continue
                        parts = getattr(content, "parts", None) or []
                        for p in parts:
                            # Different SDK versions expose either `function_call` or `functionCall`.
                            fc = getattr(p, "function_call", None) or getattr(p, "functionCall", None)
                            if not fc:
                                continue
                            # Try dict-like access first
                            if isinstance(fc, dict):
                                args = fc.get("args") or fc.get("arguments")
                            else:
                                args = getattr(fc, "args", None) or getattr(fc, "arguments", None)
                            if args is None:
                                continue
                            if isinstance(args, str):
                                try:
                                    return json.loads(args)
                                except Exception:
                                    # Some SDKs return already-parsed dict-like `args` as a JSON string
                                    # that may itself be malformed. Fall back to text extraction below.
                                    pass
                            elif isinstance(args, dict):
                                return args
                    return None
                except Exception:
                    return None

            # 2) If we have tools, try to parse function-call args first.
            data: Any = None
            if tools:
                data = _extract_function_args(response)

            if data is None:
                # 3) Fallback to parsing textual JSON output.
                cleaned_text = (response_text or "").strip()
                if cleaned_text.startswith('```json'):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()

                # Try direct parse first
                try:
                    data = json.loads(cleaned_text)
                except json.JSONDecodeError as e:
                    if os.environ.get("GEMINI_DEBUG_JSON", "0") == "1":
                        print(f"JSON parsing error: {e}")
                    # Safer extraction: find the first balanced JSON object/array.
                    def _extract_balanced_json(s: str) -> Optional[str]:
                        import itertools
                        start = None
                        opener = None
                        closer = None
                        # Find first '{' or '['
                        for i, ch in enumerate(s):
                            if ch == '{' or ch == '[':
                                start = i
                                opener = ch
                                closer = '}' if ch == '{' else ']'
                                break
                        if start is None:
                            return None
                        depth = 0
                        in_string = False
                        escaped = False
                        for j in range(start, len(s)):
                            ch = s[j]
                            if in_string:
                                if escaped:
                                    escaped = False
                                elif ch == '\\':
                                    escaped = True
                                elif ch == '"':
                                    in_string = False
                                continue
                            else:
                                if ch == '"':
                                    in_string = True
                                    continue
                                if ch == opener:
                                    depth += 1
                                elif ch == closer:
                                    depth -= 1
                                    if depth == 0:
                                        return s[start : j + 1]
                        # Not balanced
                        return None

                    candidate = _extract_balanced_json(cleaned_text)
                    if candidate is not None:
                        try:
                            data = json.loads(candidate)
                        except json.JSONDecodeError as e2:
                            if os.environ.get("GEMINI_DEBUG_JSON", "0") == "1":
                                print(f"Second JSON parsing error: {e2}")
                            # Try a minimal repair by closing strings/brackets
                            def _repair_truncated_json(s: str) -> Optional[str]:
                                # Locate start of JSON
                                start = None
                                for i, ch in enumerate(s):
                                    if ch in "[{":
                                        start = i
                                        break
                                if start is None:
                                    return None
                                in_string = False
                                escaped = False
                                stack: list[str] = []
                                for ch in s[start:]:
                                    if in_string:
                                        if escaped:
                                            escaped = False
                                        elif ch == "\\":
                                            escaped = True
                                        elif ch == '"':
                                            in_string = False
                                        continue
                                    else:
                                        if ch == '"':
                                            in_string = True
                                            continue
                                        if ch in "[{":
                                            stack.append(ch)
                                        elif ch in "]}":
                                            if stack:
                                                opener = stack[-1]
                                                if (opener == "[" and ch == "]") or (opener == "{" and ch == "}"):
                                                    stack.pop()
                                # Build repaired string
                                repaired = s[start:]
                                if in_string:
                                    repaired += '"'
                                # Close any remaining brackets/braces
                                for opener in reversed(stack):
                                    repaired += "]" if opener == "[" else "}"
                                return repaired

                            repaired = _repair_truncated_json(cleaned_text)
                            if repaired:
                                try:
                                    data = json.loads(repaired)
                                except Exception:
                                    return {"result": None, "raw": response_text}
                            else:
                                return {"result": None, "raw": response_text}
                    else:
                        # Try a minimal repair when nothing balanced was found
                        def _repair_truncated_json(s: str) -> Optional[str]:
                            start = None
                            for i, ch in enumerate(s):
                                if ch in "[{":
                                    start = i
                                    break
                            if start is None:
                                return None
                            in_string = False
                            escaped = False
                            stack: list[str] = []
                            for ch in s[start:]:
                                if in_string:
                                    if escaped:
                                        escaped = False
                                    elif ch == "\\":
                                        escaped = True
                                    elif ch == '"':
                                        in_string = False
                                    continue
                                else:
                                    if ch == '"':
                                        in_string = True
                                        continue
                                    if ch in "[{":
                                        stack.append(ch)
                                    elif ch in "]}":
                                        if stack:
                                            opener = stack[-1]
                                            if (opener == "[" and ch == "]") or (opener == "{" and ch == "}"):
                                                stack.pop()
                            repaired = s[start:]
                            if in_string:
                                repaired += '"'
                            for opener in reversed(stack):
                                repaired += "]" if opener == "[" else "}"
                            return repaired

                        repaired = _repair_truncated_json(cleaned_text)
                        if repaired:
                            try:
                                data = json.loads(repaired)
                            except Exception:
                                return {"result": None, "raw": response_text}
                        else:
                            # As a last resort, return raw text to avoid crashing
                            return {"result": None, "raw": response_text}

            if os.environ.get("GEMINI_DEBUG_PRINT", "0") == "1":
                print(data)
            if response_model:
                # Minimal post-processing to satisfy simple schema constraints
                try:
                    if isinstance(data, dict) and isinstance(data.get("questions"), list):
                        def _sanitize(text: str) -> str:
                            if not isinstance(text, str):
                                return text
                            sents = re.split(r"(?<=[.!?])\s+", text.strip())
                            sents = [
                                s for s in sents
                                if not re.search(r"\b(let\s+me|let's|assume|apologies|re-?calculate|recheck)\b", s, flags=re.IGNORECASE)
                            ]
                            out = " ".join(sents).strip()
                            return re.sub(r"\s+", " ", out)
                        for q in data["questions"]:
                            if isinstance(q, dict):
                                steps = q.get("solution_steps")
                                if isinstance(steps, list) and len(steps) > 5:
                                    q["solution_steps"] = steps[:5]
                                if "explanation" in q:
                                    q["explanation"] = _sanitize(q["explanation"])
                                if "question" in q:
                                    q["question"] = _sanitize(q["question"])
                except Exception:
                    pass
                return response_model.model_validate(data)
            return {
                "result": data
            }
        except genai_errors.ClientError as e:
            # Handle 4xx errors (including 429 rate limiting) by rotating keys
            print(f"API key failed (client error {e.code}): {e.message}")
            if getattr(e, "code", 0) == 429:
                self.api_key_manager.rotate_key(model_name)
                self._configure_genai(model_name)
                return self._generate(parts, model, generation_config, response_model)
            raise
        except genai_errors.ServerError as e:
            # Server-side issues, try rotating key once
            print(f"Server error {e.code}: {e.message}. Rotating key and retrying...")
            self.api_key_manager.rotate_key(model_name)
            self._configure_genai(model_name)
            return self._generate(parts, model, generation_config, response_model)

    def _get_model_name(self, model_str: str) -> str:
        if "lite" in model_str:
            return "lite"
        if "flash" in model_str:
            return "flash"
        if "pro" in model_str:
            return "pro"
        if "embedding" in model_str:
            return "embedding"
        return "flash"  # Default

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        generation_config: Optional[GeminiConfig | Dict[str, Any]] = None,
        response_model: Optional[Type[T]] = None,
    ) -> T | Dict[str, Any]:
        """Generate text from a prompt using the specified model."""
        gen_conf = generation_config
        # If a response model is provided but no response schema is set,
        # automatically enable structured output using that Pydantic model.
        if response_model is not None:
            if gen_conf is None:
                gen_conf = self.default_config.model_copy(deep=True)
            elif isinstance(gen_conf, dict):
                gen_conf = GeminiConfig(**gen_conf)
            elif not isinstance(gen_conf, GeminiConfig):
                gen_conf = self.default_config.model_copy(deep=True)

            if getattr(gen_conf, "response_schema", None) is None:
                gen_conf.response_schema = response_model

        return self._generate(
            [prompt], model or self.model, gen_conf, response_model
        )

    # -------------------- Helper methods for embedding batching --------------------
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using a simple 4-characters-per-token heuristic."""
        return max(1, len(text) // 4)

    def _batch_texts(self, texts: List[str], max_tokens: int = 2000) -> List[List[str]]:
        """Split ``texts`` so each batch stays within ``max_tokens`` heuristic limit."""
        batches: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0
        for t in texts:
            t_tok = self._estimate_tokens(t)
            # Extremely long text: send alone to avoid infinite loop
            if t_tok > max_tokens:
                if current:
                    batches.append(current)
                    current, current_tokens = [], 0
                batches.append([t])
                continue
            if current_tokens + t_tok > max_tokens and current:
                batches.append(current)
                current, current_tokens = [t], t_tok
            else:
                current.append(t)
                current_tokens += t_tok
        if current:
            batches.append(current)
        return batches


    def ocr(self, images: List[Dict], prompt: str, response_model: Type[T] = None) -> T:

        if not images:
            raise ValueError("No images provided for OCR")
        
        # Convert images to the format expected by generate method
        content = []
        for img in images:
            content.append({
                "mime_type": img["mime_type"],
                "data": img["data"]
            })
        
        # Add the text prompt
        content.append(prompt)
        
        # Use the existing generate method
        return self.generate(content, response_model=response_model)

    # -------------------- Main embedding method --------------------
    def embed(self, texts: List[str], model: Optional[str] = None, target_dim: Optional[int] = 768) -> List[List[float]]:
        """Generate embeddings while respecting ~2 k tokens/request and ≤5 RPM limits."""
        if not texts:
            return []

        embedding_model = model or EMBEDDING_MODEL
        batches = self._batch_texts(texts, max_tokens=2000)
        
        # Track rotation cycles for intelligent delay
        rotation_cycles = 0
        max_cycles_before_delay = 5

        all_embeddings: List[List[float]] = []
        for batch_idx, batch in enumerate(batches, 1):
            # Check if we need to wait for RPM limits BEFORE making the request
            should_delay = self._wait_for_rpm_if_needed("embedding", rotation_cycles, max_cycles_before_delay)
            if should_delay:
                rotation_cycles += 1
            
            try:
                approx_tokens = sum(self._estimate_tokens(t) for t in batch)
                print(
                    f"🔮 Embedding batch {batch_idx}/{len(batches)} "
                    f"({len(batch)} texts, ≈{approx_tokens} tokens) using {embedding_model}"
                )

                # Refresh/validate API key before each request (per embedding model)
                self._configure_genai("embedding")

                resp = self.client.models.embed_content(
                    model=embedding_model,
                    contents=batch if len(batch) > 1 else batch[0],
                    config={"task_type": "retrieval_document"},
                )

                embs = getattr(resp, 'embeddings', None) or []
                embeddings = [e.values for e in embs] if embs else []

                # Optional dimensionality reduction
                if target_dim and embeddings and len(embeddings[0]) > target_dim:
                    embeddings = [emb[:target_dim] for emb in embeddings]

                all_embeddings.extend(embeddings)

                # Update usage (tokens counted via simple whitespace split)
                key = self.api_key_manager.get_key("embedding")
                model_name = self._get_model_name(embedding_model)
                self.api_key_manager.update_usage(
                    key, model_name, sum(len(t.split()) for t in batch)
                )

            except genai_errors.ClientError as e:
                print(f"�Y"" API key failed during embedding (client {getattr(e, 'code', '')}): {e}")
                try:
                    self.api_key_manager.rotate_key("embedding")
                    self._configure_genai("embedding")
                    rotation_cycles += 1
                except ValueError as rotate_error:
                    if "All API keys are over their limits" in str(rotate_error):
                        # All keys exhausted, implement delay based on rotation cycles
                        if rotation_cycles >= max_cycles_before_delay:
                            delay_time = 70  # 70 seconds to ensure RPM window resets
                            print(f"�?� All API keys exhausted after {rotation_cycles} rotations. "
                                  f"Waiting {delay_time}s for rate limits to reset...")
                            import time as _t
                            _t.sleep(delay_time)
                            rotation_cycles = 0  # Reset counter after delay
                            # Reset to first key and try again
                            self.api_key_manager.current_key_index = 0
                            self._configure_genai("embedding")
                        else:
                            raise rotate_error
                    else:
                        raise rotate_error
                
                # Retry the same batch with the next key
                retry_embeddings = self.embed(batch, model, target_dim)
                all_embeddings.extend(retry_embeddings)

            except (StopCandidateException, ResourceExhausted) as e:
                print(f"🔄 API key failed during embedding: {e}")
                try:
                    self.api_key_manager.rotate_key("embedding")
                    self._configure_genai("embedding")
                    rotation_cycles += 1
                except ValueError as rotate_error:
                    if "All API keys are over their limits" in str(rotate_error):
                        # All keys exhausted, implement delay based on rotation cycles
                        if rotation_cycles >= max_cycles_before_delay:
                            delay_time = 70  # 70 seconds to ensure RPM window resets
                            print(f"⏰ All API keys exhausted after {rotation_cycles} rotations. "
                                  f"Waiting {delay_time}s for rate limits to reset...")
                            import time
                            time.sleep(delay_time)
                            rotation_cycles = 0  # Reset counter after delay
                            # Reset to first key and try again
                            self.api_key_manager.current_key_index = 0
                            self._configure_genai("embedding")
                        else:
                            raise rotate_error
                    else:
                        raise rotate_error
                
                # Retry the same batch with the next key
                retry_embeddings = self.embed(batch, model, target_dim)
                all_embeddings.extend(retry_embeddings)
            except Exception as e:
                print(f"❌ Error generating embeddings for batch {batch_idx}: {e}")
                raise

        print(
            f"✨ Successfully generated {len(all_embeddings)} embeddings across {len(batches)} request(s)"
        )
        return all_embeddings

    def _wait_for_rpm_if_needed(self, model: str = "flash", rotation_cycles: int = 0, max_cycles: int = 5) -> bool:
        """Wait if the current key is approaching its RPM limit. Returns True if a rotation cycle occurred."""
        from .rate_limit_data import RATE_LIMITS
        
        rate_limit = RATE_LIMITS.get(model, RATE_LIMITS["flash"])
        # Use per-model timestamps
        current_timestamps = self.api_key_manager.rpm_timestamps[self.api_key_manager.current_key_index][model]
        
        now = time.time()
        # Clean up old timestamps (older than 60 seconds)
        current_timestamps[:] = [t for t in current_timestamps if now - t < 60]
        
        # If we're at or near the RPM limit, wait until the oldest request is >60s old
        if len(current_timestamps) >= rate_limit.per_minute:
            oldest_request = min(current_timestamps)
            wait_time = 60 - (now - oldest_request) + 1  # +1 second buffer
            
            if wait_time > 0:
                # Check if we should implement extended delay after multiple rotations
                if rotation_cycles >= max_cycles:
                    extended_wait = wait_time + 60  # Additional minute after 5+ rotations
                    print(f"⏳ RPM limit reached after {rotation_cycles} rotations ({len(current_timestamps)}/{rate_limit.per_minute}). "
                          f"Extended wait: {extended_wait:.1f}s to fully reset rate limits...")
                    time.sleep(extended_wait)
                else:
                    print(f"⏳ RPM limit reached ({len(current_timestamps)}/{rate_limit.per_minute}). "
                          f"Waiting {wait_time:.1f}s to respect rate limits...")
                    time.sleep(wait_time)
                return True  # Indicate a rotation cycle occurred
        return False
