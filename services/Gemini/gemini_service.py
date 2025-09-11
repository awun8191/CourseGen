"""Wrapper around the Google Gemini (new SDK) client with typed configuration.

- Rotates API keys via ApiKeyManager
- Supports typed generation config (GeminiConfig)
- Provides robust .generate(), .ocr(), and .embed() helpers
- .ocr() correctly constructs google-genai Parts for images + prompt

Requirements (pip):
    google-genai>=1.30
"""

from __future__ import annotations

import json
import os
import sys
import time
import re
import gc

from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar, get_args, get_origin

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_models.gemini_config import GeminiConfig  # noqa: E402

from google import genai  # noqa: E402
from google.genai import types as gtypes  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402
import httpx  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# --- Project imports ---
from ..Gemini.api_key_manager import ApiKeyManager  # noqa: E402  # relative import (same folder)
try:
    from ..Gemini.gemini_api_keys import GeminiApiKeys  # noqa: E402  # optional convenience provider
except Exception:
    GeminiApiKeys = None  # type: ignore

# OCR prompt (fallback text if the helper is missing)
try:
    from ..RAG.helpers import OCR_PROMPT
except Exception:
    OCR_PROMPT = (
        "Transcribe the exact visible text from this page image into plain text.\n"
        "Preserve the original line breaks and spacing where obvious.\n"
        "Do not include any explanations, headings, labels, JSON, or code fences.\n"
        "Return ONLY the text content as a single plain string."
    )

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
IMAGE_TOKEN_COST = 1000

# --- sample models (you can remove if unused) ---
class CourseOutline(BaseModel):
    course: str
    topics: List[str]
    description: str

class CoursesResponse(BaseModel):
    courses: List[CourseOutline]


class GeminiService:
    """Simple wrapper around the new Google GenAI client with typed configuration."""

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        model: str = DEFAULT_MODEL,
        generation_config: Optional[GeminiConfig] = None,
        api_key_manager: Optional[ApiKeyManager] = None,
    ) -> None:
        self.model = model
        self.default_config = generation_config or GeminiConfig()

        # Resolve API keys (prefer explicit manager, else provided list, else optional provider)
        if api_keys is None and api_key_manager is None and GeminiApiKeys is not None:
            try:
                gemini_keys = GeminiApiKeys()
                api_keys = gemini_keys.get_keys()
            except Exception:
                api_keys = None

        if api_key_manager is None:
            if not api_keys:
                # Allow empty manager; user can rely on GOOGLE_API_KEY env fallback in direct calls
                api_keys = []
            self.api_key_manager = ApiKeyManager(api_keys)
        else:
            self.api_key_manager = api_key_manager

        self._configure_genai()

    # -------------------- Client configuration --------------------

    def _configure_genai(self, model: str = "flash") -> None:
        """(Re)configure google-genai Client using the current API key for a model family."""
        # family is 'flash'|'lite'|'pro'|'embedding'
        family = self._get_model_name(model)
        api_key = self.api_key_manager.get_key(family)
        # If no key is available here, the caller must rely on GOOGLE_API_KEY/GEMINI_API_KEY envs
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            # Fall back to default client (will use env var if present)
            self.client = genai.Client()

    def _get_model_name(self, model_str: str) -> str:
        s = model_str.lower()
        if "lite" in s:
            return "lite"
        if "flash" in s:
            return "flash"
        if "pro" in s:
            return "pro"
        if "embedding" in s:
            return "embedding"
        return "flash"  # Default class

    # -------------------- Config translation --------------------

    def _to_generation_config(
        self, config: Optional[GeminiConfig | Dict[str, Any]]
    ) -> tuple[gtypes.GenerateContentConfig, Optional[list[dict[str, Any]]]]:
        """Convert GeminiConfig or plain dict into GenerateContentConfig."""
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
            return {"type": "string"}

        def _pydantic_to_simple_schema(model_cls: Type[BaseModel]) -> dict[str, Any]:
            schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            for name, field in model_cls.model_fields.items():
                py_t = field.annotation if field.annotation is not None else str
                prop_schema = _simple_type_schema(py_t)
                if field.description:
                    prop_schema["description"] = field.description
                schema["properties"][name] = prop_schema
                is_required = getattr(field, "is_required", False)
                if is_required:
                    schema["required"].append(name)
            if not schema["required"]:
                schema.pop("required")
            return schema

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
                # Attach schema (supported by google-genai)
                gen_config.response_schema = simple_schema  # type: ignore[attr-defined]
            except Exception:
                pass

        return gen_config, tools

    # -------------------- Low-level generate --------------------

    def _generate(
        self,
        parts: Iterable[Any],
        model: str,
        generation_config: Optional[GeminiConfig | Dict[str, Any]],
        response_model: Optional[Type[T]] = None,
        input_tokens: int = 0,
    ) -> T | Dict[str, Any]:
        """Perform a generation request against google-genai."""
        try:
            gen_config, tools = self._to_generation_config(generation_config)

            model_name = self._get_model_name(model)
            attempt = 0
            max_attempts = max(1, len(self.api_key_manager.api_keys) * 2)
            while True:
                try:
                    self._configure_genai(model_name)
                    response = self.client.models.generate_content(
                        model=model,
                        contents=list(parts),
                        config=gen_config,
                    )
                    break
                except (genai_errors.APIError, httpx.HTTPError, OSError) as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise e
                    time.sleep(min(2 ** attempt, 30))
                    try:
                        self.api_key_manager.rotate_key(model_name)
                    except Exception:
                        pass
                    continue

            # Extract text (robustly)
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

            # Update usage heuristically
            key = self.api_key_manager.get_key(model_name)
            tokens = max(1, input_tokens + len(response_text) // 4)
            if key:
                self.api_key_manager.update_usage(key, model_name, int(tokens))

            # If structured output isn't requested, return raw text
            if generation_config and isinstance(generation_config, GeminiConfig) and generation_config.response_schema is None:
                result = {"result": response_text.strip()}
                del response
                gc.collect()
                return result
            if isinstance(generation_config, dict) and generation_config.get("response_schema") is None:
                result = {"result": response_text.strip()}
                del response
                gc.collect()
                return result

            # Otherwise parse to JSON-like structure when possible
            def _extract_function_args(resp) -> Optional[Dict[str, Any]]:
                try:
                    candidates = getattr(resp, "candidates", None) or []
                    for cand in candidates:
                        content = getattr(cand, "content", None)
                        if not content:
                            continue
                        parts = getattr(content, "parts", None) or []
                        for p in parts:
                            fc = getattr(p, "function_call", None) or getattr(p, "functionCall", None)
                            if not fc:
                                continue
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
                                    pass
                            elif isinstance(args, dict):
                                return args
                    return None
                except Exception:
                    return None

            data: Any = None
            if tools:
                data = _extract_function_args(response)

            if data is None:
                cleaned_text = (response_text or "").strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()

                try:
                    data = json.loads(cleaned_text)
                except json.JSONDecodeError:
                    # Try to extract a balanced JSON object/array from the text
                    def _extract_balanced_json(s: str) -> Optional[str]:
                        start = None
                        opener = closer = None
                        for i, ch in enumerate(s):
                            if ch in "{[":
                                start = i
                                opener = ch
                                closer = "}" if ch == "{" else "]"
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
                                elif ch == "\\":
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
                        return None

                    candidate = _extract_balanced_json(cleaned_text)
                    if candidate is not None:
                        try:
                            data = json.loads(candidate)
                        except Exception:
                            # attempt a trivial repair
                            repaired = self._repair_truncated_json(cleaned_text)
                            data = json.loads(repaired) if repaired else {"result": None, "raw": response_text}
                    else:
                        repaired = self._repair_truncated_json(cleaned_text)
                        data = json.loads(repaired) if repaired else {"result": None, "raw": response_text}

            # Optional response_model validation/shaping
            if response_model:
                try:
                    if isinstance(data, dict) and isinstance(data.get("questions"), list):
                        def _sanitize(text: str) -> str:
                            if not isinstance(text, str):
                                return text
                            sents = re.split(r"(?<=[.!?])\s+", text.strip())
                            sents = [
                                s for s in sents
                                if not re.search(
                                    r"\b(let\s+me|let's|assume|apologies|re-?calculate|recheck)\b",
                                    s, flags=re.IGNORECASE
                                )
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
                result = response_model.model_validate(data)
                del response
                gc.collect()
                return result
            result = {"result": data}
            del response
            gc.collect()
            return result

        except genai_errors.ClientError as e:
            # 4xx (including 429)
            if getattr(e, "code", 0) == 429:
                model_name = self._get_model_name(model)
                try:
                    self.api_key_manager.rotate_key(model_name)
                    self._configure_genai(model_name)
                except Exception:
                    pass
                # retry once
                return self._generate(parts, model, generation_config, response_model)
            raise
        except genai_errors.ServerError:
            model_name = self._get_model_name(model)
            try:
                self.api_key_manager.rotate_key(model_name)
                self._configure_genai(model_name)
            except Exception:
                pass
            # retry once
            return self._generate(parts, model, generation_config, response_model)

    @staticmethod
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

    # -------------------- High-level generate --------------------

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
        # If a response model is provided, enable structured output automatically
        if response_model is not None:
            if gen_conf is None:
                gen_conf = self.default_config.model_copy(deep=True)
            elif isinstance(gen_conf, dict):
                gen_conf = GeminiConfig(**gen_conf)
            elif not isinstance(gen_conf, GeminiConfig):
                gen_conf = self.default_config.model_copy(deep=True)

            if getattr(gen_conf, "response_schema", None) is None:
                gen_conf.response_schema = response_model

        input_tokens = max(1, len(prompt) // 4)

        return self._generate(
            parts=[prompt],
            model=(model or self.model),
            generation_config=gen_conf,
            response_model=response_model,
            input_tokens=input_tokens,
        )

    # -------------------- OCR helper (fixed for google-genai v1.x) --------------------

    def ocr(
        self,
        images: List[Dict[str, Any]],
        prompt: str = OCR_PROMPT,
        response_model: Optional[Type[T]] = None,
        generation_config: Optional[GeminiConfig | Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> T | Dict[str, Any]:
        """
        OCR helper that builds proper multimodal Parts for the new google-genai SDK.

        Args:
            images: [{"mime_type": "image/png", "data": <bytes>}, ...]
            prompt: OCR instruction appended after images
            response_model: optional Pydantic model for structured output

        Returns:
            Plain text dict {"result": "..."} by default, or response_model instance if provided.
        """
        if not images:
            raise ValueError("No images provided for OCR")

        # Build typed Parts (required by google-genai v1.x)
        parts: List[Any] = []
        for img in images:
            mt = img.get("mime_type")
            data = img.get("data")
            if not mt or data is None:
                raise ValueError("Each image must include 'mime_type' and 'data' (bytes)")
            parts.append(gtypes.Part.from_bytes(mime_type=mt, data=data))

        # Append textual instruction last
        parts.append(prompt)

        # Select model/config
        use_model = model or self.model
        use_conf = generation_config if generation_config is not None else self.default_config
        prompt_tokens = len(prompt) // 4
        img_tokens = len(images) * IMAGE_TOKEN_COST

        try:
            return self._generate(
                parts=parts,
                model=use_model,
                generation_config=use_conf,
                response_model=response_model,
                input_tokens=prompt_tokens + img_tokens,
            )
        finally:
            del parts
            try:
                del images
            except Exception:
                pass
            gc.collect()

    # -------------------- Embeddings with batching & key rotation --------------------

    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens with a simple 4-chars-per-token heuristic."""
        return max(1, len(text) // 4)

    def _batch_texts(self, texts: List[str], max_tokens: int = 2000) -> List[List[str]]:
        """Split texts so each batch stays within ~max_tokens heuristic limit."""
        batches: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0
        for t in texts:
            t_tok = self._estimate_tokens(t)
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

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        target_dim: Optional[int] = 768
    ) -> List[List[float]]:
        """Generate embeddings while respecting ~2k tokens/request and RPM limits."""
        if not texts:
            return []

        embedding_model = model or EMBEDDING_MODEL
        batches = self._batch_texts(texts, max_tokens=2000)

        rotation_cycles = 0
        max_cycles_before_delay = 5

        all_embeddings: List[List[float]] = []
        for batch_idx, batch in enumerate(batches, 1):
            should_delay = self._wait_for_rpm_if_needed("embedding", rotation_cycles, max_cycles_before_delay)
            if should_delay:
                rotation_cycles += 1

            try:
                approx_tokens = sum(self._estimate_tokens(t) for t in batch)
                print(
                    f"🔮 Embedding batch {batch_idx}/{len(batches)} "
                    f"({len(batch)} texts, ≈{approx_tokens} tokens) using {embedding_model}"
                )

                # Configure per embedding family
                self._configure_genai("embedding")

                resp = self.client.models.embed_content(
                    model=embedding_model,
                    contents=batch if len(batch) > 1 else batch[0],
                    config={"task_type": "retrieval_document"},
                )

                embs = getattr(resp, 'embeddings', None) or []
                embeddings = [e.values for e in embs] if embs else []

                if target_dim and embeddings and len(embeddings[0]) > target_dim:
                    embeddings = [emb[:target_dim] for emb in embeddings]

                all_embeddings.extend(embeddings)

                key = self.api_key_manager.get_key("embedding")
                model_name = self._get_model_name(embedding_model)
                if key:
                    self.api_key_manager.update_usage(
                        key, model_name, sum(len(t.split()) for t in batch)
                    )

            except genai_errors.ClientError as e:
                print(f"⚠️ API key failed during embedding (client {getattr(e, 'code', '')}): {e}")
                try:
                    self.api_key_manager.rotate_key("embedding")
                    self._configure_genai("embedding")
                    rotation_cycles += 1
                except ValueError as rotate_error:
                    if "All API keys are over their limits" in str(rotate_error):
                        if rotation_cycles >= max_cycles_before_delay:
                            delay_time = 70
                            print(
                                f"⏰ All API keys exhausted after {rotation_cycles} rotations. "
                                f"Waiting {delay_time}s for rate limits to reset..."
                            )
                            import time as _t
                            _t.sleep(delay_time)
                            rotation_cycles = 0
                            self.api_key_manager.current_key_index = 0
                            self._configure_genai("embedding")
                        else:
                            raise rotate_error
                    else:
                        raise rotate_error

                retry_embeddings = self.embed(batch, model, target_dim)
                all_embeddings.extend(retry_embeddings)

            except Exception as e:
                print(f"❌ Error generating embeddings for batch {batch_idx}: {e}")
                raise

        print(f"✨ Successfully generated {len(all_embeddings)} embeddings across {len(batches)} request(s)")
        return all_embeddings

    def _wait_for_rpm_if_needed(self, model: str = "flash", rotation_cycles: int = 0, max_cycles: int = 5) -> bool:
        """Wait if the current key is approaching its RPM limit. Returns True if delay occurred."""
        from ..Gemini.rate_limit_data import RATE_LIMITS

        rate_limit = RATE_LIMITS.get(model, RATE_LIMITS["flash"])
        current_timestamps = self.api_key_manager.rpm_timestamps[self.api_key_manager.current_key_index][model]

        now = time.time()
        current_timestamps[:] = [t for t in current_timestamps if now - t < 60]

        if len(current_timestamps) >= rate_limit.per_minute:
            oldest_request = min(current_timestamps)
            wait_time = 60 - (now - oldest_request) + 1
            if wait_time > 0:
                if rotation_cycles >= max_cycles:
                    extended_wait = wait_time + 60
                    print(
                        f"⏳ RPM limit reached after {rotation_cycles} rotations "
                        f"({len(current_timestamps)}/{rate_limit.per_minute}). "
                        f"Extended wait: {extended_wait:.1f}s..."
                    )
                    time.sleep(extended_wait)  # type: ignore[name-defined]
                else:
                    print(
                        f"⏳ RPM limit reached "
                        f"({len(current_timestamps)}/{rate_limit.per_minute}). "
                        f"Waiting {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)  # type: ignore[name-defined]
                return True
        return False
