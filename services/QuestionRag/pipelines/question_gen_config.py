"""Centralized configuration for question generation parameters."""

from dataclasses import dataclass
from typing import Optional

from config import load_config

@dataclass(frozen=True)
class QuestionGenerationConfig:
    """Centralized configuration for all question generation parameters."""

    # Model settings
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_calc_model: str = "gemini-2.5-flash"  # Separate model for calculation questions
    gemini_temperature: float = 0.8
    gemini_top_p: float = 0.9
    gemini_top_k: int = 50
    gemini_max_output_tokens: int = 10000

    # Generation settings
    theory_questions_per_request: int = 10
    calc_questions_per_request: int = 5
    request_delay_s: float = 1.5
    delay_jitter: float = 0.25
    request_attempts: int = 2

    # RAG settings
    rag_topk: int = 30
    rag_final_k: int = 12
    rag_tau: float = 0.35
    rag_min_similarity: float = 0.6
    rag_context_limit: int = 8
    rag_attempts: int = 2

    # Difficulty settings
    default_theory_difficulty_rank: int = 2  # Easy
    default_calculation_difficulty_rank: int = 2  # Easy

    # Advanced settings
    use_thinking: bool = False
    thinking_budget: int = 12700
    use_structured: bool = False
    latex_wrap_steps: bool = True
    resume: bool = True
    store_firestore: bool = True

    # Cache settings
    disable_cache_daily_reset: bool = False

    # Worker settings for parallel processing
    max_topic_workers: int = 3
    worker_timeout: int = 300  # 5 minutes
    worker_retry_attempts: int = 2
    enable_topic_parallelism: bool = True

    @classmethod
    def from_env(cls) -> 'QuestionGenerationConfig':
        """Create configuration using central config with environment fallbacks."""

        config = load_config()
        defaults = cls()

        def _coerce_int(value: object, fallback: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        def _coerce_float(value: object, fallback: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        def _coerce_bool(value: object, fallback: bool) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("1", "true", "yes")
            if isinstance(value, (int, float)):
                return bool(value)
            return fallback

        return cls(
            # Model settings
            gemini_model=str(getattr(config, "gemini_default_model", defaults.gemini_model)),
            gemini_calc_model=str(getattr(config, "gemini_calc_model", defaults.gemini_calc_model)),
            gemini_temperature=_coerce_float(getattr(config, "gemini_temperature", defaults.gemini_temperature), defaults.gemini_temperature),
            gemini_top_p=_coerce_float(getattr(config, "gemini_top_p", defaults.gemini_top_p), defaults.gemini_top_p),
            gemini_top_k=_coerce_int(getattr(config, "gemini_top_k", defaults.gemini_top_k), defaults.gemini_top_k),
            gemini_max_output_tokens=_coerce_int(getattr(config, "gemini_max_output_tokens", defaults.gemini_max_output_tokens), defaults.gemini_max_output_tokens),

            # Generation settings
            theory_questions_per_request=_coerce_int(getattr(config, "qg_theory_questions_per_request", defaults.theory_questions_per_request), defaults.theory_questions_per_request),
            calc_questions_per_request=_coerce_int(getattr(config, "qg_calc_questions_per_request", defaults.calc_questions_per_request), defaults.calc_questions_per_request),
            request_delay_s=_coerce_float(getattr(config, "qg_request_delay_s", defaults.request_delay_s), defaults.request_delay_s),
            delay_jitter=_coerce_float(getattr(config, "qg_delay_jitter", defaults.delay_jitter), defaults.delay_jitter),
            request_attempts=_coerce_int(getattr(config, "qg_request_attempts", defaults.request_attempts), defaults.request_attempts),

            # RAG settings
            rag_topk=_coerce_int(getattr(config, "qg_rag_topk", defaults.rag_topk), defaults.rag_topk),
            rag_final_k=_coerce_int(getattr(config, "qg_rag_final_k", defaults.rag_final_k), defaults.rag_final_k),
            rag_tau=_coerce_float(getattr(config, "qg_rag_tau", defaults.rag_tau), defaults.rag_tau),
            rag_min_similarity=_coerce_float(getattr(config, "qg_rag_min_similarity", defaults.rag_min_similarity), defaults.rag_min_similarity),
            rag_context_limit=_coerce_int(getattr(config, "qg_rag_context_limit", defaults.rag_context_limit), defaults.rag_context_limit),
            rag_attempts=_coerce_int(getattr(config, "qg_rag_attempts", defaults.rag_attempts), defaults.rag_attempts),

            # Difficulty settings
            default_theory_difficulty_rank=_coerce_int(getattr(config, "qg_default_theory_difficulty_rank", defaults.default_theory_difficulty_rank), defaults.default_theory_difficulty_rank),
            default_calculation_difficulty_rank=_coerce_int(getattr(config, "qg_default_calculation_difficulty_rank", defaults.default_calculation_difficulty_rank), defaults.default_calculation_difficulty_rank),

            # Advanced settings
            use_thinking=_coerce_bool(getattr(config, "gemini_use_thinking", defaults.use_thinking), defaults.use_thinking),
            thinking_budget=_coerce_int(getattr(config, "gemini_thinking_budget", defaults.thinking_budget), defaults.thinking_budget),
            use_structured=_coerce_bool(getattr(config, "coursegen_use_structured", defaults.use_structured), defaults.use_structured),
            latex_wrap_steps=_coerce_bool(getattr(config, "qg_latex_wrap_steps", defaults.latex_wrap_steps), defaults.latex_wrap_steps),
            resume=_coerce_bool(getattr(config, "qg_resume", defaults.resume), defaults.resume),
            store_firestore=_coerce_bool(getattr(config, "qg_store_firestore", defaults.store_firestore), defaults.store_firestore),

            # Cache settings
            disable_cache_daily_reset=_coerce_bool(getattr(config, "qg_disable_cache_daily_reset", defaults.disable_cache_daily_reset), defaults.disable_cache_daily_reset),

            # Worker settings
            max_topic_workers=_coerce_int(getattr(config, "qg_max_topic_workers", defaults.max_topic_workers), defaults.max_topic_workers),
            worker_timeout=_coerce_int(getattr(config, "qg_worker_timeout", defaults.worker_timeout), defaults.worker_timeout),
            worker_retry_attempts=_coerce_int(getattr(config, "qg_worker_retry_attempts", defaults.worker_retry_attempts), defaults.worker_retry_attempts),
            enable_topic_parallelism=_coerce_bool(getattr(config, "qg_enable_topic_parallelism", defaults.enable_topic_parallelism), defaults.enable_topic_parallelism),
        )

# Global configuration instance
_config_instance: Optional[QuestionGenerationConfig] = None

def get_question_gen_config() -> QuestionGenerationConfig:
    """Get the global question generation configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = QuestionGenerationConfig.from_env()
    return _config_instance

def reload_config() -> QuestionGenerationConfig:
    """Reload configuration from environment variables."""
    global _config_instance
    _config_instance = QuestionGenerationConfig.from_env()
    return _config_instance
