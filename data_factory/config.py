"""Configuration management for the Data Factory pipeline.

Uses Pydantic's BaseSettings for environment-aware configuration
with sensible defaults for development and production.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


def _load_env_file(path: str = ".env") -> None:
    """Load .env file into environment variables if not already set."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip("\"'")
        import os

        if key not in os.environ:
            os.environ[key] = val


_load_env_file()


class FactorySettings(BaseSettings):
    """Central configuration for the Data Factory pipeline.

    All values can be overridden via environment variables with the
    ``DATAFACTORY_`` prefix (e.g. ``DATAFACTORY_LOG_LEVEL=DEBUG``).

    Also auto-loads from ``.env`` file (standard ``KEY=value`` format).
    """

    # ── General ──────────────────────────────────────────────────────
    project_name: str = "Data Factory"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # ── Document Loading ─────────────────────────────────────────────
    max_document_size_mb: int = 50
    supported_extensions: List[str] = Field(
        default_factory=lambda: [".txt", ".md", ".pdf", ".csv", ".json", ".html"]
    )
    request_timeout: int = 30
    user_agent: str = "DataFactory/0.1.0"

    # ── Text Processing ──────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_length: int = 50
    max_chunk_length: int = 2048
    cleaning_steps: List[str] = Field(
        default_factory=lambda: [
            "strip_whitespace",
            "normalize_unicode",
            "remove_urls",
            "remove_html",
        ]
    )
    language: str = "en"

    # ── Task Generation ──────────────────────────────────────────────
    default_tasks: List[str] = Field(
        default_factory=lambda: ["qa", "summarization", "classification"]
    )
    max_chunks: int = 0
    max_examples_per_chunk: int = 3
    max_examples_per_task: int = 1000
    min_input_length: int = 10
    max_input_length: int = 4096

    # ── LLM Provider ─────────────────────────────────────────────────
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_api_key: Optional[str] = Field(default=None, exclude=True)
    llm_base_url: Optional[str] = None

    # ── Quality Control ──────────────────────────────────────────────
    quality_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "toxicity": 0.1,
            "bias": 0.2,
            "diversity": 0.15,
            "coherence": 0.3,
            "overall": 0.3,
        }
    )
    enable_toxicity_check: bool = True
    enable_bias_check: bool = True
    enable_diversity_check: bool = True
    enable_coherence_check: bool = True

    # ── API Integrations ─────────────────────────────────────────────
    decodo_api_key: Optional[str] = Field(default=None, exclude=True)
    decodo_base_url: str = "https://api.decodo.com/v1"
    enable_decodo_fallback: bool = True

    # ── Paths ────────────────────────────────────────────────────────
    output_dir: str = "output"
    cache_dir: str = ".cache"
    data_dir: str = "data"

    # ── Observability ────────────────────────────────────────────────
    track_costs: bool = True
    track_metadata: bool = True
    log_token_usage: bool = True

    model_config = {"env_prefix": "DATAFACTORY_", "extra": "ignore"}

    @model_validator(mode="after")
    def _resolve_llm_provider(self) -> "FactorySettings":
        """Auto-detect OpenRouter keys and set base URL accordingly."""
        key = self.llm_api_key
        if key and key.startswith("sk-or-v1-"):
            self.llm_provider = "openrouter"
            if not self.llm_base_url:
                self.llm_base_url = "https://openrouter.ai/api/v1"
        return self

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_positive(cls, v: int) -> int:
        if v < 64:
            raise ValueError("chunk_size must be at least 64 tokens")
        return v

    @field_validator("output_dir", "cache_dir", "data_dir")
    @classmethod
    def ensure_paths(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @property
    def quality_enabled(self) -> bool:
        return any(
            [
                self.enable_toxicity_check,
                self.enable_bias_check,
                self.enable_diversity_check,
                self.enable_coherence_check,
            ]
        )

    @property
    def llm_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }
        if self.llm_provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/data-factory"
            headers["X-Title"] = "Data Factory"
        return headers


settings = FactorySettings()
