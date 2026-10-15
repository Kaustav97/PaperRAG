"""Configuration precedence: process environment > .env > TOML > defaults."""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

DIMENSION = 384
FORBIDDEN_OPTIONS = {"model", "messages", "stream", "api_key", "base_url", "organization", "project", "http_client", "default_headers", "default_query", "extra_headers", "extra_query"}


class StageModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model", mode="before")
    @classmethod
    def empty_model(cls, value: Any) -> Any:
        return value.strip() or None if isinstance(value, str) else value

    @field_validator("options")
    @classmethod
    def safe_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        bad = FORBIDDEN_OPTIONS.intersection(value)
        extra_body = value.get("extra_body", {})
        if isinstance(extra_body, dict):
            bad |= FORBIDDEN_OPTIONS.intersection(extra_body)
        if bad:
            raise ValueError(f"Reserved stage options: {', '.join(sorted(bad))}")
        return value


class Models(BaseModel):
    model_config = ConfigDict(extra="forbid")
    breakdown: StageModel = Field(default_factory=StageModel)
    research: StageModel = Field(default_factory=StageModel)
    synthesis: StageModel = Field(default_factory=StageModel)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: Path = Path("data")
    pdf_dir: Path = Path("papers")
    base_url: str | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embed_batch_size: int = Field(default=16, ge=1, le=256)
    threads: int = Field(default=2, ge=1, le=32)
    ocr_language: str = "eng"
    models: Models = Field(default_factory=Models)

    def require_llm(self) -> None:
        missing = []
        if not self.base_url:
            missing.append("PAPERRAG_BASE_URL")
        if not self.api_key or not self.api_key.get_secret_value().strip():
            missing.append("PAPERRAG_API_KEY")
        for stage in ("breakdown", "research", "synthesis"):
            if not getattr(self.models, stage).model:
                missing.append(f"PAPERRAG_{stage.upper()}_MODEL")
        if missing:
            raise ValueError("Research requires explicit configuration: " + ", ".join(missing) + ". Set these in .env or the environment; never paste secrets into chat.")

    def prepare_models(self) -> None:
        os.environ["HF_HOME"] = str(self.data_dir / "models" / "huggingface")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def tessdata_dir() -> Path:
    """Use the running environment, never the system Tesseract data."""
    directory = Path(sys.prefix) / "share" / "tessdata"
    os.environ["TESSDATA_PREFIX"] = str(directory)
    return directory


def load_settings(config_path: Path | None = None) -> Settings:
    path = (config_path or Path.cwd() / "paperrag.toml").resolve()
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    if "api_key" in raw:
        raise ValueError("Credentials must be environment-only, not stored in TOML")
    # Resolve TOML paths before applying cwd-relative environment overrides.
    for key in ("data_dir", "pdf_dir"):
        if key in raw:
            raw[key] = str((path.parent / raw[key]).resolve())
    env = {**dotenv_values(Path.cwd() / ".env"), **os.environ}
    fields = ("data_dir", "pdf_dir", "base_url", "embedding_model", "embed_batch_size", "threads", "ocr_language", "api_key")
    for key in fields:
        value = env.get("PAPERRAG_" + key.upper())
        if value is not None:
            raw[key] = value or None if key in ("base_url", "api_key") else value
    models = raw.setdefault("models", {})
    for stage in ("breakdown", "research", "synthesis"):
        key = "PAPERRAG_" + stage.upper() + "_MODEL"
        if key in env:
            models.setdefault(stage, {})["model"] = env[key]
    settings = Settings.model_validate(raw)
    settings.data_dir = settings.data_dir.expanduser().resolve()
    settings.pdf_dir = settings.pdf_dir.expanduser().resolve()
    tessdata_dir()
    return settings
