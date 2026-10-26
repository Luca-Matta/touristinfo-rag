"""Runtime settings, read explicitly from the environment (see .env.example)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .errors import ConfigError

ROOT = Path(__file__).resolve().parent.parent

EmbeddingProvider = Literal["openai", "local"]
LLMProvider = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    index_dir: Path = ROOT / ".chroma"
    embedding_provider: EmbeddingProvider = "local"
    openai_embedding_model: str = "text-embedding-3-small"
    llm_provider: LLMProvider | None = None  # None: retrieval only
    openai_chat_model: str = "gpt-4.1-mini"
    anthropic_model: str = "claude-opus-5-5"
    planner_model: str | None = None  # defaults to the chat model of the chosen provider
    top_k: int = 5
    candidates: int = 20  # per retriever and query, before fusion
    timezone: str = "Europe/Rome"
    request_timeout_s: float = 60.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from environment variables; 'auto' providers follow the keys present."""
        env = os.environ if env is None else env

        def get(name: str, default: str = "") -> str:
            return env.get(name, "").strip() or default

        has_openai, has_anthropic = bool(get("OPENAI_API_KEY")), bool(get("ANTHROPIC_API_KEY"))

        embedding = get("EMBEDDING_PROVIDER", "auto")
        if embedding == "auto":
            embedding = "openai" if has_openai else "local"
        if embedding not in ("openai", "local"):
            raise ConfigError(f"EMBEDDING_PROVIDER must be openai, local or auto, not '{embedding}'")

        llm = get("LLM_PROVIDER", "auto")
        if llm == "auto":
            llm = "openai" if has_openai else "anthropic" if has_anthropic else ""
        if llm not in ("openai", "anthropic", ""):
            raise ConfigError(f"LLM_PROVIDER must be openai, anthropic or auto, not '{llm}'")

        defaults = cls()
        top_k = get("TOP_K", str(defaults.top_k))
        if not top_k.isdigit() or int(top_k) < 1:
            raise ConfigError(f"TOP_K must be a positive integer, not '{top_k}'")
        timezone = get("TIMEZONE", defaults.timezone)
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ConfigError(f"TIMEZONE must be an IANA time zone such as Europe/Rome, not '{timezone}'") from None
        return cls(
            data_dir=Path(get("DATA_DIR", str(defaults.data_dir))),
            index_dir=Path(get("INDEX_DIR", str(defaults.index_dir))),
            embedding_provider=cast(EmbeddingProvider, embedding),
            openai_embedding_model=get("OPENAI_EMBEDDING_MODEL", defaults.openai_embedding_model),
            llm_provider=cast(LLMProvider, llm) if llm else None,
            openai_chat_model=get("OPENAI_CHAT_MODEL", defaults.openai_chat_model),
            anthropic_model=get("ANTHROPIC_MODEL", defaults.anthropic_model),
            planner_model=get("PLANNER_MODEL") or None,
            top_k=int(top_k),
            timezone=timezone,
        )
