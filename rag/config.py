"""Runtime settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    chroma_dir: Path = ROOT / ".chroma"

    # "openai" (multilingual, recommended), "local" (Chroma's built-in MiniLM ONNX model,
    # offline but English-centric) or "auto" (openai if a key is set, else local).
    embedding_provider: str = _env("EMBEDDING_PROVIDER", "auto")
    openai_embedding_model: str = _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # "openai", "anthropic" or "auto" (whichever key is set, OpenAI first).
    llm_provider: str = _env("LLM_PROVIDER", "auto")
    openai_chat_model: str = _env("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    anthropic_model: str = _env("ANTHROPIC_MODEL", "claude-opus-5-5")

    top_k: int = int(_env("TOP_K", "5"))
    candidates: int = 20  # per retriever, before rank fusion
    timezone: str = _env("TIMEZONE", "Europe/Rome")

    @property
    def has_openai(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY", "").strip())

    @property
    def has_anthropic(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())

    def resolved_embedding_provider(self) -> str:
        if self.embedding_provider != "auto":
            return self.embedding_provider
        return "openai" if self.has_openai else "local"

    def resolved_llm_provider(self) -> str | None:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.has_openai:
            return "openai"
        if self.has_anthropic:
            return "anthropic"
        return None
