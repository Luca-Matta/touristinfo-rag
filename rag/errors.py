"""Errors the CLI can explain to a user instead of printing a traceback."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


class RagError(Exception):
    """Base class: anything raised as a RagError has a message fit for the user."""


class ConfigError(RagError):
    """Missing or invalid configuration (API keys, provider names)."""


class ProviderError(RagError):
    """An LLM or embedding API call failed (auth, rate limit, network, bad response)."""


@contextmanager
def provider_errors(what: str) -> Iterator[None]:
    """Translate OpenAI / Anthropic SDK exceptions into a ProviderError with a clear message."""
    try:
        yield
    except Exception as exc:
        message = _describe(exc)
        if message is None:
            raise
        raise ProviderError(f"{what} failed: {message}") from exc


def _describe(exc: Exception) -> str | None:
    for module in ("openai", "anthropic"):
        try:
            sdk = __import__(module)
        except ImportError:
            continue
        if isinstance(exc, sdk.AuthenticationError):
            return "the API key was rejected (check .env)"
        if isinstance(exc, sdk.RateLimitError):
            return "rate limit or quota exceeded, try again later"
        if isinstance(exc, sdk.APIConnectionError):
            return "could not reach the API (network problem or timeout)"
        if isinstance(exc, sdk.APIStatusError):
            return f"the API returned HTTP {exc.status_code}: {exc.message}"
    return None
