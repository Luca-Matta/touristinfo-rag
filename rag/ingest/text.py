"""Small text normalisation helpers shared by the parsers."""

from __future__ import annotations

import re
import unicodedata


def clean(s: str) -> str:
    """NFC, collapse spaces, and drop the float artefacts of the export ('39030.0' -> '39030')."""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\b(\d+)\.0\b", r"\1", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def normalise_phone(s: str) -> str | None:
    """'-' -> None; '390474565238' -> '+39 0474565238'; anything else unchanged."""
    s = s.strip()
    if s in {"", "-"}:
        return None
    if s.isdigit() and s.startswith("39") and len(s) >= 11:
        return f"+39 {s[2:]}"
    return s


def slugify(s: str, max_len: int = 48) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:max_len].rstrip("-")


def fold(s: str) -> str:
    """Lowercase and strip accents, for matching ('Schöneck' == 'schoneck')."""
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))
