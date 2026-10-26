"""BM25 keyword index over chunks.

Keyword search is what makes proper nouns work ("Falkenau", "Ciastel de Tor") and what
matches the DE/IT names and category labels in the chunks. Tokens are accent-folded so
"Schoneck" matches "Schöneck", and EN/DE/IT function words are dropped: without that,
"where can I find ..." matches the example questions in every overview record.
"""

from __future__ import annotations

import re
from collections.abc import Container, Sequence

from rank_bm25 import BM25Okapi

from ..ingest.text import fold
from ..models import Chunk

STOPWORDS = frozenset(
    """a an and any are at be can do does find for from has have here how i in is it me my
    near nearby of on or some the there this to what when where which who with you your
    am auf bei bis das dem den der die ein eine einen es gibt hier ich im ist kann man mit
    nach oder sind und von was welche welcher wie wo zu
    al alla c che ci con da del della di dove e il la le lo mi per posso qui si sono su un
    una uno vicino""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", fold(text)) if t not in STOPWORDS]


class KeywordIndex:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._ids = [c.id for c in chunks]
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, n: int, allowed: Container[str] | None = None) -> list[str]:
        """Chunk ids by descending BM25 score; only positive scores, only ``allowed`` ids if given."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = [self._ids[i] for i in ranked if scores[i] > 0 and (allowed is None or self._ids[i] in allowed)]
        return hits[:n]
