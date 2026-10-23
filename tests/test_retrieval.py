"""Retrieval tests with a deterministic fake embedder: no network, no model download."""

import hashlib

import pytest

from rag.config import Settings
from rag.corpus import load_corpus
from rag.retrieval import Embedder, HybridIndex, tokenize


class HashEmbedder(Embedder):
    name = "hash-test"

    def embed(self, texts):
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in texts]


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    settings = Settings(chroma_dir=tmp_path_factory.mktemp("chroma"))
    return HybridIndex(load_corpus(settings.data_dir), settings, embedder=HashEmbedder())


def test_tokenize_strips_accents_and_stopwords():
    assert tokenize("Wo ist Schloss Schöneck?") == ["schloss", "schoneck"]


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Falkenau honey", "Farm shop Falkenau"),
        ("Ciastel de Tor", "Museum Ladin Ciastel de Tor"),
        ("Wo ist die Bäckerei Gatterer?", "Gatterer bakery-confectionery with café"),
        ("Castello di Casteldarne", "Castle Ehrenburg/Casteldarne"),
    ],
)
def test_keyword_search_finds_named_entities(index, question, expected):
    top = index.search(question, k=3, mode="keyword")
    assert top[0].chunk.metadata["name"] == expected


def test_split_records_take_one_slot(index):
    names = [h.chunk.metadata["name"] for h in index.search("Falkenau speck honey stone pine", k=5)]
    assert names.count("Farm shop Falkenau") == 1


def test_index_is_reused_when_corpus_unchanged(index):
    again = HybridIndex(list(index.chunks.values()), index.settings, embedder=HashEmbedder())
    assert again.collection.count() == index.collection.count()
