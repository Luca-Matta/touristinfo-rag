from rag.models import Category, Chunk
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.keyword import KeywordIndex, tokenize


def test_rrf_rewards_agreement_and_is_deterministic():
    fused = reciprocal_rank_fusion({"a": ["x", "y", "z"], "b": ["y", "x"]})
    assert [f.id for f in fused] == ["x", "y", "z"]  # x and y tie on score, broken by id
    assert fused[0].ranks == {"a": 1, "b": 2}
    assert fused[0].score == fused[1].score > fused[2].score


def test_tokenize_folds_accents_and_drops_stopwords():
    assert tokenize("Wo ist das Schloss Schöneck?") == ["schloss", "schoneck"]
    assert tokenize("Dove posso comprare il pane?") == ["comprare", "pane"]


def test_keyword_index_respects_allowed_ids():
    # (BM25 idf turns negative for a term in more than half the documents: keep it rarer.)
    docs = ["fresh bread bakery", "bread and sausages", "castle", "church", "pizza"]
    index = KeywordIndex([Chunk(str(i), str(i), text) for i, text in enumerate(docs)])
    assert set(index.search("bread", 5)) == {"0", "1"}
    assert index.search("bread bakery", 5)[0] == "0"
    assert index.search("bread", 5, allowed={"1"}) == ["1"]
    assert index.search("the", 5) == []  # only stopwords


def test_index_is_built_once_per_content(retriever, corpus, settings):
    from rag.retrieval import HybridRetriever

    from .conftest import HashEmbedder

    assert retriever.vectors.count() == len(retriever.chunks)
    again = HybridRetriever(corpus, settings, embedder=HashEmbedder())  # reuses the persisted collection
    assert again.vectors.count() == len(retriever.chunks)


def test_hybrid_finds_proper_nouns_and_multilingual_names(retriever):
    assert retriever.search("Falkenau")[0].record.id == "butcher-2625"
    assert retriever.search("Wo ist das Schloss Schöneck?")[0].record.id == "castle-101532"
    top = [h.record.id for h in retriever.search("macelleria")]
    assert "butcher-2625" in top or "butcher-overview" in top


def test_records_not_chunks_fill_the_k_slots(retriever):
    hits = retriever.search("Ladin museum Tor castle history", k=5)
    ids = [h.record.id for h in hits]
    assert len(ids) == len(set(ids)) == 5


def test_multiple_phrasings_are_fused(retriever):
    hits = retriever.search(["Dove posso comprare il pane?", "where to buy bread bakery"], k=3)
    assert any(h.record.category is Category.BAKERY for h in hits)
    assert set(hits[0].ranks) <= {"dense:0", "dense:1", "keyword:0", "keyword:1"}


def test_category_filter_and_unfiltered_fill(retriever):
    hits = retriever.search("opening hours", k=5, categories=[Category.PIZZERIA])
    assert [h.record.category for h in hits[:2]] == [Category.PIZZERIA, Category.PIZZERIA]
    assert len(hits) == 5  # only 2 pizzeria records exist: the rest comes from an unfiltered search


def test_modes(retriever):
    for mode in ("dense", "keyword", "hybrid"):
        hits = retriever.search("Falkenau farm shop", mode=mode)
        assert hits and all(name.startswith(mode) or mode == "hybrid" for name in hits[0].ranks)
