# touristinfo.ai — minimal RAG pipeline for a tourist chatbot

A small, end-to-end Retrieval-Augmented Generation pipeline over the Kiens/Chienes
tourism corpus in `data/` (12 export files: B&Bs, hikes, bakeries, butchers, castles,
churches, pizzerias and five scraped web pages, with names and links in EN/DE/IT).

```
question ──► hybrid retrieval ──────────────► prompt with numbered sources ──► LLM ──► answer
             ├─ dense: Chroma (cosine)          + "open right now" computed      (OpenAI or
             └─ keyword: BM25                     in code from opening hours      Anthropic)
             fused with reciprocal rank fusion
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY and/or ANTHROPIC_API_KEY

python app.py "Where can I buy bread on Saturday afternoon?"
python app.py "Ist das ladinische Museum am Montag geöffnet?"
python app.py --debug "Dove posso mangiare una pizza?"   # shows chunks, scores and the prompt
python app.py                                            # interactive chat
python app.py --retrieve-only "Falkenau"                 # no API key needed

python -m eval.run_eval -v      # retrieval eval: dense vs keyword vs hybrid
python -m pytest -q             # unit tests (offline)
```

The index is built on first run and stored in `.chroma/`. It is rebuilt automatically
when the corpus or the embedding model changes (a content hash is kept on the collection).

**Providers.** With `OPENAI_API_KEY` set, embeddings use `text-embedding-3-small`
(multilingual) and answers use `gpt-4.1-mini`. With only `ANTHROPIC_API_KEY`, answers use
Claude (`claude-opus-5-5`) and embeddings fall back to Chroma's local MiniLM model
(Anthropic has no embeddings API). Everything is overridable in `.env`
(see `.env.example`).

## Project layout

| File | What it does |
| --- | --- |
| `rag/corpus.py` | Parses the exports into records, cleans the web pages, builds chunks |
| `rag/hours.py` | Parses opening hours, renders them, computes open/closed at a given time |
| `rag/retrieval.py` | Embeddings, Chroma collection, BM25, rank fusion |
| `rag/generation.py` | System prompt, context formatting, OpenAI / Anthropic clients |
| `rag/pipeline.py` | Wires it together: `RagPipeline().answer(question)` |
| `app.py` | CLI (single question, interactive, `--debug`, `--retrieve-only`) |
| `eval/` | 33 labelled questions (EN/DE/IT) and a retrieval evaluation script |
| `tests/` | Offline unit tests for parsing, hours logic and retrieval |

## What the data looks like, and what that implies

Reading the corpus before writing code shaped most of the decisions:

1. **The structured exports are already split into entities.** Each record sits between
   `### BEGIN POI|ACCOMMODATION|ACTIVITY|RESTAURANT` markers and is small (under 3 kB).
   So chunks follow records instead of fixed-size windows: a fixed window would cut
   "Opening hours" away from the name of the place they belong to. Only the two longest
   records are split, by paragraph, and every part repeats the record's title line.
2. **Opening hours are Python dict literals** (`{'period_start': '01.05', ..., 'days':
   ['tu', ...]}`). They are parsed with `ast.literal_eval` (safe, literals only), rendered
   as `Tue-Sun 10:00-18:00; closed Mon (from 01.05 to 05.11)`, and kept as JSON metadata.
   At question time the code computes `Status right now: closed now; opens tomorrow at
   10:00` and puts it in the prompt. LLMs are unreliable at weekday and date arithmetic,
   so that part is done in code.
3. **Three languages, but mostly English text.** Descriptions are English. German and
   Italian appear only in "Also known as" names and in link slugs, while guests ask in all
   three languages. To bridge this, each record chunk carries its DE/IT names plus a
   category label in all three languages (`Bakery · Bäckerei / Konditorei · panificio /
   panetteria`). Embeddings default to a multilingual model, and the LLM is told to answer
   in the guest's language and to prefer the link in that language (EN/DE/IT links are
   kept per record).
4. **The web scrapes are noisy.** They include cookie banners, "Read more" crumbs, a stray
   `à` glyph, and on the churches page every paragraph body appears twice. The cleaner
   drops the banner, the boilerplate lines and the duplicated halves/paragraphs, then
   chunks by heading.
5. **Small data-quality fixes.** Postcodes and house numbers like `39030.0` and
   `Dorfweg 22.0` become `39030` and `22`. Phones like `390474565238` get a `+39` prefix,
   a `-` placeholder phone is dropped, and the "this area" town placeholder is removed.
   The file name typo `Bucheries` is kept as the category id but labelled "Butcher".
6. **List questions** ("list all B&Bs", "which churches are there?") need *every* record
   of a category, which top-k does not guarantee. Each category therefore gets one
   overview chunk naming all its entries. Its text reuses the file's own description and
   example questions, which read exactly like user questions.

## Design decisions

**Hybrid retrieval (dense + BM25, fused with RRF).** Tourist questions mix proper nouns
("Falkenau", "Ciastel de Tor", "Casteldarne") that keyword search nails with paraphrases
and other-language questions that need embeddings. Reciprocal rank fusion
(`score = Σ 1/(60 + rank)`) merges the two lists without calibrating BM25 scores against
cosine similarity. The BM25 tokenizer lowercases, strips accents (`Schoneck` matches
`Schöneck`) and removes EN/DE/IT stopwords. Without the stopword step, "where can I find…"
matched the example questions inside every overview chunk. Parts of a split record count
as one result, so a long record cannot take several of the k slots.

**Chroma** as the vector store: embedded, persistent, zero setup. Embeddings are computed
by our own `Embedder` class and passed in explicitly, so switching between OpenAI and the
local model is a config change, and each model gets its own collection.

**Small-to-big context.** Retrieval works on chunks, but when one part of a split record
matches, the LLM gets the whole record.

**Grounded prompt.** Sources are numbered and carry their links and live opening status.
The system prompt asks for answers only from those sources, in the guest's language, with
`[n]` citations. When the answer is missing it should say so and point to the tourist
office, never invent hours, phones or links. "Hours not listed" means unknown, not closed.
The current time goes in the user turn rather than the system prompt, so the system
prompt stays identical across requests and can be cached.

**Deliberately left out (for a ~40 minute exercise):** a UI, streaming, multi-turn memory
and query rewriting, metadata filters, and re-ranking. See "Next steps".

## Evaluation and debugging

`eval/questions.json` has 33 questions (14 EN, 10 DE, 9 IT) labelled with the source(s)
that should be retrieved. `python -m eval.run_eval` reports hit@5 and MRR for each
retriever on its own and for the fusion. It needs no LLM and runs in seconds. When an
answer is wrong, this is the first thing to check: if retrieval missed, no prompt will
fix it.

Results with the **offline** MiniLM embedder (English-only model), k = 5:

| mode | hit@5 | MRR | EN | DE | IT |
| --- | --- | --- | --- | --- | --- |
| dense (MiniLM) | 88% | 0.78 | 14/14 | 9/10 | 6/9 |
| keyword (BM25) | 88% | 0.80 | 14/14 | 8/10 | 7/9 |
| **hybrid** | **97%** | **0.85** | 14/14 | 10/10 | 8/9 |

The two retrievers fail on different questions. BM25 misses "Wo bekomme ich frisches
Brot?" because "Brot" is not in the corpus. MiniLM misses "Castello di Casteldarne"
because it does not understand Italian. The fusion recovers both. The remaining miss
("Posso viaggiare in treno con la sedia a rotelle?") is purely cross-lingual: no Italian
word in it appears in the English page. That is what the multilingual OpenAI embeddings
are for (rerun `python -m eval.run_eval` with `OPENAI_API_KEY` set to compare).

How these numbers moved during development (the debugging trail):

* Raw opening-hours dicts were first embedded verbatim, and the parser silently missed
  them because `**Opening Hours for X:**` also matched the record-title pattern.
  Checking the rendered chunks caught it, and a test now guards it.
* BM25 initially ranked every overview chunk high for any "where can I…" question.
  Fixed with stopwords.
* A long record used two of the five slots. Fixed by grouping chunk parts per record
  inside the fusion, which took hybrid from 94% to 97%.

`python app.py --debug "..."` prints each retrieved chunk with its RRF score and its rank
in each retriever, followed by the exact prompt sent to the LLM.

## Limitations and next steps

* **Answer-level evaluation:** add expected facts per question and an LLM-as-judge (or
  string checks for hours and phones) to measure faithfulness, not only retrieval.
* **Multi-turn chat:** rewrite follow-up questions ("and on Sunday?") into standalone
  queries before retrieval.
* **Metadata filtering and tools:** questions like "what is open now?" or "B&Bs that
  allow dogs" are better answered by filtering structured fields (hours, services,
  category) than by similarity search. A natural next step is to give the LLM a small
  tool for that.
* **Re-ranking** with a cross-encoder once the corpus grows beyond a few dozen chunks.
* **Holidays and exceptions** are not in the hours data, so the "open now" status cannot
  know about them. The prompt keeps the wording factual.
* **Scale:** for thousands of records, swap the in-process BM25 for a search engine (or
  Qdrant/Weaviate hybrid search) and do incremental indexing instead of rebuild-on-hash.
