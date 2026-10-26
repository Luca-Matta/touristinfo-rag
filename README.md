# touristinfo.ai: a multilingual RAG assistant for Kiens/Chienes

A retrieval-augmented chatbot over the Kiens/Chienes tourism corpus in `data/` (seven
structured exports: B&Bs, hikes, bakeries, butchers, castles, churches, pizzerias, plus
five scraped web pages). Guests ask in English, German or Italian, the corpus is mostly
English. Plus the bonus crawler in `crawling/`.

```
guest message + history
   │
   ▼
query planner (1 small LLM call, JSON schema)
   standalone question · English query · language · categories · day/time · services · "list all"
   │
   ▼
hybrid retrieval: Chroma (cosine) + BM25, over both phrasings, fused per record with RRF
   + structured matches: places offering every requested service, every place of a category
   │
   ▼
answer LLM: static system prompt + history + <source> blocks
   (opening status computed in code for the moment asked about, link in the guest's language)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # = pip install -e ".[crawl,dev]"
cp .env.example .env                     # add OPENAI_API_KEY and/or ANTHROPIC_API_KEY

python app.py "Where can I buy bread on Saturday afternoon?"
python app.py --debug "Ist das Museum Ladin am Montag geöffnet?"   # plan + sources with ranks
python app.py                                                      # chat, follow-ups work
python app.py --retrieve-only "Falkenau"                           # no API key needed

pytest                                          # 79 offline tests
python -m eval.run_eval retrieval               # offline retrieval eval, dev split
python -m eval.run_eval answers --judge -v      # end-to-end eval (LLM calls)
python crawling/crawl.py                        # bonus, see crawling/README.md
```

The vector index is built on first run in `.chroma/` and rebuilt automatically when the
chunks or the embedding model change (a content hash is stored on the collection).

**Providers.** With `OPENAI_API_KEY`, embeddings use `text-embedding-3-small`
(multilingual) and the LLM is `gpt-4.1-mini`. With only `ANTHROPIC_API_KEY`, the LLM is
Claude (`claude-opus-5-5`, low effort) and embeddings use Chroma's local MiniLM model,
since Anthropic has no embeddings API. Without any key the app runs retrieval only.
Everything can be overridden in `.env`, including a cheaper `PLANNER_MODEL`.

**Cost.** A question is two LLM calls: the planner (~600 input tokens) and the answer
(~2-4k input tokens of sources). With `gpt-4.1-mini` that is roughly $0.002 per question;
indexing the whole corpus with OpenAI embeddings costs well under one cent. The Claude
default (`claude-opus-5-5`) is far more expensive per token, so with Anthropic set
`PLANNER_MODEL` (and ideally `ANTHROPIC_MODEL`) to a small model.

## Layout

| Path | Responsibility |
| --- | --- |
| `rag/models.py` | `Record` (one thing a guest can ask about), `Category` with EN/DE/IT labels |
| `rag/ingest/` | Table-driven parser for the exports, web-page cleaner, `Corpus` with structured lookups |
| `rag/hours.py` | Opening hours: parse, describe, status at a moment (`OPEN`/`CLOSED`/`NOT_LISTED`/`OUT_OF_SEASON`) |
| `rag/chunking.py` | Record-level chunks, split by paragraph/sentence only when too long |
| `rag/retrieval/` | `Embedder` protocol, Chroma store, BM25 index, RRF, `HybridRetriever` |
| `rag/planner.py` | `QueryPlan` (pydantic) and the LLM planner with a strict JSON schema |
| `rag/generation/` | `ChatModel` protocol (OpenAI, Anthropic, logging decorator), prompts |
| `rag/assistant.py` | `TouristAssistant.ask()`: plan, retrieve, select sources, answer |
| `rag/cli.py`, `app.py` | Command line chat |
| `eval/` | 69 labelled questions (dev/test), retrieval and answer evaluation |
| `tests/` | Offline tests: a hash embedder and a scripted fake LLM stand in for the APIs |
| `crawling/` | Bonus: destination page to validated JSON or to the export format |

## What the data implies

Reading the corpus first shaped most decisions:

1. **The exports are already split into entities**, each small. So a record is the unit:
   one chunk per record (only records over 1,500 characters are split, and each piece
   repeats the title line), and the LLM always receives the whole record. Fixed-size
   windows would separate "opening hours" from the place they belong to.
2. **Each record gets two views.** `search_text` (name, DE/IT aliases, category labels in
   three languages, services, description) is what is embedded and keyword-indexed; URLs,
   phone numbers and coordinates would only add noise. `context_text` is what the LLM
   reads: every field, rendered consistently, with the detail link in the guest's language.
3. **Opening hours are Python dict literals.** They are parsed with `ast.literal_eval`
   (literals only, no code execution) and every "is it open?" question is answered in
   code for the moment the guest asks about ("now", "on Sunday", "tomorrow at 8"), because
   LLMs are unreliable at weekday and date arithmetic. That includes the date itself: the
   planner only names the day as the guest said it (`"sunday"`, `"tomorrow"`, `"12-24"`)
   and `rag/planner.py` computes the calendar date, so "Sunday" asked on a Monday is the
   coming Sunday, never yesterday. The exports list the days a place
   *is* open, so an unlisted day is reported as "no hours listed (possibly closed)" rather
   than "closed".
4. **Guests write in three languages, the text is English.** Three things bridge it: the
   planner's English rewrite of every question, multilingual category labels and aliases
   in the indexed text (so BM25 matches "Bäckerei" and "macelleria"), and multilingual
   embeddings when OpenAI is configured.
5. **Similarity search cannot answer every shape of question.** "Which B&B has a sauna?"
   (every B&B looks alike) and "list all churches" (k is too small) are answered from
   structured fields: the planner extracts `services` and `list_all`, and those matches
   are placed first among the sources. Each category also has an overview record listing
   all its entries.
6. **The data is messy** and the parser says so. Unknown fields produce warnings (the
   current corpus parses with zero), `39030.0` becomes `39030`, `390474565238` gets a
   `+39`, placeholder towns and phones are dropped. The web scrapes lose their cookie
   banner, "Read more" crumbs, a stray glyph and the duplicated paragraph bodies of the
   churches page, and are split by heading.

## Design decisions

**Hybrid retrieval.** Proper nouns ("Falkenau", "Ciastel de Tor") are where BM25 is
strong; paraphrases and other languages are where embeddings are. Reciprocal rank fusion
(`Σ 1/(60 + rank)`) merges the ranked lists without calibrating BM25 against cosine.
Each (phrasing, retriever) pair is one list, chunks are collapsed to records inside each
list so a split record cannot take two slots, and a category filter falls back to an
unfiltered search when it leaves fewer than k results.

**A planner instead of a bigger prompt.** One cheap structured-output call resolves
follow-ups ("und am Sonntag?"), produces the English query, detects the language and
extracts the day, time, categories and services. Its output is validated with pydantic,
and an invalid plan falls back to the raw question, so the planner cannot break the
pipeline. A valid but wrong plan can still hurt an answer, so the parts that matter most
are not left to the model: the code computes the date from the day it names, and removes
the destination name ("in Kiens/Chienes") from the queries unless the guest wrote it.

**Grounding and prompt injection.** Sources are reference data in `<source>` tags with
their content HTML-escaped, so a scraped page cannot close its tag and pose as
instructions. The system prompt asks for answers only from the sources, with `[n]`
citations, and to say so when the answer is missing. It contains nothing per-request
(time and sources go in the user turn), so it is byte-identical across requests. At about
250 tokens it is still below the providers' minimum for prompt caching; the layout pays
off once the prompt grows (with Anthropic it would also need a `cache_control` marker).

**Small, swappable parts.** `Embedder` and `ChatModel` are protocols; the tests use a
hash embedder and a scripted fake LLM, so all 79 tests run offline in under a second.
SDK exceptions become a `ProviderError` with a readable message. Settings are read once,
explicitly, from the environment; an unknown provider, a `TOP_K` that is not a positive
integer or an unknown `TIMEZONE` fails at start-up. `ruff`, `mypy
--strict` and the tests run in CI together with the offline retrieval eval.

## Evaluation

`eval/questions.json` holds 69 questions (29 EN, 21 DE, 19 IT) with the record ids that
can answer them, facts the answer must contain (with accepted spellings), phrases it must
not contain, a reference note where the data has a trap (Vanea serves breakfast only on
request, one church is locked), optional chat history for follow-ups, a fixed "now"
(Monday 26.10.2026, 10:00) so hours answers are reproducible, and three questions the
data cannot answer. They are split per language into `dev` (46, inspect and tune) and
`test` (23, only report).

* `eval.run_eval retrieval` reports hit@5 and MRR per retriever and language. Offline it
  uses the raw question; `--planner` adds the LLM planning step (plans are cached in
  `eval/.cache`, so reruns are free).
* `eval.run_eval answers` runs the whole pipeline and checks the required facts (a number
  must match as a whole number, so "5 rooms" is not found inside a phone number) and the
  forbidden phrases. `--judge` adds an LLM judge that is a different, stronger model than
  the one answering (`gpt-4.1` by default, `--judge-model` to change it). It sees the
  expected records and the reference note next to the retrieved sources, lists the
  issues it finds, and grades grounded, correct, complete and language. Every row,
  including the plan and the sources, is appended to `eval/.cache/answers-*.jsonl` as
  it is produced.

**Retrieval**, hit@5 / MRR over the 67 questions with an expected source:

| configuration | dense | keyword | **hybrid** |
| --- | --- | --- | --- |
| offline: local MiniLM, raw question | 84% / 0.71 | 87% / 0.83 | **96% / 0.85** |
| OpenAI embeddings + planner | 100% / 0.99 | 100% / 0.98 | **100% / 0.99** |

Offline, the retrievers fail on different questions (BM25 does not know "Brot", MiniLM
does not understand Italian), which is why fusion helps; the remaining misses are
follow-ups and cross-lingual questions, the cases the planner exists for.

**Answers.** A first run (OpenAI embeddings, `gpt-4.1-mini` for planner, answers and
judge, dev split) scored 46/46 on every check. That number was too good: rereading the
answers against `data/` found 6 with problems the checks let through. Two were wrong
(q22 said both B&Bs include breakfast, Vanea only serves it on request; q48 called every
church open and then St. Martin locked), four were imprecise or unsupported (praise for
the bakeries, Market Engl without its afternoon hours, a phone number credited to the
tourist office, a detail about Bolzano station). The fact checks had passed by
coincidence (a "5" inside a phone number), and the judge was the answering model, seeing
only the retrieved sources. The checks and the judge described above are the response;
`python -m eval.run_eval answers --judge --split all` gives the current numbers.

**How the eval changed the code.** The first run with a key showed four misses: the
planner appended "in Kiens/Chienes" to most queries, and since every source mentions the
area, that phrase diluted both retrievers. Telling it not to in the prompt was not enough
(it still did in 8 of 67 English queries), so the code now strips the destination name
from the plan unless the guest wrote it. The answer eval then showed one question
answered with "no information" although dense search had found the page: BM25 over the
Italian phrasing produced noise lists that outvoted it, so keyword search now uses the
English rewrite only.

**Read these numbers with care.** Those fixes were found by looking at misses across all
questions, so the test split is no longer untouched, and at 100% retrieval is saturated:
it can catch regressions but not show further improvement. One question moves a score by
1.5 to 4.5 points, and one run is one sample: two runs give different wording, so
compare runs by their failures, not by a point of difference. Six test questions are
translations of dev questions. A credible benchmark would need questions written by
others or taken from real guest logs, and several runs.

## Limitations and next steps

* **No holidays or exceptions** in the hours data, so the computed status cannot know
  about them; the wording stays factual ("no hours listed").
* **Two LLM calls per question.** For latency, the planner could run on a small model
  (`PLANNER_MODEL`) or be skipped for single-turn questions in the guest's corpus language.
* **Overnight hours** ("18:00 - 01:00") are not modelled: the corpus has none, but bars
  imported with the crawler could.
* **Service matching** is substring-based over the services list, with a small synonym
  table ("pet friendly" finds "Dogs allowed", "gluten-free" finds "Cuisine without
  glutine") and negated entries skipped ("No pets or domestic animals" never matches
  "pets"). Wordings outside the table are missed; an embedding match would cover them.
* **Re-ranking** with a cross-encoder becomes worth it once the corpus grows beyond a few
  hundred records; so does a search engine with native hybrid search instead of
  in-process BM25, and incremental indexing instead of rebuild-on-hash.
* **Streaming** answers and a web UI were out of scope.
