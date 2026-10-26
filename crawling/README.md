# Bonus: structured extraction from a destination page

```bash
cd crawling
pip install -r requirements.txt          # or, from the repo root: pip install -e ".[crawl]"
python crawl.py                          # Bar Oswald page from the exercise, JSON to stdout
python crawl.py <other valgardena.it address URL>
python crawl.py --format export          # same block format as data/ti_export_*.txt
python crawl.py --schema                 # JSON schema of the output
python crawl.py --html saved.html --no-llm   # offline
```

Output (validated by the pydantic model `Place`): `name`, `category` (the page's own
label), `opening_hours` (one entry per validity period and time range: dates, weekdays,
open/close times), `contact` (phone, email, website, structured address, coordinates),
`description` (2-3 sentences), `source_url`, and `notes` for anything the parser had to
assume or that looks wrong on the source page.

## Approach

**Facts come from parsing, prose comes from the LLM.** Phone, email, address,
coordinates and opening hours are extracted deterministically from the HTML, so they are
exact and reproducible and an LLM cannot make them up. The LLM writes only the
description, from the page text, and only when `OPENAI_API_KEY` is set; without a key a
template description is built from the extracted fields.

**Selectors are structural, not cosmetic.** No CSS class names: `tel:` and `mailto:`
links, the contact block (the smallest element around the phone link that also contains
a postcode line), tables whose header row is `Mo Tu We Th Fr Sa Su`, the
`#c_..._container` section anchors, route-planner links with `toLat`/`toLon`.

**Validated output.** Times must be `HH:MM`, weekdays are an enum, coordinates are range
checked, so a value that is extracted is well-formed. A selector that finds nothing is a
different failure: a missing name is an error, a missing phone or address is `null`, and
when no weekday table is found `opening_hours` is empty and `notes` (and stderr) say so.
What validation cannot catch is a table that parses but means something else, e.g. a
redesign that puts an icon in every day cell, closed days included.

**Polite fetching.** The crawler identifies itself, checks `robots.txt`, fetches one
page, and falls back to headless Playwright only when the plain request fails or the page
has no `<h1>` (rendered by JavaScript).

**Feeds the RAG.** `--format export` renders the place in the `### BEGIN POI` format of
`data/`, and a test checks that `rag.ingest` parses it without warnings. Two things do not
survive the round trip: the export format has year-less periods (`dd.mm`) and no email
field.

**Details the page throws at you:**
* The email is hidden by Cloudflare's obfuscation (`/cdn-cgi/l/email-protection#<hex>`):
  the first byte is the XOR key for the rest, decoded locally.
* A weekday counts as open if its cell has any text, icon or "active"-style class. If no
  weekday of a period is marked, it is assumed to apply every day, and `notes` says so.
* Overlapping periods and periods longer than a year are reported in `notes`. When this
  page was checked, its summer period ran "01.06.2026 - 30.09.2027", most likely a typo.
* `utm_*` tracking parameters are removed from the website link.

`tests/test_crawl.py` runs the parser against a synthetic page with the same structure
(`tests/fixtures/`), so the tests need no network. The live site returned HTTP 403 from
the environment this was built in, so the selectors were written against the page's
structure as published and should be checked once against the live page.
