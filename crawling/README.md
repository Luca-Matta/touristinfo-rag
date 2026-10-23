# Bonus: structured extraction from a destination page

```bash
pip install -r crawling/requirements.txt
python crawling/crawl.py                 # Bar Oswald page from the exercise, JSON to stdout
python crawling/crawl.py <other valgardena.it address URL>
python crawling/crawl.py --html saved.html --no-llm   # offline
```

Output fields: `name`, `opening_hours` (one entry per validity period: dates, weekdays,
open/close times), `contact` (phone, email, website, structured address, coordinates),
`description` (2-3 sentences), plus `source_url` and `notes` for anything the parser had
to assume or that looks wrong on the source page.

## Approach

**Facts come from parsing, prose comes from the LLM.** Phone, email, address, coordinates
and opening hours are extracted deterministically from the HTML, so they are exact and
reproducible, and an LLM can't make them up. The LLM writes only the 2-3 sentence
description, from the page text, and only when `OPENAI_API_KEY` is set. Without a key,
a template description is built from the extracted fields and the page's own
description line.

**Selectors are structural, not cosmetic.** The parser doesn't rely on CSS class names.
It looks for `tel:` and `mailto:` links, the contact block (the smallest element around
the phone link that also contains a postcode line), tables whose header row is
`Mo Tu We Th Fr Sa Su`, the `#c_..._container` section anchors, and route-planner links
with `toLat`/`toLon` parameters. Small redesigns are less likely to break it.

**Details the page throws at you:**
* The email is hidden by Cloudflare's obfuscation (`/cdn-cgi/l/email-protection#<hex>`).
  It is decoded locally: the first byte is the XOR key for the rest.
* Opening hours come as one table per validity period. A weekday counts as open if its
  cell has any text, icon or "active"-style class. If no weekday is marked at all, the
  period is assumed to apply every day, and `notes` says so.
* Overlapping validity periods are reported in `notes`. When this page was checked, its
  summer period ran "01.06.2026 - 30.09.2027", most likely a typo for 2026.
* Tracking parameters (`utm_*`) are removed from the website link.
* Fetching uses plain `requests` with a browser User-Agent, then falls back to headless
  Playwright if the request fails or the page has no `<h1>` (JS-rendered).

`tests/test_crawl.py` runs the parser against a synthetic page with the same structure,
so the tests need no network.
