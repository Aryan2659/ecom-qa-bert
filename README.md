---
title: E-Commerce QA with BERT + Sentiment
emoji: 🛍️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Product QA with BERT and sentiment analysis
---

# E-Commerce Product QA (v2) — BERT + Sentiment + Playwright

A multi-model NLP web app that answers any question about a product.
**Spec questions** go to extractive BERT QA. **Review questions** go to a
sentiment analysis pipeline. A rule-based router picks the right path.
Scraping is done with **Playwright** so JS-rendered reviews are captured.

## What's new vs. v1

| Capability | v1 | **v2** |
|---|---|---|
| Answer factual spec questions | ✅ | ✅ |
| Answer review/opinion questions | ❌ | ✅ |
| Scrape JavaScript-rendered content | ❌ | ✅ |
| Scrape product reviews | ❌ | ✅ (when not blocked) |
| Multi-model architecture | single BERT | BERT + DistilBERT-SST |
| Question intent routing | none | rule-based router |

---

## Architecture

```
                        ┌─────────────────────┐
        Question ──────▶│  Intent Router      │ ───┐
                        │  (keyword-based)    │    │
                        └─────────────────────┘    │
                                                   │
                ┌──── "spec" ───┐  "both"   ┌──── "review" ───┐
                │               │    │      │                  │
                ▼               ▼    ▼      ▼                  ▼
        ┌──────────────┐       (both run)          ┌────────────────────┐
        │ BERT QA      │                           │ DistilBERT-SST     │
        │ (extractive) │                           │ (sentiment)        │
        │ from specs + │                           │ batched over all   │
        │ features     │                           │ scraped reviews    │
        └──────────────┘                           └────────────────────┘
              │                                            │
              ▼                                            ▼
      Answer span +                             Overall verdict +
      confidence +                              pos/neg % + top 3
      token breakdown                           positive + top 3 negative
```

## Models

| Purpose | Model | Size |
|---|---|---|
| Extractive QA (specs) | `deepset/bert-base-cased-squad2` | ~440 MB |
| Sentiment classification | `distilbert-base-uncased-finetuned-sst-2-english` | ~260 MB |

Both pre-downloaded at Docker build time for fast cold starts.

## Scraping

Uses **Playwright** (headless Chromium) to render JavaScript and extract:
- Title, features, description, specs (for BERT context)
- Overall rating + review count
- Up to 50 individual reviews (for sentiment analysis)

**Reality check — Amazon blocking:**
Cloud IPs (HuggingFace Spaces, AWS, GCP) are aggressively flagged as
non-residential. Amazon serves CAPTCHAs to ~40-50% of headless requests
from such IPs. When this happens, the app gracefully falls back to the
legacy `requests`-based scraper (no reviews) and tells the user what
happened. **Flipkart has better success rates.**

---

## Routes

| Method | Path | Purpose |
|---|---|---|
| GET    | `/`                    | Main UI |
| GET    | `/healthz`             | Liveness probe |
| POST   | `/api/scrape`          | `{url}` → product text + reviews |
| POST   | `/api/predict`         | `{question, context, reviews}` → qa and/or sentiment |
| GET    | `/api/history?limit=N` | List saved Q&A |
| DELETE | `/api/history/<id>`    | Remove one entry |
| DELETE | `/api/history`         | Clear all |

### `POST /api/predict` response shape (v2)

```json
{
  "intent": "spec" | "review" | "both",
  "classification": { "intent": "...", "review_keywords": [...], ... },
  "qa": { "answer": "...", "confidence": 0.87, ... },
  "sentiment": {
    "overall_sentiment": "positive",
    "positive_pct": 83.3,
    "negative_pct": 16.7,
    "top_positive": [{"text": "...", "confidence": 0.95, "rating": 5}],
    "top_negative": [...],
    "total": 12
  },
  "qa_error": "...",
  "sentiment_error": "..."
}
```

Only branches that ran appear in the response. If the router picked a
branch but the data was missing (e.g., review question but no reviews
scraped), a descriptive `*_error` field explains why.

---

## Project Structure

```
ecom-qa-bert-v2/
├── Dockerfile                   # HF Spaces Docker SDK, Playwright + both models baked in
├── README.md                    # (this file, with HF front-matter)
├── requirements.txt             # Pinned versions
├── .env.example                 # Config template
├── .dockerignore / .gitignore
├── src/
│   ├── app.py                   # Flask routes + orchestration
│   ├── config.py                # Env-driven config (both models, Playwright settings)
│   ├── router.py                # Question intent classifier (spec / review / both)
│   ├── model.py                 # BERT QA singleton
│   ├── sentiment.py             # DistilBERT sentiment singleton
│   ├── scraper.py               # Playwright wrapper (with legacy fallback)
│   ├── scraper_legacy.py        # Requests-based scraper (fallback)
│   └── db.py                    # SQLite history
├── templates/index.html
├── static/{css,js}/
└── tests/
    ├── test_app.py              # Route tests with mocked models
    ├── test_db.py               # SQLite CRUD
    ├── test_router.py           # Intent classifier tests
    └── test_scraper.py          # Scraper parser tests
```

---

## Deploy to HuggingFace Spaces

1. Create a new Space at <https://huggingface.co/new-space>
   - SDK: **Docker**
   - Template: **Blank**
   - Hardware: **CPU basic (free)** — works fine
2. Clone the Space locally and copy these files in
3. `git add . && git commit -m "v2" && git push`
4. First build takes **~18-20 minutes** (downloads Chromium + 2 models, ~2.6 GB image). Subsequent builds are cached and faster.
5. App boots at `https://<username>-<space-name>.hf.space`

### Configuration (optional)

Set any of these as Space secrets/variables:

| Variable | Default | Purpose |
|---|---|---|
| `HF_MODEL_NAME` | `deepset/bert-base-cased-squad2` | Swap QA model |
| `SENTIMENT_MODEL_NAME` | `distilbert-base-uncased-finetuned-sst-2-english` | Swap sentiment model |
| `PLAYWRIGHT_ENABLED` | `true` | Disable to skip headless browser |
| `PLAYWRIGHT_MAX_REVIEWS` | `50` | Max reviews scraped per page |
| `SENTIMENT_MAX_REVIEWS` | `50` | Max reviews analyzed |
| `RATE_LIMIT_SCRAPE` | `10 per minute` | Per-IP scrape limit |

---

## Local Development

```bash
git clone https://github.com/<you>/ecom-qa-bert-v2 && cd ecom-qa-bert-v2

python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m playwright install chromium

cp .env.example .env                   # optional
python -m src.app
```

Then open <http://localhost:7860>.

### Run with Docker

```bash
docker build -t ecom-qa-v2 .
docker run --rm -p 7860:7860 -v ecom_qa_data:/data ecom-qa-v2
```

---

## Testing

```bash
pytest tests/ -v
```

All tests mock network + models, so they run in under 2 seconds with no
BERT, no Chromium, no Amazon.

Current coverage:
- `test_router.py` — 23 tests for intent classification
- `test_scraper.py` — 8 tests for HTML parsers + error paths
- `test_db.py` — 6 tests for SQLite persistence
- `test_app.py` — 11 tests for Flask routes + routing logic

---

## Known limitations

1. **Amazon often blocks review scraping** from cloud IPs. The app detects this and tells the user. Flipkart works better.
2. **Sentiment analysis is aggregate** — if reviews are genuinely mixed, the verdict is "mixed", which is accurate but less satisfying than a clear answer.
3. **Router is rule-based** — it handles common cases well but can misclassify creative phrasings. A small fine-tuned classifier would be more robust; left as future work.
4. **Single Gunicorn worker** — both models in one process use ~900 MB RAM. More workers would duplicate model weights.

## License

MIT
