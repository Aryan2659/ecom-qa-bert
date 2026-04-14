"""
Central configuration loaded from environment variables.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def _bool(v: str, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# ── QA Model (extractive BERT) ─────────────────────────────────────
MODEL_NAME = os.getenv("HF_MODEL_NAME", "deepset/bert-base-cased-squad2")
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2500"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
WARMUP_ON_START = _bool(os.getenv("WARMUP_ON_START"), default=True)

# ── Sentiment Model (for review questions) ─────────────────────────
SENTIMENT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL_NAME",
    "distilbert-base-uncased-finetuned-sst-2-english",
)
SENTIMENT_MAX_REVIEWS = int(os.getenv("SENTIMENT_MAX_REVIEWS", "50"))
SENTIMENT_BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", "8"))

# ── Scraper ────────────────────────────────────────────────────────
SCRAPE_TIMEOUT = int(os.getenv("SCRAPE_TIMEOUT", "25"))
SCRAPE_MAX_RETRIES = int(os.getenv("SCRAPE_MAX_RETRIES", "3"))

# Playwright (headless browser)
PLAYWRIGHT_ENABLED = _bool(os.getenv("PLAYWRIGHT_ENABLED"), default=True)
PLAYWRIGHT_TIMEOUT_MS = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "30000"))
PLAYWRIGHT_MAX_REVIEWS = int(os.getenv("PLAYWRIGHT_MAX_REVIEWS", "50"))
PLAYWRIGHT_HEADLESS = _bool(os.getenv("PLAYWRIGHT_HEADLESS"), default=True)

# ── Rate limiting ──────────────────────────────────────────────────
RATE_LIMIT_ENABLED = _bool(os.getenv("RATE_LIMIT_ENABLED"), default=True)
RATE_LIMIT_SCRAPE = os.getenv("RATE_LIMIT_SCRAPE", "10 per minute")
RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "30 per minute")
RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "200 per hour")

# ── Persistence ────────────────────────────────────────────────────
_default_db = "/data/history.db" if Path("/data").exists() and os.access("/data", os.W_OK) else "history.db"
DB_PATH = os.getenv("DB_PATH", _default_db)
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "100"))

# ── Server ─────────────────────────────────────────────────────────
PORT = int(os.getenv("PORT", "7860"))
DEBUG = _bool(os.getenv("FLASK_DEBUG"), default=False)
