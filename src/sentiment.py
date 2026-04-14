"""
Sentiment analysis for product reviews.

Uses DistilBERT-SST (distilbert-base-uncased-finetuned-sst-2-english)
to classify each review as POSITIVE or NEGATIVE with a confidence score.
Aggregates the results into a summary the UI can render.

Singleton pattern matching src/model.py — load once per worker.
"""
import logging
import threading
import time
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config

logger = logging.getLogger(__name__)

_model: Optional[AutoModelForSequenceClassification] = None
_tokenizer: Optional[AutoTokenizer] = None
_load_lock = threading.Lock()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_sentiment(warmup: bool = None) -> None:
    """Load sentiment model + tokenizer once. Safe to call repeatedly."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return

    with _load_lock:
        if _model is not None and _tokenizer is not None:
            return

        start = time.time()
        logger.info(f"Loading sentiment model '{config.SENTIMENT_MODEL_NAME}' on {_device}…")
        _tokenizer = AutoTokenizer.from_pretrained(config.SENTIMENT_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(config.SENTIMENT_MODEL_NAME)
        _model.to(_device)
        _model.eval()
        logger.info(f"Sentiment model loaded in {time.time() - start:.1f}s")

    do_warmup = config.WARMUP_ON_START if warmup is None else warmup
    if do_warmup:
        try:
            analyze_reviews([{"text": "This product is amazing."}])
        except Exception:
            logger.warning("Sentiment warmup failed", exc_info=True)


def _require_sentiment():
    if _model is None or _tokenizer is None:
        init_sentiment()


def analyze_reviews(reviews: list[dict]) -> dict:
    """
    Analyze a list of review dicts. Each dict should have a 'text' key;
    optional 'rating' and 'title' keys are passed through.

    Returns:
      {
        "total": int,
        "positive_count": int,
        "negative_count": int,
        "positive_pct": float,
        "negative_pct": float,
        "avg_confidence": float,
        "overall_sentiment": "positive" | "negative" | "mixed",
        "top_positive": [ {text, confidence, rating?}, ... up to 3 ],
        "top_negative": [ {text, confidence, rating?}, ... up to 3 ],
        "inference_time_ms": int,
      }
    """
    _require_sentiment()

    # Filter + clamp
    clean = [r for r in reviews if r.get("text") and len(r["text"].strip()) >= 10]
    clean = clean[: config.SENTIMENT_MAX_REVIEWS]

    if not clean:
        return {
            "total": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "avg_confidence": 0.0,
            "overall_sentiment": "unknown",
            "top_positive": [],
            "top_negative": [],
            "inference_time_ms": 0,
            "error": "No usable reviews found.",
        }

    texts = [r["text"][:512] for r in clean]  # model handles 512 tokens anyway

    t0 = time.time()
    batch_size = config.SENTIMENT_BATCH_SIZE
    all_labels: list[str] = []
    all_confidences: list[float] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = _tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            logits = _model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu()

            for p in probs:
                pos_prob = float(p[1])  # index 1 = POSITIVE in SST-2
                neg_prob = float(p[0])
                if pos_prob >= neg_prob:
                    all_labels.append("positive")
                    all_confidences.append(pos_prob)
                else:
                    all_labels.append("negative")
                    all_confidences.append(neg_prob)

    inference_ms = int((time.time() - t0) * 1000)

    # Aggregate
    pos_count = sum(1 for lbl in all_labels if lbl == "positive")
    neg_count = len(all_labels) - pos_count
    total = len(all_labels)
    pos_pct = (pos_count / total) * 100 if total else 0.0
    neg_pct = (neg_count / total) * 100 if total else 0.0

    if pos_pct >= 65:
        overall = "positive"
    elif neg_pct >= 65:
        overall = "negative"
    else:
        overall = "mixed"

    # Top positive / negative by confidence
    enriched = []
    for r, lbl, conf in zip(clean, all_labels, all_confidences):
        enriched.append({
            "text": r["text"],
            "title": r.get("title"),
            "rating": r.get("rating"),
            "label": lbl,
            "confidence": round(conf, 4),
        })

    top_positive = sorted(
        (e for e in enriched if e["label"] == "positive"),
        key=lambda e: e["confidence"],
        reverse=True,
    )[:3]
    top_negative = sorted(
        (e for e in enriched if e["label"] == "negative"),
        key=lambda e: e["confidence"],
        reverse=True,
    )[:3]

    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    logger.info(
        f"Sentiment: {total} reviews → {pos_count} pos / {neg_count} neg "
        f"({pos_pct:.0f}% positive) in {inference_ms}ms"
    )

    return {
        "total": total,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_pct": round(pos_pct, 1),
        "negative_pct": round(neg_pct, 1),
        "avg_confidence": round(avg_conf, 4),
        "overall_sentiment": overall,
        "top_positive": [
            {k: v for k, v in e.items() if k != "label"} for e in top_positive
        ],
        "top_negative": [
            {k: v for k, v in e.items() if k != "label"} for e in top_negative
        ],
        "inference_time_ms": inference_ms,
    }
