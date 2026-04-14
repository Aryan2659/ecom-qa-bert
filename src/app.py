"""
Flask application entrypoint (v2 with question routing + sentiment).

Routes
  GET   /                     — main UI
  GET   /healthz              — liveness probe
  POST  /api/scrape           — scrape URL (Playwright + legacy fallback)
  POST  /api/predict          — run QA: router picks BERT, sentiment, or both
  GET   /api/history          — list stored Q&A
  DELETE /api/history/<id>    — remove one entry
  DELETE /api/history         — clear all
"""
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from . import config, db, router
from .model import init_model, predict_qa
from .sentiment import init_sentiment, analyze_reviews
from .scraper import scrape_url

FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(_PROJECT_ROOT / "templates"),
        static_folder=str(_PROJECT_ROOT / "static"),
    )

    limiter = None
    if config.RATE_LIMIT_ENABLED:
        try:
            from flask_limiter import Limiter
            from flask_limiter.util import get_remote_address
            limiter = Limiter(
                get_remote_address, app=app,
                default_limits=[config.RATE_LIMIT_DEFAULT],
                storage_uri="memory://", strategy="fixed-window",
            )
            logger.info("Rate limiting enabled")
        except ImportError:
            logger.warning("flask-limiter not installed; rate limiting disabled")

    def _limit(rule: str):
        if limiter is None:
            return lambda fn: fn
        return limiter.limit(rule)

    db.init_db()

    # ── Routes ──────────────────────────────────────────────────

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/healthz")
    def healthz():
        return jsonify({
            "status": "ok",
            "qa_model": config.MODEL_NAME,
            "sentiment_model": config.SENTIMENT_MODEL_NAME,
            "playwright_enabled": config.PLAYWRIGHT_ENABLED,
        }), 200

    @app.post("/api/scrape")
    @_limit(config.RATE_LIMIT_SCRAPE)
    def api_scrape():
        payload = request.get_json(silent=True) or {}
        url = (payload.get("url") or "").strip()
        if not url:
            return jsonify({"error": "URL is required."}), 400
        try:
            result = scrape_url(url)
            if result.get("error"):
                return jsonify(result), 400
            return jsonify(result)
        except Exception as e:
            logger.exception("Scraping failed")
            return jsonify({"error": f"Unexpected error: {e}"}), 500

    @app.post("/api/predict")
    @_limit(config.RATE_LIMIT_PREDICT)
    def api_predict():
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        context = (payload.get("context") or "").strip()
        reviews = payload.get("reviews") or []
        source_url = (payload.get("source_url") or "").strip() or None
        source_type = (payload.get("source_type") or "").strip() or None
        product_title = (payload.get("product_title") or "").strip() or None

        if not question:
            return jsonify({"error": "Question is required."}), 400
        if not context and not reviews:
            return jsonify({"error": "Context or reviews required."}), 400
        if len(question) > 500:
            return jsonify({"error": "Question is too long (max 500)."}), 400

        # Route the question
        intent = router.classify(question)
        classification = router.explain(question)
        response = {"intent": intent, "classification": classification}

        qa_result = None
        sentiment_result = None

        # Run extractive QA
        if intent in ("spec", "both"):
            if context and len(context) >= 20:
                try:
                    qa_result = predict_qa(question, context)
                    response["qa"] = qa_result
                except ValueError as e:
                    response["qa_error"] = str(e)
                except Exception as e:
                    logger.exception("QA prediction failed")
                    response["qa_error"] = f"Inference error: {e}"
            else:
                response["qa_error"] = "Not enough product text to run extractive QA."

        # Run sentiment analysis
        if intent in ("review", "both"):
            if reviews:
                try:
                    sentiment_result = analyze_reviews(reviews)
                    response["sentiment"] = sentiment_result
                except Exception as e:
                    logger.exception("Sentiment analysis failed")
                    response["sentiment_error"] = f"Sentiment error: {e}"
            else:
                response["sentiment_error"] = (
                    "No reviews were scraped. Amazon often blocks review "
                    "extraction from cloud IPs. Try Flipkart, or paste review "
                    "text in Text mode."
                )

        # Persist the primary answer for history
        try:
            if qa_result:
                db.save_qa(
                    question=question,
                    answer=qa_result["answer"],
                    confidence=qa_result["confidence"],
                    confidence_level=qa_result["confidence_level"],
                    inference_ms=qa_result["inference_time_ms"],
                    source_url=source_url, source_type=source_type,
                    product_title=product_title,
                )
            elif sentiment_result:
                summary_answer = (
                    f"{sentiment_result['overall_sentiment'].upper()}: "
                    f"{sentiment_result['positive_pct']:.0f}% positive, "
                    f"{sentiment_result['negative_pct']:.0f}% negative "
                    f"across {sentiment_result['total']} reviews"
                )
                db.save_qa(
                    question=question,
                    answer=summary_answer,
                    confidence=sentiment_result["avg_confidence"],
                    confidence_level=(
                        "high" if sentiment_result["avg_confidence"] > 0.8
                        else "medium" if sentiment_result["avg_confidence"] > 0.5
                        else "low"
                    ),
                    inference_ms=sentiment_result["inference_time_ms"],
                    source_url=source_url, source_type=source_type,
                    product_title=product_title,
                )
        except Exception:
            logger.exception("Failed to persist Q&A")

        return jsonify(response)

    @app.get("/api/history")
    def api_history():
        limit = request.args.get("limit", type=int) or config.HISTORY_LIMIT
        limit = max(1, min(limit, 500))
        try:
            return jsonify({"items": db.list_history(limit=limit)})
        except Exception as e:
            logger.exception("History listing failed")
            return jsonify({"error": str(e)}), 500

    @app.delete("/api/history/<int:entry_id>")
    def api_history_delete(entry_id: int):
        try:
            ok = db.delete_entry(entry_id)
            return jsonify({"deleted": ok, "id": entry_id}), (200 if ok else 404)
        except Exception as e:
            logger.exception("History delete failed")
            return jsonify({"error": str(e)}), 500

    @app.delete("/api/history")
    def api_history_clear():
        try:
            n = db.clear_history()
            return jsonify({"cleared": n})
        except Exception as e:
            logger.exception("History clear failed")
            return jsonify({"error": str(e)}), 500

    logger.info("Initializing BERT QA model…")
    init_model()
    logger.info("Initializing sentiment model…")
    init_sentiment()
    logger.info("Models ready.")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG)
