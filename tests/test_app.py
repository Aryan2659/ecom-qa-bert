"""
Smoke tests for Flask app routes (v2).

Mocks init_model, predict_qa, init_sentiment, analyze_reviews — so tests
run in milliseconds without loading any ML models or hitting the network.
"""
from unittest.mock import patch

import pytest

from src import config, db


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(db, "_initialized", False)
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(config, "WARMUP_ON_START", False)

    with patch("src.app.init_model") as m1, \
         patch("src.app.init_sentiment") as m2, \
         patch("src.app.predict_qa") as mock_predict, \
         patch("src.app.analyze_reviews") as mock_sentiment:
        m1.return_value = None
        m2.return_value = None

        mock_predict.return_value = {
            "answer": "5000 mAh",
            "confidence": 0.87,
            "confidence_pct": "87.0%",
            "confidence_level": "high",
            "answer_start_char": 10, "answer_end_char": 18,
            "context_used": "Battery: 5000 mAh capacity.",
            "tokens": [], "num_tokens": 0,
            "inference_time_ms": 42,
        }
        mock_sentiment.return_value = {
            "total": 10,
            "positive_count": 8, "negative_count": 2,
            "positive_pct": 80.0, "negative_pct": 20.0,
            "avg_confidence": 0.9,
            "overall_sentiment": "positive",
            "top_positive": [{"text": "Great!", "confidence": 0.95, "rating": 5}],
            "top_negative": [{"text": "Bad battery", "confidence": 0.88, "rating": 2}],
            "inference_time_ms": 120,
        }

        from src.app import create_app
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert "qa_model" in body
    assert "sentiment_model" in body


def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"E-Commerce Product QA" in resp.data


def test_predict_requires_question(client):
    resp = client.post("/api/predict", json={"context": "x" * 100})
    assert resp.status_code == 400


def test_predict_requires_context_or_reviews(client):
    resp = client.post("/api/predict", json={"question": "What?"})
    assert resp.status_code == 400


def test_spec_question_routes_to_qa_only(client):
    resp = client.post("/api/predict", json={
        "question": "What is the battery capacity?",
        "context": "Battery: 5000 mAh. Long enough context text here for validation.",
        "reviews": [],
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["intent"] == "spec"
    assert "qa" in body
    assert body["qa"]["answer"] == "5000 mAh"
    assert "sentiment" not in body


def test_review_question_routes_to_sentiment_only(client):
    resp = client.post("/api/predict", json={
        "question": "Are the reviews good?",
        "context": "Some product context here that won't be used.",
        "reviews": [
            {"text": "Great product!"}, {"text": "Love it"},
            {"text": "Bad battery"}, {"text": "Broken on arrival"},
        ],
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["intent"] == "review"
    assert "sentiment" in body
    assert body["sentiment"]["overall_sentiment"] == "positive"
    assert "qa" not in body


def test_ambiguous_question_routes_to_both(client):
    resp = client.post("/api/predict", json={
        "question": "Is the camera good?",
        "context": "Camera: 200 MP. Long enough context here to pass validation.",
        "reviews": [{"text": "Camera is amazing"}],
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["intent"] == "both"
    assert "qa" in body
    assert "sentiment" in body


def test_review_question_without_reviews_returns_error_branch(client):
    resp = client.post("/api/predict", json={
        "question": "Are the reviews good?",
        "context": "Context long enough to pass validation.",
        "reviews": [],
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["intent"] == "review"
    assert "sentiment_error" in body


def test_spec_question_persists_qa_to_history(client):
    client.post("/api/predict", json={
        "question": "What is the battery?",
        "context": "Battery: 5000 mAh capacity and long enough context here.",
        "source_url": "https://example.com/x",
        "source_type": "amazon",
        "product_title": "Phone",
    })
    hist = client.get("/api/history").get_json()
    assert len(hist["items"]) == 1
    assert hist["items"][0]["product_title"] == "Phone"
    assert hist["items"][0]["answer"] == "5000 mAh"


def test_review_question_persists_summary_to_history(client):
    client.post("/api/predict", json={
        "question": "Are reviews good?",
        "context": "Some context text here just to pass length validation.",
        "reviews": [{"text": "Great!"}, {"text": "Amazing!"}],
        "product_title": "Phone",
    })
    hist = client.get("/api/history").get_json()
    assert len(hist["items"]) == 1
    # Summary format: "POSITIVE: 80% positive, 20% negative across 10 reviews"
    assert "positive" in hist["items"][0]["answer"].lower()


def test_scrape_requires_url(client):
    resp = client.post("/api/scrape", json={})
    assert resp.status_code == 400


def test_history_delete_and_clear(client):
    client.post("/api/predict", json={
        "question": "What?",
        "context": "Context text that is clearly long enough for validation.",
    })
    hist = client.get("/api/history").get_json()
    entry_id = hist["items"][0]["id"]

    resp = client.delete(f"/api/history/{entry_id}")
    assert resp.status_code == 200
    assert resp.get_json()["deleted"] is True

    resp = client.delete("/api/history/99999")
    assert resp.status_code == 404

    client.post("/api/predict", json={
        "question": "Q2?",
        "context": "Another context sufficiently long to pass validation.",
    })
    resp = client.delete("/api/history")
    assert resp.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
