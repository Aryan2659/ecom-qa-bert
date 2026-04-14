"""
Tests for the SQLite history module.

Uses a temp DB path via monkeypatching config so we never touch
a real production database.
"""
import os

import pytest

from src import config, db


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the db module at a fresh temp file for each test."""
    db_file = tmp_path / "test_history.db"
    monkeypatch.setattr(config, "DB_PATH", str(db_file))
    # Reset the "initialized" flag so init_db runs against the new path
    monkeypatch.setattr(db, "_initialized", False)
    yield str(db_file)
    # Cleanup is automatic — tmp_path is removed by pytest


def test_init_db_creates_schema(tmp_db):
    db.init_db()
    assert os.path.exists(tmp_db)


def test_save_and_list(tmp_db):
    rid = db.save_qa(
        question="What is the battery?",
        answer="5000 mAh",
        confidence=0.87,
        confidence_level="high",
        inference_ms=450,
        source_url="https://amazon.in/dp/XYZ",
        source_type="amazon",
        product_title="Phone Model X",
    )
    assert isinstance(rid, int) and rid > 0

    items = db.list_history()
    assert len(items) == 1
    row = items[0]
    assert row["question"] == "What is the battery?"
    assert row["answer"] == "5000 mAh"
    assert row["confidence"] == pytest.approx(0.87)
    assert row["confidence_level"] == "high"
    assert row["source_type"] == "amazon"


def test_list_ordering_newest_first(tmp_db):
    for i in range(3):
        db.save_qa(
            question=f"Q{i}",
            answer=f"A{i}",
            confidence=0.5,
            confidence_level="medium",
            inference_ms=100,
        )
    items = db.list_history()
    # Newest first — last inserted (Q2) should be first
    assert items[0]["question"] == "Q2"
    assert items[-1]["question"] == "Q0"


def test_delete_entry(tmp_db):
    rid = db.save_qa(
        question="to delete", answer="x", confidence=0.1,
        confidence_level="low", inference_ms=50,
    )
    assert db.delete_entry(rid) is True
    assert db.delete_entry(rid) is False  # already gone
    assert db.list_history() == []


def test_clear_history(tmp_db):
    for i in range(5):
        db.save_qa(
            question=f"q{i}", answer="a", confidence=0.5,
            confidence_level="medium", inference_ms=10,
        )
    assert db.clear_history() == 5
    assert db.list_history() == []


def test_list_limit(tmp_db):
    for i in range(10):
        db.save_qa(
            question=f"q{i}", answer="a", confidence=0.5,
            confidence_level="medium", inference_ms=10,
        )
    assert len(db.list_history(limit=3)) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
