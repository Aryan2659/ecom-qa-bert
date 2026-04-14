"""
Tests for the question intent router.
"""
import pytest

from src.router import classify, explain


@pytest.mark.parametrize("q", [
    "What is the battery capacity?",
    "What is the processor?",
    "How much RAM does it have?",
    "What is the screen size?",
    "What is the refresh rate?",
    "What colors are available?",
    "Does it have a headphone jack?",
    "What is the weight?",
])
def test_spec_questions(q):
    assert classify(q) == "spec", f"Expected 'spec' for: {q}"


@pytest.mark.parametrize("q", [
    "Are the reviews good?",
    "Are reviews bad or good?",
    "What do people think?",
    "Is it worth buying?",
    "What do users complain about?",
    "What are the customer reviews saying?",
    "Should I buy this?",
    "Any good reviews?",
    "Is this product reliable?",
])
def test_review_questions(q):
    assert classify(q) == "review", f"Expected 'review' for: {q}"


@pytest.mark.parametrize("q", [
    "Is the camera good?",               # camera=spec, good=review
    "Is the battery life reliable?",     # battery=spec, reliable=review
    "How good is the display?",          # display=spec, good=review
])
def test_ambiguous_questions(q):
    assert classify(q) == "both", f"Expected 'both' for: {q}"


def test_empty_question_defaults_to_spec():
    assert classify("") == "spec"
    assert classify("   ") == "spec"


def test_unmatched_question_defaults_to_spec():
    assert classify("hello world") == "spec"


def test_explain_returns_keywords():
    result = explain("Are reviews good?")
    assert result["intent"] == "review"
    assert "reviews" in result["review_keywords"]
    assert "good" in result["review_keywords"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
