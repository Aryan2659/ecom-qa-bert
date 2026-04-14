"""
Question intent router.

Classifies a user's question into one of:
  - "spec"      → factual question; route to extractive BERT QA
  - "review"    → sentiment/opinion question; route to sentiment analysis
  - "both"      → ambiguous; run both and show both

Rule-based on purpose — a small ML classifier here would be overkill and
less transparent. Keywords are tuned for e-commerce product QA.
"""
import re
from typing import Literal

Intent = Literal["spec", "review", "both"]

# Strong review/opinion signals — if any match, it's a review question
REVIEW_KEYWORDS = {
    # Direct review references
    "review", "reviews", "reviewer", "reviewers", "rating", "ratings",
    "stars", "customer", "customers", "user", "users", "buyer", "buyers",
    "people", "feedback", "testimonial", "testimonials",

    # Opinion verbs
    "think", "thinks", "thought", "say", "says", "saying", "said",
    "feel", "feels", "felt", "opinion", "opinions", "experience", "experiences",

    # Subjective quality adjectives (as standalone concepts)
    "worth", "recommend", "recommended", "satisfied", "happy", "disappointed",
    "complaints", "complain", "problem", "problems", "issue", "issues",

    # Sentiment polarity words in question form
    "good", "bad", "great", "terrible", "awful", "amazing", "excellent",
    "poor", "quality", "reliable", "durable",

    # Sentiment question patterns
    "should i buy", "worth buying", "worth it", "worth the",
    "is it good", "is it bad", "is this good", "is this bad",
    "any good", "how good", "how bad",
}

# Strong spec signals — technical attributes that definitely use BERT
SPEC_KEYWORDS = {
    # Hardware
    "processor", "chipset", "cpu", "gpu", "ram", "memory", "storage",
    "battery", "mah", "capacity", "charging", "watt", "watts",
    "display", "screen", "resolution", "refresh", "hertz", "hz",
    "camera", "megapixel", "mp", "lens", "aperture",
    "weight", "height", "width", "depth", "dimensions", "size",
    "color", "colour", "colors", "colours",
    "port", "ports", "headphone", "jack", "usb", "bluetooth", "wifi",
    "os", "android", "ios", "version",
    "speaker", "speakers", "audio", "microphone",
    "price", "cost", "mrp",

    # Feature words
    "feature", "features", "specification", "specifications", "specs",
    "include", "includes", "included", "support", "supports",
}


def _tokenize(text: str) -> set[str]:
    """Lowercased word tokens from text."""
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def _has_phrase(text: str, phrase: str) -> bool:
    return phrase in (text or "").lower()


def classify(question: str) -> Intent:
    """Classify question intent. Deterministic, no side effects."""
    if not question or not question.strip():
        return "spec"

    q_lower = question.lower().strip()
    tokens = _tokenize(q_lower)

    # Phrase-level check for multi-word review patterns first
    review_phrases = [
        "should i buy", "worth buying", "worth it", "worth the",
        "is it good", "is it bad", "is this good", "is this bad",
        "any good", "how good", "how bad", "what do people",
        "what do users", "what do customers", "what do reviewers",
    ]
    has_review_phrase = any(_has_phrase(q_lower, p) for p in review_phrases)

    review_hits = len(tokens & REVIEW_KEYWORDS) + (1 if has_review_phrase else 0)
    spec_hits = len(tokens & SPEC_KEYWORDS)

    # If pure review signal → review
    if review_hits >= 1 and spec_hits == 0:
        return "review"

    # If pure spec signal → spec
    if spec_hits >= 1 and review_hits == 0:
        return "spec"

    # Both hit → run both (ambiguous: "is the camera good?" could mean
    # spec lookup ("the camera is 48 MP") OR review sentiment)
    if spec_hits >= 1 and review_hits >= 1:
        return "both"

    # Neither — default to spec (existing behaviour, safer)
    return "spec"


def explain(question: str) -> dict:
    """Return classification + the keywords that triggered it (for debug/UI)."""
    q_lower = (question or "").lower().strip()
    tokens = _tokenize(q_lower)
    review_matches = sorted(tokens & REVIEW_KEYWORDS)
    spec_matches = sorted(tokens & SPEC_KEYWORDS)

    phrase_matches = [
        p for p in [
            "should i buy", "worth buying", "worth it", "worth the",
            "is it good", "is it bad", "any good",
        ] if _has_phrase(q_lower, p)
    ]

    return {
        "intent": classify(question),
        "review_keywords": review_matches,
        "review_phrases": phrase_matches,
        "spec_keywords": spec_matches,
    }
