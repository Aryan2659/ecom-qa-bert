"""
Tests for the scraper module.

These run offline by mocking `requests.Session.get` — we test parsing
logic against fixture HTML, never hit Amazon/Flipkart in CI.
"""
from unittest.mock import patch, MagicMock

import pytest

from src.scraper_legacy import scrape_url, _amazon, _flipkart, _generic
from bs4 import BeautifulSoup


AMAZON_HTML = """
<html><body>
  <span id="productTitle">  Samsung Galaxy S24 Ultra 5G  </span>
  <div id="feature-bullets">
    <ul>
      <li><span>6.8-inch QHD+ AMOLED display</span></li>
      <li><span>200 MP main camera</span></li>
      <li class="hidden">hidden bullet should be ignored</li>
    </ul>
  </div>
  <div id="productDescription"><p>A flagship Android phone with titanium build.</p></div>
  <table id="productDetails_techSpec_section_1">
    <tr><th>RAM</th><td>12 GB</td></tr>
    <tr><th>Battery</th><td>5000 mAh</td></tr>
  </table>
</body></html>
"""

FLIPKART_HTML = """
<html><body>
  <h1><span class="VU-ZEz">OnePlus 12R 5G (Cool Blue, 256 GB)</span></h1>
  <ul class="_21Ahn-">
    <li class="col-12">Snapdragon 8 Gen 2</li>
    <li class="col-12">5500 mAh battery</li>
  </ul>
</body></html>
"""

GENERIC_HTML = """
<html><head>
  <title>Widget Pro</title>
  <meta property="og:title" content="Widget Pro — The Best Widget">
  <meta name="description" content="Widget Pro is a premium widget with advanced features.">
</head><body>
  <h1>Widget Pro</h1>
  <p>This paragraph has enough length to be captured by the generic extractor.</p>
  <p>Another long paragraph describing how the widget works in practice.</p>
</body></html>
"""


def _mock_response(html: str, status: int = 200):
    """Build a mock response that behaves like requests.Response enough for scrape_url."""
    resp = MagicMock()
    resp.status_code = status
    resp.encoding = "utf-8"
    resp.text = html
    resp.content = html.encode("utf-8")
    resp.raw.read.return_value = html.encode("utf-8")
    return resp


def test_amazon_parser_extracts_title_features_specs():
    soup = BeautifulSoup(AMAZON_HTML, "html.parser")
    data = _amazon(soup)
    assert "Samsung Galaxy S24 Ultra" in data["title"]
    assert "6.8-inch" in data["features"]
    assert "200 MP" in data["features"]
    assert "hidden bullet" not in data["features"]
    assert "titanium" in data["description"]
    assert "RAM: 12 GB" in data["specs"]
    assert "Battery: 5000 mAh" in data["specs"]


def test_flipkart_parser_extracts_title_and_highlights():
    soup = BeautifulSoup(FLIPKART_HTML, "html.parser")
    data = _flipkart(soup)
    assert "OnePlus 12R" in data["title"]
    assert "Snapdragon" in data["features"]
    assert "5500 mAh" in data["features"]


def test_generic_parser_uses_og_metadata():
    soup = BeautifulSoup(GENERIC_HTML, "html.parser")
    data = _generic(soup)
    assert "Widget Pro" in data["title"]
    assert "premium widget" in data["description"]


def test_scrape_url_amazon_integration():
    with patch("src.scraper_legacy._get_session") as mock_session:
        sess = MagicMock()
        sess.get.return_value = _mock_response(AMAZON_HTML)
        mock_session.return_value = sess

        result = scrape_url("https://www.amazon.in/dp/B0TEST")
        assert result["source"] == "amazon"
        assert "Samsung" in result["title"]
        assert result["char_count"] > 50
        assert "error" not in result
        assert "Product:" in result["context"]
        assert "Features:" in result["context"]


def test_scrape_url_handles_http_error():
    with patch("src.scraper_legacy._get_session") as mock_session:
        sess = MagicMock()
        sess.get.return_value = _mock_response("", status=403)
        mock_session.return_value = sess

        result = scrape_url("https://example.com/blocked")
        assert "error" in result
        assert "403" in result["error"] or "blocked" in result["error"].lower()


def test_scrape_url_rejects_empty_input():
    assert "error" in scrape_url("")
    assert "error" in scrape_url("   ")


def test_scrape_url_adds_https_scheme():
    with patch("src.scraper_legacy._get_session") as mock_session:
        sess = MagicMock()
        sess.get.return_value = _mock_response(GENERIC_HTML)
        mock_session.return_value = sess

        scrape_url("example.com/product")
        called_url = sess.get.call_args[0][0]
        assert called_url.startswith("https://")


def test_scrape_url_returns_warning_on_sparse_content():
    with patch("src.scraper_legacy._get_session") as mock_session:
        sess = MagicMock()
        sess.get.return_value = _mock_response("<html><body></body></html>")
        mock_session.return_value = sess

        result = scrape_url("https://example.com/empty")
        # Either an error or a warning is acceptable for an empty page
        assert result.get("warning") or result.get("error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
