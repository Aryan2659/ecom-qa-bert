"""
ScraperAPI-based scraper (primary path when SCRAPERAPI_KEY is set).

Uses ScraperAPI's residential proxy pool to bypass anti-bot systems.
Free tier: 1,000 requests/month at scraperapi.com.

When enabled, this runs BEFORE Playwright in the scraper chain — most
requests to Amazon/Flipkart should succeed here without ever spinning
up the browser.
"""
import logging
import re
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from . import config
from .scraper_legacy import _amazon, _flipkart, _generic, _clean

logger = logging.getLogger(__name__)

SCRAPERAPI_ENDPOINT = "https://api.scraperapi.com"


def scrape_with_scraperapi(url: str) -> dict:
    """
    Fetch `url` through ScraperAPI's proxy, then reuse our existing
    HTML parsers. Returns the same dict shape as the other scrapers.

    Returns `{"error": "..."}` on failure so the caller can fall back
    to Playwright.
    """
    if not config.SCRAPERAPI_KEY:
        return {"error": "ScraperAPI key not configured."}

    params = {
        "api_key": config.SCRAPERAPI_KEY,
        "url": url,
        "country_code": "in",       # route through Indian IPs for .in stores
        "render": "true" if config.SCRAPERAPI_RENDER_JS else "false",
        "keep_headers": "false",
    }

    request_url = f"{SCRAPERAPI_ENDPOINT}?{urlencode(params)}"

    try:
        logger.info(f"Fetching via ScraperAPI: {url}")
        resp = requests.get(request_url, timeout=config.SCRAPERAPI_TIMEOUT)
    except requests.exceptions.Timeout:
        return {"error": f"ScraperAPI timed out after "
                         f"{config.SCRAPERAPI_TIMEOUT}s"}
    except requests.exceptions.RequestException as e:
        logger.exception("ScraperAPI request failed")
        return {"error": f"ScraperAPI network error: {e}"}

    if resp.status_code == 401:
        return {"error": "ScraperAPI rejected the key (401). "
                         "Check SCRAPERAPI_KEY is correct."}
    if resp.status_code == 403:
        return {"error": "ScraperAPI forbade the request (403). "
                         "You may be out of credits or blocked."}
    if resp.status_code == 429:
        return {"error": "ScraperAPI rate-limited (429). "
                         "Wait a moment and retry."}
    if resp.status_code == 500:
        return {"error": "Target site returned 500 through ScraperAPI."}
    if resp.status_code >= 400:
        return {"error": f"ScraperAPI returned HTTP {resp.status_code}."}

    html = resp.text or ""
    if not html.strip():
        return {"error": "ScraperAPI returned empty body."}

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    url_lower = url.lower()
    if "amazon." in url_lower:
        data = _amazon(soup)
        data["source"] = "amazon"
        data = _amazon_extra(soup, data)
    elif "flipkart." in url_lower:
        data = _flipkart(soup)
        data["source"] = "flipkart"
        data = _flipkart_extra(soup, data)
    else:
        data = _generic(soup)
        data["source"] = "generic"
        data["reviews"] = []
        data["rating_text"] = ""

    parts = []
    if data.get("title"):
        parts.append(f"Product: {data['title']}.")
    if data.get("features"):
        parts.append(f"Features: {data['features']}")
    if data.get("description"):
        parts.append(f"Description: {data['description']}")
    if data.get("specs"):
        parts.append(f"Specifications: {data['specs']}")
    if data.get("rating_text"):
        parts.append(f"Rating: {data['rating_text']}")

    context = _clean(" ".join(parts), limit=20000)
    data["context"] = context
    data["char_count"] = len(context)
    data["reviews"] = data.get("reviews", [])
    data["review_count"] = len(data["reviews"])

    if len(context) < 50 and not data["reviews"]:
        # ScraperAPI returned HTML but it was empty/blocked content
        return {"error": "ScraperAPI returned a page with no usable product text. "
                         "The target site may have shown a CAPTCHA or blank page."}

    logger.info(
        f"ScraperAPI scraped [{data['source']}] "
        f"title={data.get('title', '?')[:60]!r} "
        f"chars={len(context)} reviews={data['review_count']}"
    )
    return data


# ── Extra extractors for ratings + reviews (same HTML parsers) ─────

def _amazon_extra(soup: BeautifulSoup, data: dict) -> dict:
    """Pull reviews and rating out of a pre-parsed Amazon soup."""
    rating_text = ""
    rating_el = soup.select_one("span[data-hook='rating-out-of-text'], "
                                "#acrPopover .a-icon-alt")
    if rating_el:
        rating_text = _clean(rating_el.get_text(" ", strip=True), limit=100)
    count_el = soup.select_one("#acrCustomerReviewText")
    if count_el:
        rating_text = (rating_text + " — " +
                       _clean(count_el.get_text(" ", strip=True), limit=50)).strip(" —")
    data["rating_text"] = rating_text

    reviews = []
    seen = set()
    for rb in soup.select("div[data-hook='review'], "
                          "#cm-cr-dp-review-list div.review, "
                          "#reviewsMedley div.review"):
        title_el = rb.select_one("a[data-hook='review-title'], "
                                 "span[data-hook='review-title']")
        body_el = rb.select_one("span[data-hook='review-body']")
        rating_el = rb.select_one("i[data-hook='review-star-rating'] span")

        title = _clean(title_el.get_text(" ", strip=True), limit=200) if title_el else ""
        body = _clean(body_el.get_text(" ", strip=True), limit=1500) if body_el else ""
        rating = None
        if rating_el:
            m = re.search(r"([0-9.]+)", rating_el.get_text(" ", strip=True))
            if m:
                try:
                    rating = float(m.group(1))
                except ValueError:
                    pass

        key = (title[:80], body[:100])
        if body and key not in seen:
            seen.add(key)
            reviews.append({"title": title, "text": body, "rating": rating})
            if len(reviews) >= config.PLAYWRIGHT_MAX_REVIEWS:
                break
    data["reviews"] = reviews
    return data


def _flipkart_extra(soup: BeautifulSoup, data: dict) -> dict:
    rating_el = soup.select_one("div._2d4LTz, div.XQDdHH")
    rating_text = _clean(rating_el.get_text(" ", strip=True), limit=50) if rating_el else ""
    count_el = soup.select_one("span._2_R_DZ, span.Wphh3N")
    if count_el:
        rating_text = (rating_text + " · " +
                       _clean(count_el.get_text(" ", strip=True), limit=80)).strip(" ·")
    data["rating_text"] = rating_text

    reviews = []
    seen = set()
    for rb in soup.select("div._16PBlm, div.col.EPCmJX"):
        rating_el = rb.select_one("div._3LWZlK, div.XQDdHH")
        title_el = rb.select_one("p._2-N8zT, p.z9E0IG")
        body_el = rb.select_one("div.t-ZTKy div, div.ZmyHeo div")

        title = _clean(title_el.get_text(" ", strip=True), limit=200) if title_el else ""
        body = _clean(body_el.get_text(" ", strip=True), limit=1500) if body_el else ""
        rating = None
        if rating_el:
            try:
                rating = float(rating_el.get_text(" ", strip=True))
            except ValueError:
                pass

        key = (title[:80], body[:100])
        if body and key not in seen:
            seen.add(key)
            reviews.append({"title": title, "text": body, "rating": rating})
            if len(reviews) >= config.PLAYWRIGHT_MAX_REVIEWS:
                break
    data["reviews"] = reviews
    return data
