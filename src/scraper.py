"""
Playwright-based scraper for e-commerce product pages.

Extracts product info AND reviews. Falls back to the legacy
requests-based scraper (scraper_legacy.py) if Playwright fails.

Amazon blocks headless browsers aggressively from cloud IPs —
expect ~50-70% success rate on HF Spaces. Flipkart is friendlier.
"""
import logging
import random
import re

from bs4 import BeautifulSoup

from . import config
from .scraper_legacy import scrape_url as legacy_scrape_url
from .scraper_legacy import _amazon, _flipkart, _generic, _clean

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]


def scrape_url(url: str) -> dict:
    """
    Scraping chain: ScraperAPI → Playwright → requests (legacy).
    First one that returns usable data wins.
    """
    if not url:
        return {"error": "URL is required."}

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # 1) ScraperAPI (residential proxy) — best chance of avoiding blocks
    if config.SCRAPERAPI_ENABLED:
        try:
            from .scraper_proxy import scrape_with_scraperapi
            result = scrape_with_scraperapi(url)
            if not result.get("error"):
                result["scraper_used"] = "scraperapi"
                return result
            logger.warning(f"ScraperAPI failed: {result['error']} — trying Playwright")
            last_error = result["error"]
        except Exception as e:
            logger.exception("ScraperAPI crashed")
            last_error = f"ScraperAPI crashed: {e}"
    else:
        last_error = None

    # 2) Playwright (headless browser)
    if config.PLAYWRIGHT_ENABLED:
        try:
            result = _scrape_with_playwright(url)
            if not result.get("error"):
                result["scraper_used"] = "playwright"
                return result
            logger.warning(f"Playwright failed: {result['error']} — trying legacy")
            last_error = result["error"]
        except Exception as e:
            logger.exception("Playwright crashed")
            last_error = f"Playwright crashed: {e}"

    # 3) Legacy requests (last resort)
    try:
        fallback = legacy_scrape_url(url)
        fallback["reviews"] = []
        fallback["scraper_used"] = "legacy"
        if last_error:
            fallback["upstream_error"] = last_error
        return fallback
    except Exception as e:
        logger.exception("Legacy scraper crashed")
        return {
            "error": f"All scrapers failed. Last error: {last_error or e}",
            "scraper_used": "none",
        }


def _scrape_with_playwright(url: str) -> dict:
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError
    except ImportError:
        return {"error": "Playwright not installed."}

    url_lower = url.lower()

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(
                headless=config.PLAYWRIGHT_HEADLESS,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
        except Exception as e:
            return {"error": f"Browser launch failed: {e}"}

        try:
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={"width": 1280, "height": 900},
                locale="en-US",
            )
            # Block heavy resources we don't need
            context.route(
                "**/*",
                lambda route: (
                    route.abort()
                    if route.request.resource_type in {"image", "media", "font"}
                    else route.continue_()
                ),
            )

            page = context.new_page()
            page.set_default_timeout(config.PLAYWRIGHT_TIMEOUT_MS)

            try:
                page.goto(url, wait_until="domcontentloaded",
                          timeout=config.PLAYWRIGHT_TIMEOUT_MS)
            except PWTimeoutError:
                return {"error": f"Page load timed out "
                                 f"({config.PLAYWRIGHT_TIMEOUT_MS//1000}s)"}

            # CAPTCHA detection (Amazon)
            head_html = page.content()[:5000].lower()
            if ("enter the characters you see below" in head_html
                    or "type the characters" in head_html
                    or "automated access" in head_html):
                return {"error": "Site served a CAPTCHA. Cloud IPs get "
                                 "blocked often — try Text mode."}

            # Let dynamic content render
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except PWTimeoutError:
                pass

            if "amazon." in url_lower:
                data = _amazon_playwright(page)
                data["source"] = "amazon"
            elif "flipkart." in url_lower:
                data = _flipkart_playwright(page)
                data["source"] = "flipkart"
            else:
                data = _generic_playwright(page)
                data["source"] = "generic"

        finally:
            try:
                context.close()
            except Exception:
                pass
            browser.close()

    # Build QA context from product info (not reviews — those go to sentiment)
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
        data["warning"] = (
            "Very little text was extracted. The site may have blocked the "
            "request or requires JS we couldn't render. Try Text mode."
        )

    logger.info(
        f"Scraped [{data['source']}] title={data.get('title', '?')[:60]!r} "
        f"chars={len(context)} reviews={data['review_count']}"
    )
    return data


def _amazon_playwright(page) -> dict:
    html = page.content()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    data = _amazon(soup)

    rating_text = ""
    rating_el = soup.select_one("span[data-hook='rating-out-of-text'], "
                                "#acrPopover .a-icon-alt")
    if rating_el:
        rating_text = _clean(rating_el.get_text(" ", strip=True), limit=100)

    reviews_count_el = soup.select_one("#acrCustomerReviewText")
    if reviews_count_el:
        rating_text = (rating_text + " — " +
                       _clean(reviews_count_el.get_text(" ", strip=True), limit=50)).strip(" —")
    data["rating_text"] = rating_text

    reviews = []
    seen = set()
    review_blocks = soup.select(
        "div[data-hook='review'], "
        "#cm-cr-dp-review-list div.review, "
        "#reviewsMedley div.review"
    )
    for rb in review_blocks:
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


def _flipkart_playwright(page) -> dict:
    # Auto-scroll so lazy loaders fire
    try:
        for _ in range(3):
            page.mouse.wheel(0, 1500)
            page.wait_for_timeout(400)
    except Exception:
        pass

    html = page.content()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    data = _flipkart(soup)

    rating_el = soup.select_one("div._2d4LTz, div.XQDdHH")
    rating_text = _clean(rating_el.get_text(" ", strip=True), limit=50) if rating_el else ""
    count_el = soup.select_one("span._2_R_DZ, span.Wphh3N")
    if count_el:
        rating_text = (rating_text + " · " +
                       _clean(count_el.get_text(" ", strip=True), limit=80)).strip(" ·")
    data["rating_text"] = rating_text

    reviews = []
    seen = set()
    review_blocks = soup.select("div._16PBlm, div.col.EPCmJX")
    for rb in review_blocks:
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


def _generic_playwright(page) -> dict:
    html = page.content()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    data = _generic(soup)
    data["reviews"] = []
    data["rating_text"] = ""
    return data
