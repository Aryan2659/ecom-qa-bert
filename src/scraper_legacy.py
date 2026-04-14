"""
Robust web scraper for product pages.

Improvements over the original:
  * Retries with exponential backoff (urllib3.Retry via HTTPAdapter)
  * Session reuse + connection pooling
  * Rotating User-Agent pool
  * Extra headers that make Amazon/Flipkart less likely to block
  * Multiple fallback selectors per marketplace
  * Structured error responses instead of raw exception strings
  * Defensive text length limits so pathological pages don't OOM the worker
"""
import logging
import random
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

logger = logging.getLogger(__name__)

# A small, modern UA pool. We rotate per request to reduce the chance of being
# served a simplified / bot-filtered page.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
    "Gecko/20100101 Firefox/124.0",
]

BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

MAX_HTML_BYTES = 4 * 1024 * 1024   # 4 MB safety limit
MAX_FIELD_CHARS = 8000             # per-field clamp

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Lazy-init a shared Session with retry policy attached."""
    global _session
    if _session is not None:
        return _session

    s = requests.Session()
    retry = Retry(
        total=config.SCRAPE_MAX_RETRIES,
        connect=config.SCRAPE_MAX_RETRIES,
        read=config.SCRAPE_MAX_RETRIES,
        backoff_factor=0.8,                         # 0.8s, 1.6s, 3.2s…
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    _session = s
    return s


def _build_headers() -> dict:
    headers = dict(BASE_HEADERS)
    headers["User-Agent"] = random.choice(USER_AGENTS)
    return headers


def _clean(text: str, limit: int = MAX_FIELD_CHARS) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def scrape_url(url: str) -> dict:
    """Fetch `url` and return a structured dict with product text ready for QA."""
    if not url:
        return {"error": "URL is required."}

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    session = _get_session()
    try:
        logger.info(f"Scraping: {url}")
        resp = session.get(
            url,
            headers=_build_headers(),
            timeout=config.SCRAPE_TIMEOUT,
            stream=True,
        )
        # Guard against huge pages that could exhaust memory
        content = resp.raw.read(MAX_HTML_BYTES + 1, decode_content=True)
        if len(content) > MAX_HTML_BYTES:
            return {"error": "Page is too large to process (>4 MB)."}
        resp._content = content

        if resp.status_code == 403:
            return {"error": "Site blocked the request (HTTP 403). "
                             "Try Text mode with the product description pasted manually."}
        if resp.status_code == 404:
            return {"error": "Page not found (HTTP 404). Check the URL."}
        if resp.status_code >= 400:
            return {"error": f"HTTP {resp.status_code}. "
                             "The site may rate-limit or block scrapers."}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to {url}. Check the URL."}
    except requests.exceptions.Timeout:
        return {"error": f"Request timed out after {config.SCRAPE_TIMEOUT}s."}
    except requests.exceptions.TooManyRedirects:
        return {"error": "Too many redirects — the URL may be broken."}
    except requests.exceptions.RequestException as e:
        logger.exception("Scrape failed")
        return {"error": f"Network error: {e}"}

    try:
        html = resp.content.decode(resp.encoding or "utf-8", errors="replace")
    except Exception:
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer",
                     "header", "form", "svg"]):
        tag.decompose()

    url_lower = url.lower()
    if "amazon." in url_lower:
        data = _amazon(soup)
        data["source"] = "amazon"
    elif "flipkart." in url_lower:
        data = _flipkart(soup)
        data["source"] = "flipkart"
    else:
        data = _generic(soup)
        data["source"] = "generic"

    # Build combined context for QA
    parts = []
    if data.get("title"):
        parts.append(f"Product: {data['title']}.")
    if data.get("features"):
        parts.append(f"Features: {data['features']}")
    if data.get("description"):
        parts.append(f"Description: {data['description']}")
    if data.get("specs"):
        parts.append(f"Specifications: {data['specs']}")

    context = _clean(" ".join(parts), limit=20000)
    data["context"] = context
    data["char_count"] = len(context)

    if len(context) < 50:
        data["warning"] = (
            "Very little text was extracted. The site probably blocks scrapers "
            "or renders content with JavaScript. Switch to Text mode and paste "
            "the product description manually."
        )

    logger.info(
        f"Scraped [{data['source']}] title={data.get('title', '?')[:60]!r} "
        f"chars={len(context)}"
    )
    return data


# ── Marketplace-specific extractors ──────────────────────────────────────

def _first_text(soup: BeautifulSoup, *selectors: str) -> str:
    """Return text of the first matching CSS selector, or ''."""
    for sel in selectors:
        tag = soup.select_one(sel)
        if tag:
            txt = tag.get_text(" ", strip=True)
            if txt:
                return txt
    return ""


def _amazon(soup: BeautifulSoup) -> dict:
    d = {"title": "", "features": "", "description": "", "specs": ""}

    d["title"] = _clean(_first_text(
        soup,
        "span#productTitle",
        "h1#title",
        "h1.a-size-large",
    ))

    feat = soup.select_one("div#feature-bullets, #featurebullets_feature_div")
    if feat:
        bullets = [
            li.get_text(" ", strip=True)
            for li in feat.select("li")
            if li.get_text(strip=True) and "hidden" not in (li.get("class") or [])
        ]
        d["features"] = _clean(" • ".join(bullets))

    desc = soup.select_one(
        "div#productDescription, "
        "#productDescription_feature_div, "
        "div[data-feature-name='productDescription']"
    )
    if desc:
        d["description"] = _clean(desc.get_text(" ", strip=True))
    else:
        aplus = soup.select_one("div#aplus, #aplus_feature_div")
        if aplus:
            chunks = [p.get_text(" ", strip=True) for p in aplus.select("p, li")]
            d["description"] = _clean(" ".join(c for c in chunks if c)[:MAX_FIELD_CHARS])

    specs = []
    spec_tables = soup.select(
        "table.prodDetTable, "
        "table.a-keyvalue, "
        "table#productDetails_techSpec_section_1, "
        "table#productDetails_detailBullets_sections1"
    )
    for table in spec_tables:
        for row in table.select("tr"):
            th = row.find(["th", "td"])
            cells = row.find_all("td")
            if th and cells:
                k = th.get_text(" ", strip=True)
                v = cells[-1].get_text(" ", strip=True)
                if k and v and k != v:
                    entry = f"{k}: {v}"
                    if entry not in specs:
                        specs.append(entry)

    # Detail bullets (left column on many Amazon pages)
    for li in soup.select("div#detailBullets_feature_div li, #detailBulletsWrapper_feature_div li"):
        text = li.get_text(" ", strip=True)
        if ":" in text:
            specs.append(text)

    d["specs"] = _clean(" | ".join(specs))
    return d


def _flipkart(soup: BeautifulSoup) -> dict:
    d = {"title": "", "features": "", "description": "", "specs": ""}

    d["title"] = _clean(_first_text(
        soup,
        "span.VU-ZEz",
        "span.B_NuCI",
        "h1 span",
        "h1",
    ))

    highlights = soup.select("div._2418kt li, ._21Ahn- li, li.col-12")
    if highlights:
        d["features"] = _clean(
            " • ".join(h.get_text(" ", strip=True) for h in highlights[:20])
        )

    desc = soup.select_one("div._1mXcCf, div._1AN87F, div.RmoJUa")
    if desc:
        d["description"] = _clean(desc.get_text(" ", strip=True))

    specs = []
    for table in soup.select("table._14cfVK, table._1s_Smc, table._0ZhAN9"):
        for row in table.select("tr"):
            cells = row.select("td")
            if len(cells) >= 2:
                k = cells[0].get_text(" ", strip=True)
                v = cells[-1].get_text(" ", strip=True)
                if k and v:
                    specs.append(f"{k}: {v}")
    d["specs"] = _clean(" | ".join(specs))
    return d


def _generic(soup: BeautifulSoup) -> dict:
    d = {"title": "", "features": "", "description": "", "specs": ""}

    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        d["title"] = _clean(og_title["content"])
    elif soup.find("h1"):
        d["title"] = _clean(soup.find("h1").get_text(" ", strip=True))
    elif soup.title:
        d["title"] = _clean(soup.title.get_text(strip=True))

    og_desc = soup.find("meta", attrs={"name": "description"}) or \
              soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"):
        d["description"] = _clean(og_desc["content"])

    seen, texts = set(), []
    for tag in soup.find_all(["p", "li", "td"]):
        t = tag.get_text(" ", strip=True)
        if t and len(t) > 30 and t not in seen:
            seen.add(t)
            texts.append(t)
            if len(texts) >= 30:
                break
    extra = " ".join(texts)
    # Append to description rather than replace so OG metadata survives
    if extra:
        d["description"] = _clean(f"{d['description']} {extra}".strip())
    return d
