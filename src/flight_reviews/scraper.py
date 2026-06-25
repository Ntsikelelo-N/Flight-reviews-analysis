"""
Web scraper for British Airways customer reviews from Skytrax.

Design decisions
----------------
* Early-stop logic: the loop halts when a page returns no reviews,
  avoiding 4 000 HTTP requests against a site that has ~36 pages.
* Rate limiting: a configurable delay between requests avoids hammering
  the server and reduces the risk of being IP-blocked.
* Response validation: non-2xx HTTP codes raise immediately rather than
  silently producing empty data.
* Pure functions: ``_build_page_url`` and ``_parse_reviews_from_page`` are
  separated from the stateful loop so they can be tested independently.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from flight_reviews.config import (
    BASE_URL,
    DATA_RAW_DIR,
    MAX_PAGES,
    PAGE_SIZE,
    REQUEST_DELAY_SECS,
    REQUEST_TIMEOUT_SECS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_page_url(page: int) -> str:
    """Construct the paginated Skytrax URL for a given page number."""
    return (
        f"{BASE_URL}/page/{page}/"
        f"?sortby=post_date%3ADesc&pagesize={PAGE_SIZE}"
    )


def _parse_reviews_from_page(html: bytes) -> list[str]:
    """
    Extract all review text blocks from a single Skytrax HTML page.

    Returns an empty list when no review divs are found.  The caller
    uses this as a signal to stop pagination rather than iterating to
    a fixed upper bound that may not reflect actual site content.
    """
    soup = BeautifulSoup(html, "html.parser")
    review_divs = soup.find_all("div", {"class": "text_content"})
    return [div.get_text() for div in review_divs]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_reviews(max_pages: int = MAX_PAGES) -> list[str]:
    """
    Scrape British Airways reviews from Skytrax across multiple pages.

    The function stops early when a page returns no reviews (i.e., the
    end of available content has been reached).

    Parameters
    ----------
    max_pages : int
        Hard upper limit on the number of pages to request.

    Returns
    -------
    list[str]
        Raw review strings, each potentially containing the
        ``VERIFIED_TAG`` prefix inserted by Skytrax.

    Raises
    ------
    requests.HTTPError
        If the first page returns a non-2xx response (subsequent page
        failures log a warning and stop the loop gracefully).
    """
    all_reviews: list[str] = []

    for page in range(1, max_pages + 1):
        url = _build_page_url(page)
        logger.info("Scraping page %d of %d — %s", page, max_pages, url)

        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECS)
            response.raise_for_status()
        except requests.HTTPError as exc:
            if page == 1:
                raise  # First page failure is unrecoverable
            logger.warning("HTTP error on page %d: %s — stopping.", page, exc)
            break
        except requests.RequestException as exc:
            logger.warning("Request failed on page %d: %s — stopping.", page, exc)
            break

        page_reviews = _parse_reviews_from_page(response.content)

        if not page_reviews:
            logger.info("No reviews on page %d — end of content reached.", page)
            break

        all_reviews.extend(page_reviews)
        logger.debug(
            "Page %d: %d reviews (running total: %d)",
            page, len(page_reviews), len(all_reviews),
        )
        time.sleep(REQUEST_DELAY_SECS)

    logger.info("Scraping complete. Total reviews collected: %d", len(all_reviews))
    return all_reviews


def save_raw_reviews(
    reviews: list[str],
    filename: str = "BA_reviews_raw.csv",
) -> Path:
    """
    Persist raw reviews to CSV in the ``data/raw/`` directory.

    Parameters
    ----------
    reviews : list[str]
        Raw review strings as returned by :func:`scrape_reviews`.
    filename : str
        Output filename within ``data/raw/``.

    Returns
    -------
    Path
        Absolute path to the saved CSV file.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_RAW_DIR / filename
    pd.DataFrame({"reviews": reviews}).to_csv(output_path, index=False)
    logger.info("Raw reviews saved → %s", output_path)
    return output_path
