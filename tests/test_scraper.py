"""
Unit tests for flight_reviews.scraper.

All tests that involve HTTP calls use ``unittest.mock.patch`` to replace
``requests.get`` with a fake response — this means the test suite runs
offline, deterministically, and in under a second.

Coverage targets
----------------
* _build_page_url produces correctly formatted URLs.
* _parse_reviews_from_page extracts text from well-formed HTML.
* _parse_reviews_from_page returns [] on HTML with no review divs.
* scrape_reviews stops early when a page returns no reviews.
* scrape_reviews re-raises HTTPError on first-page failure.
* scrape_reviews handles mid-run connection errors gracefully.
"""

from unittest.mock import MagicMock, patch

import pytest

from flight_reviews.scraper import (
    _build_page_url,
    _parse_reviews_from_page,
    scrape_reviews,
)


# ---------------------------------------------------------------------------
# _build_page_url
# ---------------------------------------------------------------------------

class TestBuildPageUrl:
    def test_page_number_in_url(self):
        url = _build_page_url(3)
        assert "/page/3/" in url

    def test_pagesize_query_param_present(self):
        url = _build_page_url(1)
        assert "pagesize=" in url

    def test_base_domain_present(self):
        url = _build_page_url(1)
        assert "airlinequality.com" in url

    def test_returns_string(self):
        assert isinstance(_build_page_url(1), str)

    def test_different_pages_produce_different_urls(self):
        assert _build_page_url(1) != _build_page_url(2)


# ---------------------------------------------------------------------------
# _parse_reviews_from_page
# ---------------------------------------------------------------------------

_VALID_HTML = b"""
<html><body>
  <div class="text_content">Great service on this flight.</div>
  <div class="text_content">Lost my luggage, very unhappy.</div>
  <div class="other_class">This div should be ignored.</div>
</body></html>
"""

_EMPTY_HTML = b"<html><body><p>No reviews here.</p></body></html>"


class TestParseReviewsFromPage:
    def test_extracts_all_review_divs(self):
        reviews = _parse_reviews_from_page(_VALID_HTML)
        assert len(reviews) == 2

    def test_returns_list_of_strings(self):
        reviews = _parse_reviews_from_page(_VALID_HTML)
        assert all(isinstance(r, str) for r in reviews)

    def test_captures_correct_text(self):
        reviews = _parse_reviews_from_page(_VALID_HTML)
        assert any("Great service" in r for r in reviews)

    def test_ignores_non_review_divs(self):
        reviews = _parse_reviews_from_page(_VALID_HTML)
        assert not any("This div should be ignored" in r for r in reviews)

    def test_empty_page_returns_empty_list(self):
        assert _parse_reviews_from_page(_EMPTY_HTML) == []


# ---------------------------------------------------------------------------
# scrape_reviews (mocked HTTP)
# ---------------------------------------------------------------------------

def _make_mock_response(html: bytes, status_code: int = 200) -> MagicMock:
    """Return a MagicMock that mimics a successful requests.Response."""
    mock = MagicMock()
    mock.content = html
    mock.status_code = status_code
    mock.raise_for_status = MagicMock()
    return mock


class TestScrapeReviews:
    @patch("flight_reviews.scraper.requests.get")
    def test_returns_list(self, mock_get):
        mock_get.return_value = _make_mock_response(_VALID_HTML)
        result = scrape_reviews(max_pages=1)
        assert isinstance(result, list)

    @patch("flight_reviews.scraper.requests.get")
    def test_collects_reviews_from_one_page(self, mock_get):
        mock_get.return_value = _make_mock_response(_VALID_HTML)
        result = scrape_reviews(max_pages=1)
        assert len(result) == 2

    @patch("flight_reviews.scraper.requests.get")
    def test_stops_early_on_empty_page(self, mock_get):
        """After a page with reviews, an empty page triggers early stop."""
        mock_get.side_effect = [
            _make_mock_response(_VALID_HTML),
            _make_mock_response(_EMPTY_HTML),
        ]
        result = scrape_reviews(max_pages=10)
        assert len(result) == 2
        assert mock_get.call_count == 2

    @patch("flight_reviews.scraper.requests.get")
    def test_reraises_http_error_on_first_page(self, mock_get):
        """A non-2xx on page 1 should propagate — it is not recoverable."""
        import requests as req
        error_response = _make_mock_response(b"", status_code=403)
        error_response.raise_for_status.side_effect = req.HTTPError("403 Forbidden")
        mock_get.return_value = error_response
        with pytest.raises(req.HTTPError):
            scrape_reviews(max_pages=1)

    @patch("flight_reviews.scraper.requests.get")
    def test_connection_error_mid_run_stops_gracefully(self, mock_get):
        """A connection error on page 2 should stop the loop, not crash."""
        import requests as req
        mock_get.side_effect = [
            _make_mock_response(_VALID_HTML),
            req.ConnectionError("Network unreachable"),
        ]
        result = scrape_reviews(max_pages=5)
        assert len(result) == 2
