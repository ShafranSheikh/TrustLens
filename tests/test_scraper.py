"""Tests for the parts of the live scraper that need no browser.

The scraping itself cannot be tested without a network and a Chrome install, so
what is covered here is the logic that has historically gone wrong silently:
address validation, date parsing, and the error raised when Selenium is absent.
"""
import datetime as _dt

import pytest

from trustlens import review_scraper as rs


@pytest.mark.parametrize('url, ok', [
    ('https://example.com/products/thing-i123.html', True),
    ('http://example.com/a', True),
    ('example.com/products/thing.html', False),   # no scheme
    ('not-a-url', False),
    ('https://example.com', False),               # no path: a home page, not a product
    ('  https://example.com/x  ', True),          # surrounding space is tolerated
    ('ftp://example.com/x', False),
])
def test_looks_like_product_url(url, ok):
    assert rs.looks_like_product_url(url) is ok


TODAY = _dt.date(2026, 9, 26)


@pytest.mark.parametrize('raw, want', [
    ('2 weeks ago',  '2026-09-12'),
    ('3 days ago',   '2026-09-23'),
    ('1 month ago',  '2026-08-27'),
    ('08 Feb 2025',  '2025-02-08'),
    ('8 Feb 2025',   '2025-02-08'),
])
def test_parse_review_date(raw, want):
    assert rs.parse_review_date(raw, today=TODAY) == want


def test_unparseable_date_is_kept_verbatim():
    # Better a value the analyst can see and fix than a silent None.
    assert rs.parse_review_date('garbled', today=TODAY) == 'garbled'


@pytest.mark.parametrize('blank', ['', '   ', None])
def test_blank_date_is_none_not_empty_string(blank):
    assert rs.parse_review_date(blank, today=TODAY) is None


def test_missing_selenium_gives_an_actionable_error():
    pytest.importorskip  # keep the import list honest
    try:
        import selenium  # noqa: F401
    except ImportError:
        with pytest.raises(rs.ScrapeError, match='pip install selenium'):
            rs.make_driver()
    else:
        pytest.skip('Selenium is installed here, so the absent-Selenium path cannot run')


def test_a_bad_address_is_rejected_before_any_browser_work():
    """Must hold whether or not Selenium is installed - no Chrome should start."""
    with pytest.raises(rs.ScrapeError, match='does not look like a product'):
        rs.scrape_product('not-a-url')


def test_result_helpers():
    r = rs.ScrapeResult(url='u', product_name='p')
    r.reviews = [{'review_text': 'good', 'star_rating': 5},
                 {'review_text': 'bad',  'star_rating': 1}]
    assert r.texts() == ['good', 'bad']
    assert r.stars() == [5, 1]
