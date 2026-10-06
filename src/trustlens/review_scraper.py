"""Live review collection from a single product page.

The corpus for this study came from one platform, and the extraction written to
collect it reads that platform's markup directly. A demonstration has to cope
with more than one platform, so collection here is layered - most reliable
first, and the result records which layer produced the rows:

  1. A site profile. Hand-written selectors for a platform we know, driving real
     pagination so every review page is read rather than only the first.

  2. Structured data. schema.org `Product.review`, as JSON-LD or microdata. This
     is what a search engine reads to show review stars beside a shop's result,
     so a shop that wants those stars publishes it whatever it is built with.
     No selectors and no pagination - the whole set arrives in one object.

  3. A generic reader. Looks for repeated blocks pairing a star widget with a
     paragraph of text. Least reliable, and the result says so, but it is the
     difference between reading an unknown shop and refusing it.

Layer 2 is the reason this works on sites nobody wrote code for. Layer 1 exists
because the platform the corpus came from publishes no structured review data,
which is common for sites that render reviews from JavaScript after load.

Two fixes from the Phase 1 collection scripts are carried over deliberately.
Both were found the hard way and both silently produce wrong data if reverted:

  Star rating - counted by asking the browser for each star path's computed fill
      and testing for the gold colour. The alternative, reading the percentage out
      of the SVG mask attribute, returns 100% for every star whatever the true
      rating, so every review comes back as five stars.

  Pagination - the next-page button is matched with a concat() XPath on the whole
      class token. `contains(@class,'next')` also matches the PREVIOUS arrow,
      whose class is `next-pagination-item` and which is disabled on page one, so
      the scraper stops after a single page every time.

Reviewer names are never read, on any layer, per the study's ethical clearance.

Selenium is imported lazily so the rest of the package works without it.

    from trustlens import review_scraper as rs
    result = rs.scrape_product('https://…/product-page.html')
    print(result.source, len(result.reviews))

To find out what a candidate URL will do before relying on it in a demonstration:

    python -m trustlens.review_scraper https://…/product-page.html
"""
from __future__ import annotations
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse

# Safety stops, not targets. The caller asks for every review; these exist so a
# pagination widget that never reports "last page" cannot spin for ever.
MAX_PAGES = 200
MAX_REVIEWS = 3000
TIME_BUDGET_S = 300

GOLD = ('255', '200')          # rgb(255, 200, 60) - a filled star


class ScrapeError(RuntimeError):
    """Raised with a message meant to be shown to a person, not a stack trace."""


@dataclass(frozen=True)
class Profile:
    """Selectors for one platform whose markup we have read."""
    name: str
    domains: tuple[str, ...]
    container: str
    item: str
    content: str
    star: str
    title: str
    pagination: str
    next_button: str


PROFILES: tuple[Profile, ...] = (
    Profile(
        name='marketplace-a',          # the platform the corpus was collected from
        domains=('daraz.lk', 'daraz.com', 'daraz.pk', 'daraz.com.bd', 'daraz.com.np'),
        container="//*[@class='mod-reviews']",
        item=".//div[@class='item']",
        content=".//*[@class='content']",
        star=".//div[@class='i-rate-star i-rate-star-readonly']",
        title="//h1[contains(@class,'pdp-mod-product-badge-title')]",
        pagination="//*[contains(@class,'review-pagination')]",
        next_button=".//button[contains(concat(' ', @class, ' '),"
                    " ' next-pagination-item next ')]",
    ),
)

# Kept for backward compatibility with code and tests written against the
# single-platform version.
SEL = {
    'reviews_container': PROFILES[0].container,
    'review_item':       PROFILES[0].item,
    'review_content':    PROFILES[0].content,
    'star':              PROFILES[0].star,
    'product_title':     PROFILES[0].title,
    'pagination':        PROFILES[0].pagination,
    'next_button':       PROFILES[0].next_button,
}


def profile_for(url: str) -> Profile | None:
    """The site profile matching this address, or None for an unknown site."""
    host = (urlparse(url).hostname or '').lower().lstrip('.')
    for p in PROFILES:
        for d in p.domains:
            if host == d or host.endswith('.' + d):
                return p
    return None


@dataclass
class ScrapeResult:
    url: str
    product_name: str = ''
    reviews: list[dict] = field(default_factory=list)
    pages_read: int = 0
    stopped_early: bool = False
    source: str = ''            # which layer produced the rows
    platform: str = ''          # the host, for display

    def texts(self) -> list[str]:
        return [r.get('review_text', '') for r in self.reviews]

    def stars(self) -> list[int]:
        return [r.get('star_rating', 0) for r in self.reviews]


# --------------------------------------------------------------------- helpers

def looks_like_product_url(url: str) -> bool:
    try:
        p = urlparse(url.strip())
    except Exception:
        return False
    return p.scheme in ('http', 'https') and bool(p.netloc) and len(p.path) > 1


def parse_review_date(raw, today=None):
    """Relative dates ("2 weeks ago") to ISO.

    Text that cannot be parsed is returned unchanged rather than dropped, so an
    unrecognised date format shows up in the output where it can be seen and fixed
    instead of vanishing into a column of Nones. Blank input is None either way.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    today = today or datetime.now()
    low = s.lower()
    if low == 'today':
        return today.strftime('%Y-%m-%d')
    if low == 'yesterday':
        return (today - timedelta(days=1)).strftime('%Y-%m-%d')
    m = re.match(r'(\d+)\s*(day|week|month|year)s?\s*ago', low)
    if m:
        qty, unit = int(m.group(1)), m.group(2)
        delta = {'day': timedelta(days=qty), 'week': timedelta(weeks=qty),
                 'month': timedelta(days=30*qty), 'year': timedelta(days=365*qty)}[unit]
        return (today - delta).strftime('%Y-%m-%d')
    for fmt in ('%d %b %Y', '%d-%b-%y', '%d-%b-%Y', '%d/%m/%Y', '%Y-%m-%d',
                '%Y-%m-%dT%H:%M:%S', '%B %d, %Y'):
        try:
            return datetime.strptime(s[:len(fmt)+6], fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return s


def _need_selenium():
    """Import Selenium, turning the ImportError into something a user can act on.

    Selenium is imported here rather than at the top of the module on purpose: the
    rest of the pipeline - and the app's offline modes - must keep working on a
    machine that has never installed it.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError as exc:
        raise ScrapeError(
            'Live scraping needs Selenium. Install it with:  '
            'pip install selenium webdriver-manager') from exc
    return webdriver, Options, Service, By, ChromeDriverManager


def make_driver(headless: bool = True):
    webdriver, Options, Service, _By, ChromeDriverManager = _need_selenium()
    opts = Options()
    if headless:
        opts.add_argument('--headless=new')
    opts.add_argument('--window-size=1400,1000')
    opts.add_argument('--lang=en-US')
    opts.add_argument('--disable-blink-features=AutomationControlled')
    opts.add_experimental_option('excludeSwitches', ['enable-automation'])
    try:
        return webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                                options=opts)
    except Exception as exc:
        raise ScrapeError(f'Could not start Chrome: {exc}') from exc


# ------------------------------------------------- layer 2: structured data
# Read in one call in the page rather than through dozens of Selenium round
# trips. Author names are present in this data and are deliberately not read.
JS_STRUCTURED = r"""
return (() => {
  const out = {product: null, reviews: [], how: null};
  const push = (text, rating, date) => {
    text = (text == null ? '' : String(text)).trim();
    if (!text) return;
    let r = null;
    if (rating !== null && rating !== undefined && rating !== '') {
      const n = parseFloat(rating);
      if (!isNaN(n)) r = Math.max(1, Math.min(5, Math.round(n)));
    }
    out.reviews.push({review_text: text, star_rating: r, date: date || null});
  };
  document.querySelectorAll('script[type="application/ld+json"]').forEach(s => {
    let parsed; try { parsed = JSON.parse(s.textContent); } catch (e) { return; }
    const nodes = [];
    const walk = (n) => {
      if (!n || typeof n !== 'object') return;
      if (Array.isArray(n)) { n.forEach(walk); return; }
      nodes.push(n);
      if (n['@graph']) walk(n['@graph']);
    };
    walk(parsed);
    nodes.forEach(n => {
      const t = n['@type'];
      const types = Array.isArray(t) ? t : [t];
      if (types.indexOf('Product') !== -1 && !out.product) out.product = n.name || null;
      const revs = n.review || n.reviews;
      if (!revs) return;
      (Array.isArray(revs) ? revs : [revs]).forEach(rv => {
        if (!rv || typeof rv !== 'object') return;
        const rr = rv.reviewRating;
        const val = (rr && typeof rr === 'object') ? rr.ratingValue : (rr || null);
        push(rv.reviewBody || rv.description, val, rv.datePublished);
      });
    });
  });
  if (out.reviews.length) { out.how = 'json-ld'; return out; }
  document.querySelectorAll('[itemtype*="Review" i]').forEach(el => {
    const q = (sel) => el.querySelector(sel);
    const body = q('[itemprop="reviewBody"]') || q('[itemprop="description"]');
    const rate = q('[itemprop="ratingValue"]');
    const date = q('[itemprop="datePublished"]');
    push(body ? body.textContent : '',
         rate ? (rate.getAttribute('content') || rate.textContent) : null,
         date ? (date.getAttribute('datetime') || date.getAttribute('content')
                 || date.textContent) : null);
  });
  if (out.reviews.length) out.how = 'microdata';
  return out;
})();
"""

# --------------------------------------------- getting the reviews on screen
# A very common reason a product page returns nothing: the reviews exist but sit
# behind a "Reviews" tab, or below a "Load more" button, so they are not in the
# DOM until something is clicked. Both are handled before the readers run.
#
# Only links, buttons and tab roles are clicked, and only when their whole label
# is a review-ish word, so this cannot wander into "Delete" or "Buy now". Links
# that would navigate to a different page are skipped - an anchor on the same
# page is fine.
JS_OPEN_REVIEWS = r"""
return (() => {
  const want = /^(reviews?|ratings?|customer reviews?|reviews?\s*\(\d+\)|\d+\s*reviews?|ratings?\s*(&|and)\s*reviews?|reviews?\s*&\s*ratings?)$/i;
  const clicked = [];
  const here = location.pathname;
  const nodes = document.querySelectorAll('a,button,[role="tab"],[role="button"]');
  for (const el of nodes) {
    if (clicked.length >= 3) break;
    const t = (el.textContent || '').trim();
    if (!t || t.length > 40 || !want.test(t)) continue;
    const href = el.getAttribute('href');
    if (href && !href.startsWith('#') && !href.startsWith('?')) {
      try { if (new URL(href, location.href).pathname !== here) continue; }
      catch (e) { continue; }
    }
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    try { el.scrollIntoView({block: 'center'}); el.click(); clicked.push(t); }
    catch (e) { /* an unclickable match is not a failure */ }
  }
  return clicked;
})();
"""

JS_LOAD_MORE = r"""
return (() => {
  const want = /^(load|show|see|view)\s+(more|all)(\s+reviews?)?$|^more\s+reviews?$|^show\s+\d+\s+more$/i;
  for (const el of document.querySelectorAll('a,button,[role="button"]')) {
    const t = (el.textContent || '').trim();
    if (!t || t.length > 40 || !want.test(t)) continue;
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') continue;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    try { el.scrollIntoView({block: 'center'}); el.click(); return t; }
    catch (e) { return null; }
  }
  return null;
})();
"""


# --------------------------------------------------- layer 3: generic reader
# Finds the largest group of sibling elements that each pair a star reading with
# a paragraph of text. Deliberately conservative: it would rather find nothing
# than return a page's navigation as reviews.
#
# A caveat this layer cannot fix: on an unknown site the reviewer's display name
# may sit inside the same block as the review text, and there is no reliable way
# to tell them apart without knowing the markup. Live-fetched reviews are held in
# memory for the length of one request and never written to disk, so nothing is
# collected; but do not use this layer to build a dataset.
JS_GENERIC = r"""
return (() => {
  const starsOf = (el) => {
    const attrs = [el.getAttribute('aria-label'), el.getAttribute('title')];
    el.querySelectorAll('[aria-label],[title]').forEach(s => {
      attrs.push(s.getAttribute('aria-label'), s.getAttribute('title'));
    });
    for (const a of attrs) {
      if (!a) continue;
      let m = a.match(/([\d.]+)\s*(?:out of|\/|of)\s*5/i);
      if (m) return Math.max(1, Math.min(5, Math.round(parseFloat(m[1]))));
      m = a.match(/^\s*([1-5](?:\.\d)?)\s*stars?\s*$/i);
      if (m) return Math.round(parseFloat(m[1]));
    }
    const glyphs = ((el.textContent || '').match(/[★✭⭐]/g) || []).length;
    if (glyphs >= 1 && glyphs <= 5) return glyphs;
    const bar = el.querySelector('[class*="star" i] [style*="width"],'
                               + '[class*="rating" i] [style*="width"]');
    if (bar) {
      const m = (bar.getAttribute('style') || '').match(/width:\s*([\d.]+)%/);
      if (m) {
        const v = Math.round(parseFloat(m[1]) / 20);
        if (v >= 1 && v <= 5) return v;
      }
    }
    const full = el.querySelectorAll('[class*="star" i][class*="full" i],'
                                   + '[class*="star" i][class*="active" i],'
                                   + '[class*="star" i][class*="checked" i]').length;
    if (full >= 1 && full <= 5) return full;
    return null;
  };
  const groups = [];
  document.querySelectorAll('*').forEach(parent => {
    const kids = parent.children;
    if (kids.length < 2 || kids.length > 300) return;
    const rows = [];
    for (const k of kids) {
      const t = (k.innerText || '').trim();
      if (t.length < 15 || t.length > 3000) continue;
      const s = starsOf(k);
      if (s === null) continue;
      rows.push({stars: s, text: t});
    }
    if (rows.length >= 2) groups.push(rows);
  });
  if (!groups.length) return {reviews: [], how: 'generic'};
  groups.sort((a, b) => b.length - a.length);
  const out = {reviews: [], how: 'generic'};
  groups[0].forEach(r => {
    const lines = r.text.split('\n').map(x => x.trim()).filter(Boolean)
      .filter(x => !/^[★✭⭐☆\s]+$/.test(x))
      .filter(x => !/^[\d.]+\s*(out of|\/)\s*5/i.test(x))
      .filter(x => !/^(helpful|report|reply|verified purchase|translate)\b/i.test(x));
    const body = lines.filter(x => x.length > 15).join(' ').slice(0, 1200);
    if (body) out.reviews.push({review_text: body, star_rating: r.stars, date: null});
  });
  return out;
})();
"""


# --------------------------------------------------- layer 1: site profile

def compute_rating(driver, item, prof: Profile | None = None) -> int:
    """Count stars painted gold, via the browser's computed style.

    Do not replace this with the SVG mask percentage - that reads 100% for every
    star regardless of the real rating.
    """
    from selenium.webdriver.common.by import By
    prof = prof or PROFILES[0]
    filled = 0
    for star in item.find_elements(By.XPATH, prof.star):
        paths = star.find_elements(By.TAG_NAME, 'path')
        if len(paths) < 2:
            continue
        try:
            fill = driver.execute_script(
                'return window.getComputedStyle(arguments[0]).fill;', paths[1])
        except Exception:
            continue
        if fill and all(c in fill for c in GOLD):
            filled += 1
    return filled


def _extract_page(driver, prof: Profile, product_name, product_url) -> list[dict]:
    from selenium.webdriver.common.by import By
    out = []
    try:
        container = driver.find_element(By.XPATH, prof.container)
    except Exception:
        return out
    for item in container.find_elements(By.XPATH, prof.item):
        try:
            lines = [l for l in item.text.strip().split('\n') if l.strip()]
            if not lines:
                continue
            date = parse_review_date(lines[0])
            header = lines[1] if len(lines) > 1 else ''
            verified = 'Verified Purchase' in header
            # The reviewer's name sits in `header` and is deliberately not read.
            helpful = int(lines[-1]) if lines[-1].strip().isdigit() else 0
            variant = None
            for l in lines[2:]:
                if (l.startswith('Color Family') or l.startswith('Size')
                        or (':' in l and len(l) < 80 and not l.strip().isdigit())):
                    if l != header:
                        variant = l
                        break
            try:
                text = item.find_element(By.XPATH, prof.content).text.strip()
            except Exception:
                text = ''
            if not text:
                continue                       # a rating with no words is no use here
            out.append({'product_name': product_name, 'product_url': product_url,
                        'date': date, 'verified_purchase': verified,
                        'star_rating': compute_rating(driver, item, prof),
                        'review_text': text, 'variant': variant,
                        'helpful_count': helpful})
        except Exception:
            continue                           # one malformed card must not stop the run
    return out


def _scroll_to_reviews(driver, prof: Profile) -> bool:
    from selenium.webdriver.common.by import By
    for _ in range(35):
        driver.execute_script('window.scrollBy(0, 700);')
        time.sleep(0.4)
        try:
            driver.find_element(By.XPATH, prof.container)
            time.sleep(1.2)
            return True
        except Exception:
            continue
    return False


def _click_next(driver, prof: Profile) -> bool:
    """Advance one review page. False when there is no next page."""
    from selenium.webdriver.common.by import By
    try:
        bar = driver.find_element(By.XPATH, prof.pagination)
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", bar)
        time.sleep(0.7)
        btn = bar.find_element(By.XPATH, prof.next_button)
    except Exception:
        return False
    if 'disabled' in (btn.get_attribute('class') or '') or btn.get_attribute('disabled'):
        return False
    try:
        driver.execute_script('arguments[0].click();', btn)
        time.sleep(2.2)
        return True
    except Exception:
        return False


# ------------------------------------------------------------- orchestration

def _say(progress, stage, pages=0, reviews=0):
    if progress:
        try:
            progress(stage, pages, reviews)
        except Exception:
            pass                                # reporting must never break a run


def _scroll_page(driver, steps: int = 14):
    """Nudge the page down so anything lazy-loaded has a chance to appear."""
    for _ in range(steps):
        driver.execute_script('window.scrollBy(0, 900);')
        time.sleep(0.3)
    driver.execute_script('window.scrollTo(0, 0);')
    time.sleep(0.4)


def _open_reviews(driver) -> list:
    """Click a reviews tab if there is one, then let the page settle."""
    try:
        clicked = driver.execute_script(JS_OPEN_REVIEWS) or []
    except Exception:
        return []
    if clicked:
        time.sleep(1.8)
        _scroll_page(driver, steps=6)
    return clicked


def _load_more(driver, progress=None, rounds: int = 25) -> int:
    """Press "Load more" until it stops appearing, or the budget runs out."""
    n = 0
    for i in range(rounds):
        try:
            hit = driver.execute_script(JS_LOAD_MORE)
        except Exception:
            break
        if not hit:
            break
        n += 1
        time.sleep(1.4)
        _say(progress, f'Loading more reviews ({n})')
    if n:
        _scroll_page(driver, steps=6)
    return n


def _product_name(driver, prof, url) -> str:
    from selenium.webdriver.common.by import By
    if prof:
        try:
            t = driver.find_element(By.XPATH, prof.title).text.strip()
            if t:
                return t[:120]
        except Exception:
            pass
    for how in ('h1', '[itemprop="name"]', 'meta[property="og:title"]'):
        try:
            el = driver.find_element(By.CSS_SELECTOR, how)
            t = (el.get_attribute('content') if how.startswith('meta')
                 else el.text) or ''
            if t.strip():
                return t.strip()[:120]
        except Exception:
            continue
    try:
        if driver.title.strip():
            return driver.title.strip()[:120]
    except Exception:
        pass
    return url.rstrip('/').split('/')[-1].split('.html')[0].replace('-', ' ')[:90]


def _dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = (r.get('review_text', '').strip().lower(), r.get('star_rating'))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _normalise(rows, product_name, url, default_star=5) -> list[dict]:
    """Fill the fields the pipeline expects, whichever layer produced the row.

    A review with no star rating cannot be checked against its stars - that
    comparison is the whole point - so those rows are dropped rather than given
    an invented rating.
    """
    out = []
    for r in rows:
        text = (r.get('review_text') or '').strip()
        star = r.get('star_rating')
        if not text or not star:
            continue
        out.append({'product_name': product_name, 'product_url': url,
                    'date': parse_review_date(r.get('date')),
                    'verified_purchase': bool(r.get('verified_purchase', False)),
                    'star_rating': int(star), 'review_text': text,
                    'variant': r.get('variant'), 'helpful_count': r.get('helpful_count', 0)})
    return out


def scrape_product(url: str, max_pages: int = MAX_PAGES, headless: bool = True,
                   driver=None, progress=None) -> ScrapeResult:
    """Fetch every review on one product page.

    Tries the three layers in order and returns as soon as one produces rows.
    `progress` is called as progress(stage, pages_read, reviews_found) so a caller
    can report what is happening during a run that may take minutes.
    """
    url = url.strip()
    if not looks_like_product_url(url):
        raise ScrapeError('That does not look like a product web address. '
                          'Paste the full link, starting with https://')

    # Order matters: the address is checked first, so a typo is reported as a typo
    # rather than as a missing-Selenium error on a machine that has neither.
    _need_selenium()
    own_driver = driver is None
    driver = driver or make_driver(headless=headless)
    started = time.time()
    res = ScrapeResult(url=url, platform=(urlparse(url).hostname or '').lower())

    try:
        _say(progress, 'Opening the page')
        try:
            driver.get(url)
        except Exception as exc:
            raise ScrapeError(f'Could not open that page: {str(exc)[:160]}') from exc
        time.sleep(3.0)

        prof = profile_for(url)
        res.product_name = _product_name(driver, prof, url)

        # ---- layer 1: a platform we have selectors for, with real pagination
        if prof is not None:
            _say(progress, 'Reading the reviews')
            if _scroll_to_reviews(driver, prof):
                for page in range(1, min(max_pages, MAX_PAGES) + 1):
                    if page > 1:
                        if not _click_next(driver, prof):
                            break
                        try:
                            from selenium.webdriver.common.by import By
                            c = driver.find_element(By.XPATH, prof.container)
                            driver.execute_script(
                                "arguments[0].scrollIntoView({block:'start'});", c)
                            time.sleep(0.8)
                        except Exception:
                            pass
                    rows = _extract_page(driver, prof, res.product_name, url)
                    res.reviews.extend(rows)
                    res.pages_read = page
                    _say(progress, f'Read page {page}', page, len(res.reviews))
                    if not rows:
                        break
                    if len(res.reviews) >= MAX_REVIEWS:
                        res.stopped_early = True
                        break
                    if time.time() - started > TIME_BUDGET_S:
                        res.stopped_early = True
                        break
                    time.sleep(random.uniform(0.5, 1.0))
            if res.reviews:
                res.reviews = _dedupe(res.reviews)
                res.source = 'site profile'
                return res

        # ---- make sure the reviews are actually in the page before reading it
        _say(progress, 'Looking for the reviews')
        _scroll_page(driver)
        _open_reviews(driver)
        _load_more(driver, progress)

        # ---- layer 2: whatever the site publishes for search engines
        _say(progress, 'Looking for published review data')
        try:
            got = driver.execute_script(JS_STRUCTURED) or {}
        except Exception:
            got = {}
        rows = _normalise(got.get('reviews') or [], res.product_name, url)
        if rows:
            res.reviews = _dedupe(rows)
            res.pages_read = 1
            res.source = f"structured data ({got.get('how') or 'schema.org'})"
            if got.get('product'):
                res.product_name = str(got['product'])[:120]
            return res

        # ---- layer 3: last resort
        _say(progress, 'Reading the page directly')
        try:
            got = driver.execute_script(JS_GENERIC) or {}
        except Exception:
            got = {}
        rows = _normalise(got.get('reviews') or [], res.product_name, url)
        if rows:
            res.reviews = _dedupe(rows)
            res.pages_read = 1
            res.source = 'generic reader'
            return res

        raise ScrapeError(
            'No reviews could be read from that page. Either the product has no '
            'written reviews, or the site presents them in a way this reader does '
            'not recognise. Reviews shown only after clicking a tab, or loaded from '
            'a separate request, are the usual cause.')
    finally:
        if own_driver:
            try:
                driver.quit()
            except Exception:
                pass


def probe(url: str, headless: bool = True, driver=None) -> dict:
    """Report what each layer finds for a URL, without committing to one.

    For checking candidate links before relying on them in a demonstration:
    it says which layer would fire, how many reviews it would return, and shows
    the first one so the extraction can be eyeballed.
    """
    _need_selenium()
    own = driver is None
    driver = driver or make_driver(headless=headless)
    out = {'url': url, 'profile': None, 'structured': 0, 'generic': 0,
           'how': None, 'sample': None, 'product': None, 'opened': [], 'error': None}
    try:
        driver.get(url)
        time.sleep(3.0)
        prof = profile_for(url)
        out['profile'] = prof.name if prof else None
        out['product'] = _product_name(driver, prof, url)
        _scroll_page(driver)
        # the same preparation the real run does, or the probe lies about a
        # site whose reviews only appear once a tab is clicked
        out['opened'] = _open_reviews(driver)
        _load_more(driver, rounds=6)
        try:
            s = driver.execute_script(JS_STRUCTURED) or {}
        except Exception:
            s = {}
        try:
            g = driver.execute_script(JS_GENERIC) or {}
        except Exception:
            g = {}
        sr = _normalise(s.get('reviews') or [], out['product'], url)
        gr = _normalise(g.get('reviews') or [], out['product'], url)
        out['structured'] = len(sr)
        out['generic'] = len(gr)
        out['how'] = s.get('how') if sr else ('generic' if gr else None)
        pick = sr or gr
        if pick:
            out['sample'] = {'star_rating': pick[0]['star_rating'],
                             'review_text': pick[0]['review_text'][:200]}
        return out
    finally:
        if own:
            try:
                driver.quit()
            except Exception:
                pass


def probe_many(urls, headless: bool = True) -> list[dict]:
    """Probe several links with one browser - Chrome's startup dominates otherwise."""
    _need_selenium()
    driver = make_driver(headless=headless)
    out = []
    try:
        for u in urls:
            try:
                out.append(probe(u, driver=driver))
            except Exception as exc:                  # noqa: BLE001
                out.append({'url': u, 'error': str(exc)[:90], 'profile': None,
                            'structured': 0, 'generic': 0, 'how': None,
                            'sample': None, 'product': None, 'opened': []})
    finally:
        try:
            driver.quit()
        except Exception:
            pass
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    show = '--show' in sys.argv
    if not args:
        print('usage: python -m trustlens.review_scraper URL [URL ...]')
        print('       python -m trustlens.review_scraper --file candidates.txt')
        raise SystemExit(2)
    if '--file' in sys.argv:
        args = [l.strip() for l in open(args[0]) if l.strip()
                and not l.startswith('#')]

    print(f'probing {len(args)} link{"" if len(args) == 1 else "s"}'
          f'{"" if show else " (headless)"}\n')
    try:
        results = probe_many(args, headless=not show)
    except ScrapeError as e:
        print(f'  {e}')
        raise SystemExit(1)

    usable = 0
    for r in results:
        host = (urlparse(r['url']).hostname or r['url'])[:34]
        if r.get('error'):
            print(f'  {host:<36} error: {r["error"]}')
            continue
        best = max(r['structured'], r['generic'])
        if r['profile']:
            verdict, usable = f'site profile "{r["profile"]}" - run the app, it paginates', usable+1
        elif best:
            verdict, usable = f'{best} reviews via {r["how"]}', usable+1
        else:
            verdict = 'nothing readable'
        print(f'  {host:<36} {verdict}')
        if r['opened']:
            print(f'  {"":<36}   opened: {", ".join(r["opened"])}')
        if r['sample']:
            print(f'  {"":<36}   e.g. {r["sample"]["star_rating"]}* '
                  f'{r["sample"]["review_text"][:66]}')
    print(f'\n  {usable} of {len(results)} usable for a demonstration.')
    if usable < len(results):
        print('  "nothing readable" usually means the product genuinely has no '
              'reviews, or\n  they load from a separate request after the page. '
              'Try another product on\n  the same site before writing the site off.')
