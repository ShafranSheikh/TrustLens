"""The two platform-independent extraction layers, against synthetic pages.

These layers are JavaScript running inside the browser, which is the part of the
scraper most likely to break quietly: a selector that matches nothing returns an
empty list, and an empty list looks exactly like a product with no reviews. So
each idiom gets a fixture with a known answer.

The pages are loaded from data: URLs, so this needs Chrome but no network. It is
skipped when Selenium is not installed.

Run just these:  pytest tests/test_extractors.py -v
"""
from urllib.parse import quote

import pytest

from trustlens import review_scraper as rs

pytest.importorskip('selenium', reason='live-extraction tests need Selenium')


JSONLD = """<!doctype html><html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product","name":"Widget Pro 3000",
 "review":[
  {"@type":"Review","author":{"@type":"Person","name":"Alice Perera"},
   "datePublished":"2025-02-08","reviewBody":"Excellent build quality and fast delivery.",
   "reviewRating":{"@type":"Rating","ratingValue":"5"}},
  {"@type":"Review","author":{"@type":"Person","name":"Bob Silva"},
   "reviewBody":"Stopped working after two days. Waste of money.",
   "reviewRating":{"@type":"Rating","ratingValue":"5"}},
  {"@type":"Review","author":"Carol","reviewBody":"Okay for the price.",
   "reviewRating":{"@type":"Rating","ratingValue":"3"}}]}
</script></head><body><h1>Widget Pro 3000</h1></body></html>"""

GRAPH = """<!doctype html><html><head>
<script type="application/ld+json">
{"@context":"https://schema.org","@graph":[
 {"@type":"Organization","name":"Shop"},
 {"@type":["Product"],"name":"Nested Widget","review":[
   {"@type":"Review","reviewBody":"Works fine, no complaints at all here.",
    "reviewRating":{"ratingValue":4}}]}]}
</script></head><body></body></html>"""

MICRODATA = """<!doctype html><html><body>
<div itemscope itemtype="http://schema.org/Product"><h1 itemprop="name">Micro Widget</h1>
 <div itemprop="review" itemscope itemtype="http://schema.org/Review">
   <span itemprop="author">Dinesh</span>
   <meta itemprop="datePublished" content="2025-05-04">
   <span itemprop="reviewBody">Battery life is poor but the screen is bright.</span>
   <div itemprop="reviewRating" itemscope itemtype="http://schema.org/Rating">
     <meta itemprop="ratingValue" content="2"></div></div>
 <div itemprop="review" itemscope itemtype="http://schema.org/Review">
   <span itemprop="reviewBody">Arrived early and works exactly as described.</span>
   <div itemprop="reviewRating" itemscope itemtype="http://schema.org/Rating">
     <meta itemprop="ratingValue" content="5"></div></div>
</div></body></html>"""

ARIA = """<!doctype html><html><body>
<nav><a href="#">Home</a><a href="#">Shop</a></nav><h1>Generic Widget</h1>
<section class="reviews-list">
  <article><div class="rating" aria-label="5 out of 5 stars"></div>
    <p>Very good product, delivery was quick and the packaging was solid.</p></article>
  <article><div class="rating" aria-label="1 out of 5 stars"></div>
    <p>Terrible. It broke within a week and support never replied to me.</p></article>
  <article><div class="rating" aria-label="4 out of 5 stars"></div>
    <p>Good value overall although the cable feels a little bit flimsy.</p></article>
</section></body></html>"""

GLYPH = """<!doctype html><html><body><div id="c">
  <div class="rv"><span>★★★★★</span>
     <p>Sound quality is amazing for this price range, very happy.</p></div>
  <div class="rv"><span>★★</span>
     <p>Not as described, the colour is completely different from photos.</p></div>
</div></body></html>"""

WIDTH = """<!doctype html><html><body><ul id="rl">
  <li><div class="star-bar"><i style="width: 80%"></i></div>
      <p>Comfortable to wear for long periods and looks quite premium.</p></li>
  <li><div class="star-bar"><i style="width: 20%"></i></div>
      <p>Stopped charging after ten days of normal everyday use here.</p></li>
</ul></body></html>"""

PLAIN = """<!doctype html><html><body><h1>About us</h1>
<p>We are a shop that sells things to people across the country every day.</p>
</body></html>"""


@pytest.fixture(scope='module')
def driver():
    try:
        d = rs.make_driver(headless=True)
    except rs.ScrapeError as exc:
        pytest.skip(str(exc))
    yield d
    d.quit()


def _run(driver, html, js):
    driver.get('data:text/html;charset=utf-8,' + quote(html))
    return driver.execute_script(js) or {}


@pytest.mark.parametrize('label, html, js, want', [
    ('json-ld',            JSONLD,    'S', 3),
    ('json-ld in @graph',  GRAPH,     'S', 1),
    ('microdata',          MICRODATA, 'S', 2),
    ('generic aria-label', ARIA,      'G', 3),
    ('generic star glyphs', GLYPH,    'G', 2),
    ('generic bar width',  WIDTH,     'G', 2),
])
def test_each_idiom_is_read(driver, label, html, js, want):
    got = _run(driver, html, rs.JS_STRUCTURED if js == 'S' else rs.JS_GENERIC)
    assert len(got.get('reviews') or []) == want, label


def test_a_page_with_no_reviews_yields_none(driver):
    """The expensive failure is a false positive - navigation read as reviews."""
    assert not (_run(driver, PLAIN, rs.JS_GENERIC).get('reviews') or [])
    assert not (_run(driver, PLAIN, rs.JS_STRUCTURED).get('reviews') or [])


def test_structured_layer_does_not_fire_without_structured_data(driver):
    assert not (_run(driver, ARIA, rs.JS_STRUCTURED).get('reviews') or [])


def test_reviewer_names_are_never_extracted(driver):
    """Ethical clearance forbids collecting them; the JSON-LD fixture has them."""
    got = _run(driver, JSONLD, rs.JS_STRUCTURED)
    blob = ' '.join(r['review_text'] for r in got['reviews'])
    for name in ('Alice', 'Perera', 'Bob', 'Silva', 'Carol'):
        assert name not in blob


def test_ratings_survive_extraction(driver):
    got = _run(driver, MICRODATA, rs.JS_STRUCTURED)
    assert sorted(r['star_rating'] for r in got['reviews']) == [2, 5]


TABBED = """<!doctype html><html><body>
<h1>Tabbed Widget</h1>
<div role="tab">Description</div>
<button id="rv">Reviews (3)</button>
<button id="buy">Buy now</button>
<a href="/somewhere-else">Reviews</a>
<div id="panel"></div>
<button id="more" style="display:none">Load more</button>
<script>
  const R=[['5','Excellent little speaker, very loud for the size.'],
           ['1','Died after a week and the seller ignored me completely.'],
           ['4','Good but the charging cable is far too short for use.'],
           ['2','Sound is muddy at higher volume, would not buy again.']];
  let shown=0;
  function draw(n){
    const p=document.getElementById('panel');
    for(let i=0;i<n && shown<R.length;i++,shown++){
      const d=document.createElement('div');
      d.innerHTML='<span aria-label="'+R[shown][0]+' out of 5 stars"></span><p>'+R[shown][1]+'</p>';
      p.appendChild(d);
    }
    document.getElementById('more').style.display = shown<R.length ? '' : 'none';
  }
  document.getElementById('rv').onclick=()=>{draw(2);};
  document.getElementById('more').onclick=()=>{draw(2);};
  document.getElementById('buy').onclick=()=>{document.title='BOUGHT';};
</script></body></html>"""


def test_reviews_hidden_behind_a_tab_are_opened_and_fully_loaded(driver, tmp_path):
    """The commonest reason a real product page returned nothing.

    Reviews are often absent from the DOM until a tab is clicked, and then only
    partly present until "Load more" is pressed. Both must happen before the
    readers run, or a page with reviews looks like a page without any.
    """
    f = tmp_path / 'tabbed.html'
    f.write_text(TABBED, encoding='utf-8')
    driver.get(f.as_uri())

    assert not (driver.execute_script(rs.JS_GENERIC).get('reviews') or []), \
        'fixture should start with nothing readable'

    clicked = rs._open_reviews(driver)
    assert clicked == ['Reviews (3)']
    assert len(driver.execute_script(rs.JS_GENERIC)['reviews']) == 2

    assert rs._load_more(driver) == 1
    assert len(driver.execute_script(rs.JS_GENERIC)['reviews']) == 4


def test_opening_reviews_never_clicks_anything_else(driver, tmp_path):
    """Clicking by label could wander into Buy now, or navigate off the page."""
    f = tmp_path / 'tabbed2.html'
    f.write_text(TABBED, encoding='utf-8')
    driver.get(f.as_uri())
    before = driver.current_url
    rs._open_reviews(driver)
    assert driver.title != 'BOUGHT', 'clicked the buy button'
    assert driver.current_url == before, 'followed a link to another page'


def test_a_review_with_no_rating_is_dropped_not_invented():
    """Comparing words against stars is the point; a missing star has no default."""
    rows = rs._normalise([{'review_text': 'no stars here', 'star_rating': None},
                          {'review_text': 'four stars', 'star_rating': 4}],
                         'p', 'https://e.com/x')
    assert [r['review_text'] for r in rows] == ['four stars']
