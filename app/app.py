"""TrustLens web application.

The landing page asks for one thing: a product link. Reviews are fetched live,
every page of them, and run through the trained pipeline.

Fetching every review can take minutes, which is far too long to hold a web
request open, so a scrape runs on a background thread and the page polls it.
That is also what makes a progress indicator possible - a synchronous request
can only show a spinner, because nothing can report back until it is finished.

`analyse_reviews()` is the seam: hand it texts and star ratings from any source
and everything downstream works unchanged.

/demo keeps the offline paths - a product from the annotated corpus, or reviews
typed by hand. They need neither network nor Chrome, so a demonstration always
has a route that cannot fail on someone else's wifi. It is not linked from the
landing page because the landing page is meant to show one idea.

Run:  python app/app.py              then open http://127.0.0.1:5000
      TRUSTLENS_SHOW_BROWSER=1 python app/app.py   to watch Chrome work
"""
from __future__ import annotations
import os
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, redirect, render_template, request, url_for
from trustlens import aggregate, data
from trustlens import review_scraper as scraper

try:                      # imported as a package: python -c "from app.app import ..."
    from . import charts, lexicon
except ImportError:       # run as a script:        python app/app.py
    import charts
    import lexicon

app = Flask(__name__)

# Set TRUSTLENS_SHOW_BROWSER=1 to watch Chrome work. Worth doing in a live
# demonstration: the audience sees the pages being read, and a site that refuses
# a headless browser often accepts a visible one.
HEADLESS = os.environ.get('TRUSTLENS_SHOW_BROWSER', '') not in ('1', 'true', 'yes')

PIPELINE = None
GOLD = None
PRODUCTS: list[tuple[str, str, int]] = []

# Sentiment is a continuous score, P(positive) - P(negative), and the lexicon
# correction adjusts that score rather than a class. To show a class without
# contradicting the score the rest of the page uses, the cut points were chosen
# to reproduce the model's own argmax label: they agree on 98.9% of the corpus.
POS_AT, NEG_AT = 0.100, -0.275


def warm():
    """Load models and the corpus once, at startup. Never on a request."""
    global PIPELINE, GOLD, PRODUCTS
    if PIPELINE is None:
        # Deployment-time lexicon correction - see app/lexicon.py for the
        # measured trade-off and why it is not part of the trained system.
        PIPELINE = aggregate.Pipeline(sentiment_adjust=lexicon.correct)
        GOLD = data.load_gold()
        counts = GOLD.groupby('product_name').size()
        PRODUCTS = sorted(
            ((n, GOLD[GOLD.product_name == n]['category'].iat[0], int(c))
             for n, c in counts.items() if c >= 5),
            key=lambda x: -x[2])
    return PIPELINE


def sentiment_class(score: float) -> str:
    return ('Positive' if score > POS_AT
            else 'Negative' if score < NEG_AT else 'Neutral')


def _aspect_bars(aspects: dict) -> list[dict]:
    """Aspect counts as percentages, ready to draw."""
    out = []
    for name, d in aspects.items():
        total = sum(d.values())
        if not total:
            continue
        out.append({'name': name, 'total': total,
                    'pos': round(100*d['Positive']/total),
                    'neu': round(100*d['Neutral']/total),
                    'neg': round(100*d['Negative']/total)})
    return sorted(out, key=lambda a: -a['total'])


def analyse_reviews(texts, stars, product_name=None) -> dict:
    """The seam. Any source of reviews plugs in here."""
    p = warm()
    r = p.score_product(list(texts), list(stars), product_name)
    rows = r['reviews'].copy()
    rows['trust_pct'] = (rows['trust']*100).round().astype(int)
    rows['sentiment'] = [sentiment_class(s) for s in rows['sentiment_score']]

    # Two separate signals, deliberately shown separately.
    #   trust  - the detector's view of whether a review is genuine. Dominated by
    #            review length, because the label it learned from is.
    #   match  - whether the words agree with the stars. This uses SSG directly,
    #            NOT the composite index.
    #
    # Why not CII here: the weights tuned in step 5 put 0.95 on ASCS and 0.05 on
    # SSG, because they were fitted to maximise the index's AUC against the
    # authenticity label. That makes CII almost insensitive to the sentiment-star
    # gap - a five-star rating on a plain complaint scores SSG 0.61 but CII 0.03.
    # The question this column asks is exactly what SSG measures, so it asks SSG.
    def _match(g):
        if g != g:
            return ('Not checked', 'na')
        if g >= 0.50:
            return ('No', 'low')
        if g >= 0.30:
            return ('Not really', 'mid')
        return ('Yes', 'ok')
    rows[['match', 'match_cls']] = [_match(g) for g in rows['ssg']]
    rows = rows.sort_values(['ssg', 'trust'], ascending=[False, True])

    # The hard alternative, reported next to the soft one rather than instead of
    # it. Down-weighting barely moves a product whose flagged reviews are
    # five-star and agree with the majority they are diluted into - which
    # describes most products with a high average. Showing both bounds the effect
    # honestly; showing only whichever happens to look larger would not.
    hard = aggregate.filtered_score(rows['star_rating'], rows['p_suspicious'])
    hard = None if hard != hard else round(float(hard), 2)
    flagged = int(round(r['suspicious_share']*r['n_reviews']))
    mismatched = int((rows['ssg'] >= 0.50).sum())

    # Why the visible rating can sit still while reviews are flagged. Stated in
    # the interface so the result is not left looking like a failure when it is
    # not.
    if abs(r['shift']) < 0.005 and flagged == 0:
        why = ('Nothing was flagged, so the rating is unchanged.'
               if mismatched == 0 else
               f'The rating is unchanged because none of these reviews were '
               f'flagged as doubtful - including the {mismatched} whose wording '
               f'does not match its stars.')
    elif abs(r['shift']) < 0.005:
        why = (f'The rating barely moves: the {flagged} flagged review'
               f'{"" if flagged == 1 else "s"} rate about the same as the rest, '
               f'so counting {"it" if flagged == 1 else "them"} for less changes '
               f'little. Weighting only shifts a rating when the doubtful reviews '
               f'disagree with the honest ones.')
    else:
        why = None

    star_counts = rows['star_rating'].value_counts().to_dict()
    by_star = []
    for s in (5, 4, 3, 2, 1):
        sub = rows[rows['star_rating'] == s]
        if not len(sub):
            continue
        c = sub['sentiment'].value_counts()
        by_star.append((f'{s} ★', int(c.get('Negative', 0)),
                        int(c.get('Neutral', 0)), int(c.get('Positive', 0))))
    bars = _aspect_bars(r['aspects'])
    aspect_rows = [(b['name'],
                    round(b['total']*b['neg']/100),
                    round(b['total']*b['neu']/100),
                    round(b['total']*b['pos']/100)) for b in bars]

    return {
        'product': product_name,
        'n': r['n_reviews'],
        'plain': r['plain_mean'],
        'weighted': r['trust_weighted'],
        'shift': r['shift'],
        'filtered': hard,
        'shift_filtered': None if hard is None else round(hard - r['plain_mean'], 2),
        'why': why,
        'flagged': flagged,
        'mismatched': mismatched,
        'bars': bars,
        'charts': {
            'rating': charts.rating_comparison(r['plain_mean'], r['trust_weighted'], hard),
            'stars': charts.star_distribution(star_counts),
            'words': charts.diverging_stacked(
                by_star, 'What the words say, at each star rating',
                'a long red arm on a five-star row is the case this system exists for',
                label_w=54),
            'aspects': charts.diverging_stacked(
                aspect_rows, 'What the reviews are about',
                'sentiment per aspect, across every review read'),
        },
        'reviews': rows.to_dict('records'),
    }


# ------------------------------------------------------------- scraping jobs
# A scrape reads every review page and can run for minutes. Holding a request
# open that long gives the browser nothing to show and risks a gateway timeout,
# so the work goes on a thread and the page asks how it is going.

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_KEEP = 12                       # finished jobs retained, so a reload still works


def _start_job(url: str) -> str:
    jid = uuid4().hex[:12]
    with _LOCK:
        for old in list(_JOBS)[:-_KEEP]:
            _JOBS.pop(old, None)
        _JOBS[jid] = {'stage': 'Starting up', 'pages': 0, 'reviews': 0,
                      'done': False, 'error': None, 'result': None,
                      'url': url, 'started': time.time()}
    threading.Thread(target=_run_job, args=(jid, url), daemon=True).start()
    return jid


def _set(jid, **kw):
    with _LOCK:
        if jid in _JOBS:
            _JOBS[jid].update(kw)


def _run_job(jid: str, url: str):
    def progress(stage, pages, reviews):
        _set(jid, stage=stage, pages=pages, reviews=reviews)
    try:
        sc = scraper.scrape_product(url, headless=HEADLESS, progress=progress)
        _set(jid, stage=f'Analysing {len(sc.reviews)} reviews',
             reviews=len(sc.reviews))
        res = analyse_reviews(sc.texts(), sc.stars(), sc.product_name)
        res.update(source=sc.source, platform=sc.platform, pages=sc.pages_read,
                   stopped_early=sc.stopped_early, url=url)
        _set(jid, done=True, result=res, stage='Done')
    except scraper.ScrapeError as exc:
        _set(jid, done=True, error=str(exc))
    except Exception as exc:                          # noqa: BLE001
        _set(jid, done=True, error=f'Something went wrong: {exc}')


# ----------------------------------------------------------------- routes

@app.route('/')
def index():
    warm()
    return render_template('index.html', error=request.args.get('error'))


@app.post('/scrape')
def start_scrape():
    url = (request.form.get('url') or '').strip()
    if not url:
        return jsonify(error='Paste a product link first.'), 400
    if not scraper.looks_like_product_url(url):
        return jsonify(error='That does not look like a product web address. '
                             'Paste the full link, starting with https://'), 400
    return jsonify(job=_start_job(url))


@app.get('/progress/<jid>')
def progress(jid):
    with _LOCK:
        j = _JOBS.get(jid)
        if j is None:
            return jsonify(error='That run has expired. Try the link again.'), 404
        return jsonify(stage=j['stage'], pages=j['pages'], reviews=j['reviews'],
                       done=j['done'], error=j['error'],
                       elapsed=round(time.time()-j['started']))


@app.get('/result/<jid>')
def result(jid):
    warm()
    with _LOCK:
        j = _JOBS.get(jid)
    if j is None:
        return redirect(url_for('index', error='That run has expired. '
                                               'Paste the link again.'))
    if j['error']:
        return redirect(url_for('index', error=j['error']))
    if not j['done']:
        return redirect(url_for('index'))
    return render_template('result.html', result=j['result'])


@app.route('/demo', methods=['GET', 'POST'])
def demo():
    """Offline fallback: the annotated corpus, or reviews typed by hand."""
    warm()
    result = error = None
    mode = request.form.get('mode', 'product')
    chosen = request.form.get('product', '')
    pasted = request.form.get('reviews', '')
    if request.method == 'POST':
        try:
            if mode == 'product' and chosen:
                g = GOLD[GOLD.product_name == chosen]
                result = analyse_reviews(g['review_text'].tolist(),
                                         g['star_rating'].astype(int).tolist(),
                                         chosen)
            elif mode == 'text' and pasted.strip():
                texts, stars = [], []
                for line in pasted.strip().splitlines():
                    if '|' not in line:
                        continue
                    a, b = line.split('|', 1)
                    try:
                        s = int(float(a.strip()))
                    except ValueError:
                        continue
                    if 1 <= s <= 5 and b.strip():
                        stars.append(s)
                        texts.append(b.strip())
                if texts:
                    result = analyse_reviews(texts, stars, 'Your reviews')
                else:
                    error = ('No reviews found. Use one line per review, '
                             'like:  5 | the review text')
            else:
                error = 'Choose a product, or paste some reviews.'
        except Exception as exc:                      # noqa: BLE001
            error = f'Something went wrong: {exc}'
    return render_template('demo.html', products=PRODUCTS, result=result,
                           error=error, mode=mode, chosen=chosen, pasted=pasted)


@app.get('/health')
def health():
    warm()
    return {'status': 'ok', 'products': len(PRODUCTS), 'headless': HEADLESS}


if __name__ == '__main__':
    print('loading models...')
    warm()
    print(f'ready - {len(PRODUCTS)} products in the offline fallback (/demo)')
    print(f'chrome will be {"headless" if HEADLESS else "VISIBLE"}')
    print('open http://127.0.0.1:5000')
    app.run(debug=False, port=5000, threaded=True)
