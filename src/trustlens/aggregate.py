"""Trust-weighted product scoring, and the one pipeline the web app calls.

A product's score is a weighted mean of its star ratings, where each review's
weight is how genuine the detector thinks it is:

    score = sum(r_i * w_i) / sum(w_i),    w_i = 1 - P(suspicious_i)

The gap between that and the plain mean is the headline number of the whole
system: how much the visible rating moves once suspicious reviews stop counting
as much as genuine ones.

`Pipeline` loads every model once and runs a batch of reviews end to end -
sentiment, aspects, the index, the detector. The Flask app holds one instance;
it never retrains on a request.

    from trustlens import aggregate
    p = aggregate.Pipeline()
    p.analyse(['good product', 'stopped working after a week'], [5, 5])
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from . import config, data, cii, sentiment, absa, fake

SCORES_PATH = config.TABLES/'22_product_scores.csv'


def trust_weight(p_suspicious) -> np.ndarray:
    """1 - P(suspicious), clipped away from zero so no review is silenced entirely."""
    return np.clip(1.0 - np.asarray(p_suspicious, dtype=float), 0.02, 1.0)


def weighted_score(stars, weights) -> float:
    stars = np.asarray(stars, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float((stars*w).sum()/w.sum()) if w.sum() > 0 else float('nan')


class Pipeline:
    """Every model, loaded once. Thread-safe for reads; the app keeps one."""

    def __init__(self, sentiment_adjust=None):
        """sentiment_adjust: optional fn(scores, texts) -> scores.

        A hook for deployment-time correction. Nothing in this package passes it;
        the web application does, and documents why. Leaving it None reproduces
        exactly the system the thesis measures.
        """
        self.sentiment_adjust = sentiment_adjust
        self.sentiment = sentiment.load()
        self.absa = absa.load()
        bundle = fake.load()
        self.detector = bundle['model']
        self.columns = bundle['columns']
        self.product_stats = pd.Series(bundle['product_stats'])
        self.fit = json.loads(cii.FIT_PATH.read_text())
        self.scale = self.fit['intensity_scale']
        self.expected = self.fit['expected_per_star']
        self.weights = self.fit['weights_tuned']

    def analyse(self, texts, stars, product_name: str | None = None,
                helpful=None, dup_count=None) -> pd.DataFrame:
        """Score a batch of reviews. Returns one row per review."""
        texts = [str(t) for t in texts]
        stars = np.asarray(stars, dtype=float)
        n = len(texts)

        s = sentiment.score(self.sentiment, texts)
        if self.sentiment_adjust is not None:
            s = self.sentiment_adjust(s, texts)
        aspects = self.absa.predict(texts)

        frame = pd.DataFrame({
            'review_text': texts,
            'star_rating': stars,
            'language_type': 'English',          # only gates readability features
            'helpful_count': 0 if helpful is None else helpful,
            'dup_count': 1 if dup_count is None else dup_count,
            'product_name': product_name or '(unknown)',
            'n_chars': [len(t) for t in texts],
            'review_id': [f'Q{i:04d}' for i in range(n)],
        })

        comp = pd.DataFrame({
            'review_id': frame['review_id'],
            'ssg_centred': cii.ssg_centred(s, stars, self.expected),
            'ascs': cii.ascs(aspects, stars),
            'isg': cii.isg(texts, stars, self.scale, self.expected),
        })
        comp['cii'] = cii.combine(comp['ssg_centred'], comp['ascs'], comp['isg'],
                                  self.weights)

        X = fake.build_matrix(frame, comp, fake.FAMILIES, self.product_stats)
        X = X.reindex(columns=self.columns)
        p_susp = self.detector.predict_proba(X)[:, 1]

        return pd.DataFrame({
            'review_text': texts,
            'star_rating': stars.astype(int),
            'sentiment_score': s.round(3),
            'aspects': ['; '.join(f'{a}:{p}' for a, p in d.items()) for d in aspects],
            'ssg': comp['ssg_centred'].round(3),
            'ascs': comp['ascs'].round(3),
            'isg': comp['isg'].round(3),
            'cii': comp['cii'].round(3),
            'p_suspicious': p_susp.round(3),
            'trust': trust_weight(p_susp).round(3),
        })

    def score_product(self, texts, stars, product_name=None) -> dict:
        """Plain mean against trust-weighted mean, plus the aspect summary."""
        rows = self.analyse(texts, stars, product_name)
        plain = float(np.mean(stars))
        weighted = weighted_score(rows['star_rating'], rows['trust'])
        aspects = {}
        for line in rows['aspects']:
            for part in str(line).split(';'):
                part = part.strip()
                if not part or ':' not in part:
                    continue
                cat, pol = part.rsplit(':', 1)
                d = aspects.setdefault(cat.strip(), {'Positive': 0, 'Neutral': 0,
                                                     'Negative': 0})
                if pol.strip() in d:
                    d[pol.strip()] += 1
        return {'product': product_name, 'n_reviews': len(rows),
                'plain_mean': round(plain, 3), 'trust_weighted': round(weighted, 3),
                'shift': round(weighted - plain, 3),
                'suspicious_share': round(float((rows['p_suspicious'] > 0.5).mean()), 3),
                'aspects': aspects, 'reviews': rows}


def filtered_score(stars, p_suspicious, threshold: float = 0.5) -> float:
    """Hard alternative: drop flagged reviews outright rather than down-weight.

    Soft weighting moves scores very little on this corpus, because the flagged
    reviews are overwhelmingly five-star and agree with the majority they are
    diluted into. Reporting both bounds the effect honestly rather than picking
    whichever looks better.
    """
    stars = np.asarray(stars, dtype=float)
    keep = np.asarray(p_suspicious, dtype=float) <= threshold
    return float(stars[keep].mean()) if keep.any() else float('nan')


def product_table(min_reviews: int = 10) -> pd.DataFrame:
    """Plain against trust-weighted scores for every product in the corpus.

    The distribution of `shift` is the headline evidence that the system changes
    what a shopper would see.
    """
    p = Pipeline()
    df = data.load_gold()
    rows = []
    for name, g in df.groupby('product_name'):
        if len(g) < min_reviews:
            continue
        r = p.score_product(g['review_text'].tolist(),
                            g['star_rating'].astype(int).tolist(), name)
        hard = filtered_score(r['reviews']['star_rating'],
                              r['reviews']['p_suspicious'])
        rows.append({'Product': name[:60], 'Category': g['category'].iat[0],
                     'Reviews': r['n_reviews'], 'Plain mean': r['plain_mean'],
                     'Trust-weighted': r['trust_weighted'], 'Shift': r['shift'],
                     'Filtered': round(hard, 3),
                     'Shift if filtered': round(hard - r['plain_mean'], 3),
                     'Flagged %': round(100*r['suspicious_share'], 1)})
    return pd.DataFrame(rows).sort_values('Shift').reset_index(drop=True)


def report_all():
    from . import eda
    t = product_table()
    print(t.to_string(index=False))
    print(f'\nproducts scored        : {len(t)}')
    print(f'mean shift             : {t["Shift"].mean():+.3f} stars')
    print(f'largest downward shift : {t["Shift"].min():+.3f} stars')
    print(f'products moved by 0.1+ : {(t["Shift"].abs() >= 0.1).sum()} of {len(t)}')
    print()
    print('hard filter (drop anything over P=0.5) instead of soft weighting:')
    print(f'  mean shift             : {t["Shift if filtered"].mean():+.3f} stars')
    print(f'  largest downward shift : {t["Shift if filtered"].min():+.3f} stars')
    print(f'  products moved by 0.1+ : {(t["Shift if filtered"].abs() >= 0.1).sum()} of {len(t)}')
    eda._save_table(t, '22_product_scores')
    fig_shift(t)
    return t


def fig_shift(t: pd.DataFrame):
    """How far each product's visible rating moves under the two schemes."""
    import matplotlib.pyplot as plt
    from . import eda
    eda.setup_style()
    d = t.sort_values('Shift if filtered')
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.barh(y + 0.19, d['Shift if filtered'], height=0.36, color=eda.ORANGE,
            label='Flagged reviews dropped')
    ax.barh(y - 0.19, d['Shift'], height=0.36, color=eda.BLUE,
            label='Trust-weighted')
    ax.axvline(0, color=eda.BASE, linewidth=1.0)
    ax.set_yticks(y, [p[:34] for p in d['Product']], fontsize=6.2)
    ax.set_xlabel('Change in the product\u2019s visible rating (stars)')
    ax.set_title('How much trust weighting moves a product\u2019s rating')
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.yaxis.grid(False)
    # Bottom-left is where the largest downward bars sit, so a legend there
    # lands on top of them. Lower right is the only empty corner.
    ax.legend(loc='lower right', fontsize=7, framealpha=0.95)
    return eda._save(fig, '14_rating_shift')


if __name__ == '__main__':
    from trustlens import aggregate as _a
    _a.report_all()
