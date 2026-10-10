"""The Composite Inconsistency Index - the research contribution.

    CII = w1*SSG + w2*ASCS + w3*ISG

Three ways a review can contradict itself, each on [0, 1]:

  SSG   Sentiment-Star Gap    the text's sentiment against the stars given
  ASCS  Aspect-Star Conflict  aspect opinions pointing against the stars
  ISG   Intensity-Star Gap    emotional emphasis against the stars

Definitions
-----------
SSG  = |s - (r-3)/2| / 2      s = P(positive) - P(negative) on [-1, 1]
                              (r-3)/2 maps 1..5 stars onto [-1, 1]
                              divided by 2 because the difference spans [-2, 2]

ASCS = contradicting aspects / aspects mentioned
       A 4- or 5-star review is contradicted by a Negative aspect; a 1- or 2-star
       review by a Positive one. Neutral never contradicts. Three-star reviews are
       EXCLUDED - a neutral rating gives no direction to contradict - and return
       NaN, not 0. Coding a skipped row as zero would quietly call it consistent.
       A review mentioning no aspect also returns NaN.

ISG  = |intensity - expected intensity at that star rating| / scale
       Intensity combines four markers - elongated letters, emoji, exclamation and
       question marks, intensifier words - each normalised by its 95th percentile
       IN THE TRAINING SET, clipped to [0, 1], then averaged.

       The expected intensity is measured from the training data, one value per
       star rating, NOT assumed. The obvious assumption - that extreme ratings
       imply intense language, so expected intensity is |r-3|/2 - is wrong on this
       corpus and was tried first. Mean intensity here is 0.10, so |0.10 - 1.0|
       scored almost every plain five-star review ("Great.") as maximally
       inconsistent, and ISG measured nothing but the absence of emoji. Centring on
       the observed per-star mean makes ISG what its name says: emphasis unusual
       for that rating. The original linear form is kept as `isg_linear` so the
       pre-specified definition can still be reported alongside.

Missing components
------------------
When ASCS is NaN the weights are renormalised over the components that exist, so
CII stays defined and on [0, 1]. The three components are also returned in their
own columns, NaN intact, so the fake-review detector can use them directly -
Random Forest handles missing values natively.

Build the columns:  python -m trustlens.cii
"""
from __future__ import annotations
import re
import json
import numpy as np
import pandas as pd
from . import config, data, splits

FEATURES_PATH = config.PROCESSED/'cii_features.csv'
FIT_PATH = config.MODELS/'cii_fit.json'

# Word-boundary matching throughout: a naive substring test makes "very" match
# inside "delivery", which is exactly the bug this guards against.
INTENSIFIERS = [
    # English
    'very', 'really', 'so', 'super', 'extremely', 'absolutely', 'totally',
    'highly', 'truly', 'definitely', 'perfect', 'perfectly', 'amazing',
    'excellent', 'awesome', 'best', 'worst', 'terrible', 'horrible', 'wonderful',
    'fantastic', 'brilliant', 'outstanding', 'superb', 'must',
    # Singlish (romanised Sinhala)
    'godak', 'hari', 'niyama', 'niyamai', 'supiri', 'harima', 'boruwata',
    # Sinhala script
    'ගොඩක්', 'හරි', 'නියම', 'නියමයි', 'සුපිරි', 'හරිම', 'ඉතාම',
]
INTENSIFIER_RE = re.compile(
    r'(?<!\w)(' + '|'.join(re.escape(w) for w in INTENSIFIERS) + r')(?!\w)',
    re.IGNORECASE)
ELONGATION_RE = re.compile(r'(\w)\1{2,}')          # 3+ of the same letter: "sooo"
PUNCT_RE = re.compile(r'[!?]')
EMOJI_RE = re.compile(
    '[\U0001F300-\U0001FAFF☀-➿⬀-⯿️❤⭐]')

MARKERS = ['elongation', 'emoji', 'punctuation', 'intensifier']


# ------------------------------------------------------------------ components

def star_value(star) -> np.ndarray:
    """Map 1..5 stars onto [-1, 1]. Five stars -> +1, one star -> -1."""
    return (np.asarray(star, dtype=float) - 3.0)/2.0


def ssg(sentiment_score, star) -> np.ndarray:
    """Sentiment-Star Gap on [0, 1]."""
    s = np.asarray(sentiment_score, dtype=float)
    return np.abs(s - star_value(star))/2.0


def ascs(aspect_dicts, star) -> np.ndarray:
    """Aspect-Star Conflict Score on [0, 1]. NaN for 3 stars or no aspects."""
    star = np.asarray(star, dtype=float)
    out = np.full(len(star), np.nan)
    for i, (d, r) in enumerate(zip(aspect_dicts, star)):
        if r == 3 or not d:
            continue
        opposing = 'Negative' if r > 3 else 'Positive'
        out[i] = sum(1 for p in d.values() if p == opposing)/len(d)
    return out


def marker_counts(texts) -> pd.DataFrame:
    """Raw counts of the four intensity markers."""
    rows = []
    for t in texts:
        t = str(t)
        rows.append({'elongation': len(ELONGATION_RE.findall(t)),
                     'emoji': len(EMOJI_RE.findall(t)),
                     'punctuation': len(PUNCT_RE.findall(t)),
                     'intensifier': len(INTENSIFIER_RE.findall(t))})
    return pd.DataFrame(rows, columns=MARKERS)


def fit_intensity(texts) -> dict:
    """95th-percentile scale per marker. Fit on TRAINING text only."""
    c = marker_counts(texts)
    return {m: float(max(np.percentile(c[m], 95), 1.0)) for m in MARKERS}


def intensity(texts, scale: dict) -> np.ndarray:
    """Combined emphasis on [0, 1]: each marker scaled, clipped, then averaged."""
    c = marker_counts(texts)
    norm = np.stack([np.clip(c[m].values/scale[m], 0, 1) for m in MARKERS])
    return norm.mean(axis=0)


def isg_linear(texts, star, scale: dict) -> np.ndarray:
    """The pre-specified form: gap from |r-3|/2. Kept for comparison only.

    On this corpus it is dominated by the absence of emphasis rather than by any
    inconsistency - see the module docstring.
    """
    implied = np.abs(np.asarray(star, dtype=float) - 3.0)/2.0
    return np.abs(intensity(texts, scale) - implied)


def fit_expected(texts, star, sentiment_score, scale: dict) -> dict:
    """Per-star mean intensity and mean sentiment, measured on TRAINING data.

    These centre ISG and SSG on what a review at that rating actually looks like,
    rather than on an assumed linear mapping.
    """
    star = np.asarray(star, dtype=float)
    inten = intensity(texts, scale)
    sent = np.asarray(sentiment_score, dtype=float)
    out = {'intensity': {}, 'sentiment': {}}
    for r in np.unique(star):
        m = star == r
        out['intensity'][str(int(r))] = float(inten[m].mean())
        out['sentiment'][str(int(r))] = float(sent[m].mean())
    return out


def _expected_for(star, table: dict, fallback) -> np.ndarray:
    star = np.asarray(star, dtype=float)
    return np.array([table.get(str(int(r)), fallback(r)) for r in star], dtype=float)


def isg(texts, star, scale: dict, expected: dict | None = None) -> np.ndarray:
    """Intensity-Star Gap on [0, 1], centred on the observed per-star mean."""
    inten = intensity(texts, scale)
    if expected is None:
        return isg_linear(texts, star, scale)
    exp = _expected_for(star, expected['intensity'], lambda r: abs(r - 3)/2)
    gap = np.abs(inten - exp)
    return np.clip(gap/np.maximum(np.maximum(exp, 1 - exp), 1e-6), 0, 1)


def ssg_centred(sentiment_score, star, expected: dict) -> np.ndarray:
    """SSG measured from the observed mean sentiment at that rating.

    Step 3 found systematic offsets - two-star reviews sit about 0.30 below where
    (r-3)/2 puts them - so part of the pre-specified SSG is a constant per-star
    offset rather than inconsistency. This variant removes that.
    """
    s = np.asarray(sentiment_score, dtype=float)
    exp = _expected_for(star, expected['sentiment'], lambda r: (r - 3)/2)
    return np.clip(np.abs(s - exp)/2.0, 0, 1)


def combine(ssg_v, ascs_v, isg_v, weights: dict | None = None) -> np.ndarray:
    """Weighted sum, renormalised over whichever components are present."""
    w = weights or config.CII_WEIGHTS
    comps = np.vstack([np.asarray(ssg_v, float),
                       np.asarray(ascs_v, float),
                       np.asarray(isg_v, float)])
    wts = np.array([w['ssg'], w['ascs'], w['isg']])[:, None]*np.ones_like(comps)
    wts[np.isnan(comps)] = 0.0
    filled = np.nan_to_num(comps, nan=0.0)
    total = wts.sum(axis=0)
    return np.where(total > 0, (filled*wts).sum(axis=0)/np.maximum(total, 1e-9), np.nan)


# ------------------------------------------------------------------ pipeline

def _oof_components(train, verbose=True):
    """Out-of-fold sentiment and aspect predictions for the training set.

    The step-3 and step-4 models were fitted on the whole training set, so their
    predictions on training rows are in-sample and optimistic. Tuning the CII
    weights on those would overfit. Refitting per fold - with the hyperparameters
    already chosen, no new search - gives honest training-set components.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from . import sentiment as sent, absa as ab

    base = ab.load()
    C_det = {a: m.C for a, m in base.detectors.items()}
    C_pol = {a: (m.C if m is not None else None) for a, m in base.polarity.items()}
    sent_params = json.loads((config.MODELS/'sentiment_meta.json').read_text())['best_params']

    X, y = train['review_text'].values, train['sentiment_polarity'].values
    folds = train['fold'].astype(int).values
    scores = np.zeros(len(train))
    aspects: list[dict] = [dict() for _ in range(len(train))]

    for f in sorted(set(folds)):
        tr, va = folds != f, folds == f
        if verbose:
            print(f'  fold {f}: refitting on {tr.sum()} rows', flush=True)
        # sentiment
        sm = sent._pipeline().set_params(
            tfidf__ngram_range=tuple(sent_params['tfidf__ngram_range']),
            tfidf__min_df=sent_params['tfidf__min_df'],
            tfidf__sublinear_tf=sent_params['tfidf__sublinear_tf'],
            clf__C=sent_params['clf__C'])
        keep = y[tr] != ''
        sm.fit(X[tr][keep], y[tr][keep])
        scores[va] = sent.score(sm, X[va])
        # aspects
        vec = TfidfVectorizer(**ab.VECTORIZER).fit(X[tr])
        Xtr, Xva = vec.transform(X[tr]), vec.transform(X[va])
        M = data.aspect_matrix(train)
        for aspect in config.ASPECTS:
            present = (M[aspect].values != '')
            det = LogisticRegression(class_weight='balanced', max_iter=4000,
                                     random_state=config.SEED, C=C_det[aspect])
            det.fit(Xtr, present[tr].astype(int))
            hit = det.predict(Xva).astype(bool)
            if not hit.any() or C_pol[aspect] is None:
                continue
            sub = present[tr]
            labs = M[aspect].values[tr][sub]
            keep_cls = [c for c, n in pd.Series(labs).value_counts().items() if n >= MIN_CLASS_OOF]
            ok = np.isin(labs, keep_cls)
            if ok.sum() < 20 or len(keep_cls) < 2:
                continue
            pol = LogisticRegression(class_weight='balanced', max_iter=4000,
                                     random_state=config.SEED, C=C_pol[aspect])
            pol.fit(Xtr[sub][ok], labs[ok])
            preds = pol.predict(Xva[hit])
            for idx, lab in zip(np.flatnonzero(va)[hit], preds):
                aspects[idx][aspect] = str(lab)
    return scores, aspects


MIN_CLASS_OOF = 5


def _annotated_aspects(df) -> list[dict]:
    """The aspect dicts as annotated - the oracle upper bound, never deployable."""
    M = data.aspect_matrix(df)
    return [{a: p for a, p in row.items() if p} for _, row in M.iterrows()]


def tune_weights(ssg_v, ascs_v, isg_v, y_binary, step: float = 0.05) -> tuple[dict, pd.DataFrame]:
    """Grid search the three weights on the simplex, maximising authenticity AUC.

    Reported alongside the neutral 1/3 baseline, so the weights are shown to be
    derived rather than chosen.
    """
    from sklearn.metrics import roc_auc_score
    rows = []
    grid = np.arange(0, 1 + 1e-9, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1 - w1 - w2
            if w3 < -1e-9:
                continue
            w = {'ssg': w1, 'ascs': w2, 'isg': max(w3, 0.0)}
            v = combine(ssg_v, ascs_v, isg_v, w)
            ok = ~np.isnan(v)
            if ok.sum() < 50:
                continue
            rows.append({**w, 'auc': roc_auc_score(y_binary[ok], v[ok])})
    t = pd.DataFrame(rows).sort_values('auc', ascending=False).reset_index(drop=True)
    best = {k: round(float(t.iloc[0][k]), 3) for k in ['ssg', 'ascs', 'isg']}
    return best, t


def build_features(verbose: bool = True) -> dict:
    """Compute every CII column for the whole corpus and save them.

    Two aspect variants are produced:
      predicted - what the deployed system can actually compute
      oracle    - using the annotated aspects, an upper bound only
    The gap between them measures how much ASCS loses to aspect-model error.
    """
    from . import sentiment as sent, absa as ab
    df = splits.attach(data.load_gold())
    tr_mask = (df['split'] == 'train').values
    train = df[tr_mask].reset_index(drop=True)

    scale = fit_intensity(train['review_text'].values)
    if verbose:
        print('intensity scale (95th percentile on train):', scale)
        print('out-of-fold components for the training set:')
    s_tr, a_tr = _oof_components(train, verbose)
    expected = fit_expected(train['review_text'].values, train['star_rating'].values,
                            s_tr, scale)
    if verbose:
        print('expected intensity per star:',
              {k: round(v, 3) for k, v in expected['intensity'].items()})

    test = df[~tr_mask].reset_index(drop=True)
    sm, am = sent.load(), ab.load()
    s_te = sent.score(sm, test['review_text'].values)
    a_te = am.predict(test['review_text'].values)

    parts = []
    for frame, s, a_pred in [(train, s_tr, a_tr), (test, s_te, a_te)]:
        a_orac = _annotated_aspects(frame)
        out = pd.DataFrame({
            'review_id': frame['review_id'].values,
            'split': frame['split'].values,
            'star_rating': frame['star_rating'].values,
            'authenticity': frame['authenticity'].values,
            'n_chars': frame['n_chars'].values,
            'sentiment_score': s,
            'ssg': ssg(s, frame['star_rating'].values),
            'ssg_centred': ssg_centred(s, frame['star_rating'].values, expected),
            'ascs': ascs(a_pred, frame['star_rating'].values),
            'ascs_oracle': ascs(a_orac, frame['star_rating'].values),
            'isg_linear': isg_linear(frame['review_text'].values,
                                     frame['star_rating'].values, scale),
            'isg': isg(frame['review_text'].values, frame['star_rating'].values,
                       scale, expected),
        })
        parts.append(out)
    feat = pd.concat(parts, ignore_index=True)

    tr = feat[feat['split'] == 'train']
    y = (tr['authenticity'] == 'Suspicious').astype(int).values
    best, grid = tune_weights(tr['ssg_centred'].values, tr['ascs'].values,
                              tr['isg'].values, y)
    baseline = dict(config.CII_WEIGHTS)

    # pre-specified definition, exactly as designed
    feat['cii_prespecified'] = combine(feat['ssg'], feat['ascs'],
                                       feat['isg_linear'], baseline)
    # empirically centred components, equal weights and tuned weights
    feat['cii_baseline'] = combine(feat['ssg_centred'], feat['ascs'],
                                   feat['isg'], baseline)
    feat['cii'] = combine(feat['ssg_centred'], feat['ascs'], feat['isg'], best)
    feat['cii_oracle'] = combine(feat['ssg_centred'], feat['ascs_oracle'],
                                 feat['isg'], best)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(FEATURES_PATH, index=False)
    fit = {'intensity_scale': scale, 'expected_per_star': expected,
           'weights_tuned': best, 'weights_baseline': baseline,
           'tuned_on': 'training split, out-of-fold components, maximising authenticity ROC-AUC',
           'markers': MARKERS, 'n_intensifiers': len(INTENSIFIERS)}
    FIT_PATH.write_text(json.dumps(fit, indent=2))
    if verbose:
        print(f'\ntuned weights {best}   (baseline {baseline})')
        print(f'wrote {FEATURES_PATH.name} and {FIT_PATH.name}')
    return {'features': feat, 'grid': grid, 'fit': fit}


def evaluate(feat: pd.DataFrame) -> pd.DataFrame:
    """How well each component and each CII variant separates Suspicious, on test."""
    from sklearn.metrics import roc_auc_score
    te = feat[feat['split'] == 'test']
    y = (te['authenticity'] == 'Suspicious').astype(int).values
    rows = []
    for col, label in [('ssg', 'SSG, pre-specified'), ('ssg_centred', 'SSG, centred'),
                       ('ascs', 'ASCS, predicted aspects'),
                       ('ascs_oracle', 'ASCS, annotated aspects'),
                       ('isg_linear', 'ISG, pre-specified'), ('isg', 'ISG, centred'),
                       ('cii_prespecified', 'CII, pre-specified definition'),
                       ('cii_baseline', 'CII, centred, equal 1/3 weights'),
                       ('cii', 'CII, centred, tuned weights'),
                       ('cii_oracle', 'CII, centred, tuned, annotated aspects'),
                       ('n_chars', 'Review length (the baseline to beat)')]:
        v = te[col].values.astype(float)
        if col == 'n_chars':
            v = -v                      # shorter means more suspicious
        ok = ~np.isnan(v)
        rows.append({'Feature': label, 'Rows scored': int(ok.sum()),
                     'ROC-AUC': round(roc_auc_score(y[ok], v[ok]), 3)})
    return pd.DataFrame(rows)


LENGTH_STRATA = [0, 20, 40]


def _boot_auc(y, v, n=4000, seed=None):
    """Point estimate plus a stratified bootstrap 95% CI."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    ok = ~np.isnan(v)
    y, v = np.asarray(y)[ok], np.asarray(v)[ok]
    if y.sum() < 3 or (1 - y).sum() < 3:
        return None
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        if len(set(y[i])) > 1:
            draws.append(roc_auc_score(y[i], v[i]))
    return (roc_auc_score(y, v), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


def evaluate_by_length(feat: pd.DataFrame) -> pd.DataFrame:
    """The headline result: where length works, and where only CII does.

    Review length predicts the Suspicious label well overall, because the label is
    close to a length rule. But it carries no signal at all once reviews pass about
    twenty characters - and that is exactly where CII works. The two are
    complementary rather than competing, which is the honest framing of the
    contribution.

    Positive counts in the longer strata are small (10 and 6). The intervals are
    tight because the separation is near-complete, not because the sample is large;
    say so when reporting.
    """
    te = feat[feat['split'] == 'test'].copy()
    te['y'] = (te['authenticity'] == 'Suspicious').astype(int)
    rows = []
    for lo in LENGTH_STRATA:
        sub = te[te['n_chars'] >= lo]
        for col, label, inv in [('n_chars', 'Review length', True),
                                ('cii', 'CII (predicted aspects)', False),
                                ('cii_oracle', 'CII (annotated aspects)', False)]:
            v = -sub[col].values.astype(float) if inv else sub[col].values.astype(float)
            r = _boot_auc(sub['y'].values, v)
            if r is None:
                continue
            a, lo_ci, hi_ci = r
            rows.append({'Reviews of length': f'{lo}+ characters', 'Feature': label,
                         'n': len(sub), 'Suspicious': int(sub['y'].sum()),
                         'ROC-AUC': round(a, 3), 'CI low': round(lo_ci, 3),
                         'CI high': round(hi_ci, 3),
                         'Beats chance': 'yes' if lo_ci > 0.5 else 'no'})
    return pd.DataFrame(rows)


def fig_auc_by_length(strat: pd.DataFrame):
    """Point and whisker per stratum - the range form, not floating bars."""
    import matplotlib.pyplot as plt
    from . import eda
    eda.setup_style()
    strata = list(dict.fromkeys(strat['Reviews of length']))
    series = [('Review length', eda.ORANGE), ('CII (predicted aspects)', eda.BLUE)]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    off = 0.14
    for k, (name, colour) in enumerate(series):
        d = strat[strat['Feature'] == name].set_index('Reviews of length')
        xs = [i + (k - 0.5)*2*off for i in range(len(strata))]
        for x, st in zip(xs, strata):
            if st not in d.index:
                continue
            r = d.loc[st]
            ax.plot([x, x], [r['CI low'], r['CI high']], color=colour, linewidth=1.6,
                    solid_capstyle='round', zorder=2)
            ax.plot(x, r['ROC-AUC'], 'o', color=colour, markersize=8,
                    markeredgecolor=eda.SURFACE, markeredgewidth=1.6, zorder=3)
            ax.text(x + 0.055, r['ROC-AUC'], f"{r['ROC-AUC']:.2f}", va='center',
                    fontsize=7, color=eda.INK2)
        ax.plot([], [], 'o-', color=colour, label=name, markersize=6)
    ax.axhline(0.5, color=eda.BASE, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(len(strata) - 0.45, 0.515, 'chance', fontsize=7, color=eda.MUTED, ha='right')
    ax.set_xticks(range(len(strata)),
                  [f"{st}\n(n={strat[strat['Reviews of length']==st]['n'].iat[0]}, "
                   f"{strat[strat['Reviews of length']==st]['Suspicious'].iat[0]} suspicious)"
                   for st in strata], fontsize=7.5)
    ax.set_ylabel('ROC-AUC, Suspicious'); ax.set_ylim(0.2, 1.05)
    ax.set_title('Where each signal works, with 95% confidence intervals')
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc='upper left')        # lower right sits on the 40+ length whisker
    return eda._save(fig, '11_cii_auc_by_length')


def fig_cii_distribution(feat: pd.DataFrame):
    import matplotlib.pyplot as plt
    from . import eda
    eda.setup_style()
    te = feat[feat['split'] == 'test']
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    groups = [('Genuine', eda.BLUE), ('Suspicious', eda.ORANGE)]
    for i, (g, colour) in enumerate(groups):
        v = te[te['authenticity'] == g]['cii'].dropna().values
        parts = ax.violinplot(v, positions=[i], widths=0.7, showextrema=False,
                              showmedians=True)
        for b in parts['bodies']:
            b.set(facecolor=colour, alpha=0.35, edgecolor=colour, linewidth=1.2)
        parts['cmedians'].set(color=eda.INK, linewidth=1.4)
        ax.text(i, -0.07, f'n={len(v)}', ha='center', fontsize=7, color=eda.MUTED)
    ax.set_xticks(range(2), [g for g, _ in groups], fontsize=8)
    ax.set_ylabel('CII'); ax.set_ylim(-0.12, 1.05)
    ax.set_title('CII by annotated authenticity (test set)')
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    return eda._save(fig, '12_cii_distribution')


def report_all():
    from . import eda
    res = build_features()
    feat = res['features']
    ev = evaluate(feat)
    print('\n=== separation of Suspicious on the test set ===')
    print(ev.to_string(index=False))
    print('\ntables ->', config.TABLES)
    eda._save_table(ev, '14_cii_auc')
    eda._save_table(res['grid'].head(15).round(4), '15_cii_weight_grid')
    desc = feat.groupby('split')[['ssg', 'ascs', 'isg', 'cii']].describe().T.round(3)
    eda._save_table(desc.reset_index(), '16_cii_distributions')
    strat = evaluate_by_length(feat)
    print('\n=== where each signal works (95% CI) ===')
    print(strat.to_string(index=False))
    eda._save_table(strat, '17_cii_auc_by_length')
    print('\nfigures ->', config.FIGURES)
    fig_auc_by_length(strat)
    fig_cii_distribution(feat)
    res['evaluation'] = ev
    res['stratified'] = strat
    return res


if __name__ == '__main__':
    from trustlens import cii as _cii
    _cii.report_all()
