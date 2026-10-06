"""Comparison with existing solutions - Chapter 5.9.

Two questions a reader of that section asks, answered against the same held-out
test split the rest of the evaluation uses:

  1. Does a purpose-built cross-lingual sentiment model earn its place, or would
     an off-the-shelf tool have done? Compared against VADER, the standard
     lexicon baseline in the review-mining literature, and against the same
     classifier on word features rather than character n-grams. Reported per
     language, because that is where the answer lives: an English lexicon does
     not fail politely on Sinhala, it fails silently, calling everything neutral.

  2. Does the detection pipeline beat what a platform could do trivially?
     The honest baselines are the ones that need no model at all - the star
     rating, and the length of the review. Length is not a strawman: the
     annotation is close to a rule about length, which Chapter 4 documents.
     Two figures describe this and they are not interchangeable - 0.951 AUC
     across the whole corpus, which is how tightly the label tracks length, and
     0.841 on the held-out test split, which is the number the system has to
     beat. Only the second is a fair comparison, so only the second appears in
     the comparison table.

Differences between approaches are tested by PAIRED bootstrap - resampling the
same test rows for both approaches and taking the difference. Comparing two
independently-computed confidence intervals and looking for overlap is the
wrong test and is much too conservative here, because the two scores are highly
correlated: they are predictions about the very same reviews.

Nothing here retrains the shipped models. Baselines are fitted on the training
split and scored on the test split, exactly as the reported results were.

    python -m trustlens.compare
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from . import config, data, eda, fake, sentiment, splits

VADER_CUT = 0.05          # the thresholds VADER's own documentation specifies
BOOT = 2000


def _boot_ci(y, s, fn, n=BOOT, seed=config.SEED):
    """Percentile bootstrap CI, so a difference can be read as real or not."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        try:
            out.append(fn(y[i], s[i]))
        except ValueError:
            continue
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) \
        if out else (float('nan'), float('nan'))


# ----------------------------------------------------------------- sentiment

def vader_labels(texts) -> np.ndarray:
    """VADER's compound score, mapped to this study's three classes.

    VADER is an English social-media lexicon. It is included precisely because
    it is the obvious thing a practitioner would reach for first, and because
    what it does on Sinhala script is the argument for not reaching for it.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        c = an.polarity_scores(str(t))['compound']
        out.append('Positive' if c >= VADER_CUT
                   else 'Negative' if c <= -VADER_CUT else 'Neutral')
    return np.array(out)


def _word_tfidf_model():
    """The same classifier on word features - isolates the character n-gram."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='word', lowercase=True,
                                  ngram_range=(1, 2), min_df=2)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=4000,
                                   random_state=config.SEED)),
    ])


def sentiment_comparison(verbose: bool = True) -> pd.DataFrame:
    """Macro-F1 overall and per language, for each approach, on the test split."""
    df = splits.attach(data.sentiment_data())
    train = df[df.split == 'train'].reset_index(drop=True)
    test = df[df.split == 'test'].reset_index(drop=True)
    ytr, yte = train['sentiment_polarity'].values, test['sentiment_polarity'].values

    preds: dict[str, np.ndarray] = {}

    maj = DummyClassifier(strategy='most_frequent').fit(
        np.zeros((len(train), 1)), ytr)
    preds['Majority class'] = maj.predict(np.zeros((len(test), 1)))

    preds['VADER (English lexicon)'] = vader_labels(test['review_text'])

    w = _word_tfidf_model().fit(train['review_text'], ytr)
    preds['Word TF-IDF + logistic regression'] = w.predict(test['review_text'])

    ours = sentiment.load()
    preds['This study (char n-gram + LR)'] = ours.predict(list(test['review_text']))

    langs = ['English', 'Sinhala', 'Singlish']
    rows = []
    for name, p in preds.items():
        row = {'Approach': name,
               'Macro-F1': round(f1_score(yte, p, average='macro',
                                          zero_division=0), 3),
               'Accuracy': round(float((p == yte).mean()), 3)}
        for lg in langs:
            m = test['language_type'].values == lg
            row[f'{lg} (n={int(m.sum())})'] = (
                round(f1_score(yte[m], p[m], average='macro',
                               zero_division=0), 3) if m.sum() >= 5 else None)
        row['n non-English correct'] = int(
            ((p == yte) & (test['language_type'].values != 'English')).sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    if verbose:
        n = {lg: int((test['language_type'] == lg).sum()) for lg in langs}
        print(f'\nSentiment, on the held-out test split (n = {len(test)}: '
              f'{n["English"]} English, {n["Sinhala"]} Sinhala, '
              f'{n["Singlish"]} Singlish)')
        print('  per-language columns are macro-F1 within that language\n')
        print(t.to_string(index=False))
    return t


def vader_failure_detail(verbose: bool = True) -> pd.DataFrame:
    """What VADER actually outputs on non-English text.

    The headline number understates it: a lexicon that recognises none of the
    words returns a compound of exactly zero, which maps to Neutral. So the
    failure is not noisy prediction, it is abstention that scores as a class.
    """
    df = splits.attach(data.sentiment_data())
    test = df[df.split == 'test'].reset_index(drop=True)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    comp = np.array([an.polarity_scores(str(t))['compound']
                     for t in test['review_text']])
    rows = []
    for lg in ['English', 'Sinhala', 'Singlish']:
        m = test['language_type'].values == lg
        if m.sum() < 5:
            continue
        rows.append({'Language': lg, 'Reviews': int(m.sum()),
                     'Scored exactly 0.0 by VADER': int((comp[m] == 0).sum()),
                     '% of that language': round(100*float((comp[m] == 0).mean()), 1)})
    t = pd.DataFrame(rows)
    if verbose:
        print('\nVADER returns a compound of exactly zero - no lexicon match at all:')
        print(t.to_string(index=False))
    return t


def paired_delta(y, score_a, score_b, metric=roc_auc_score, n=BOOT,
                 seed=config.SEED) -> dict:
    """Bootstrap the DIFFERENCE between two scorers on the same resampled rows.

    Returns the observed difference, its CI, and the share of resamples in which
    b beats a - a one-sided bootstrap p-value in all but name.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y); a = np.asarray(score_a); b = np.asarray(score_b)
    obs = metric(y, b) - metric(y, a)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        try:
            diffs.append(metric(y[i], b[i]) - metric(y[i], a[i]))
        except ValueError:
            continue
    d = np.array(diffs)
    return {'delta': round(float(obs), 4),
            'ci': (round(float(np.percentile(d, 2.5)), 4),
                   round(float(np.percentile(d, 97.5)), 4)),
            'p_better': round(float((d > 0).mean()), 4),
            'n_boot': len(d)}


# ------------------------------------------------------------------ detection

def detection_comparison(verbose: bool = True, min_chars: int = 0):
    """The pipeline against what a platform could do with no model at all.

    min_chars restricts both sides to reviews of at least that length. At 0 the
    comparison is dominated by very short reviews, where the label is close to a
    length rule and every approach is really measuring the same thing. At 20 the
    length baseline has little left to exploit, and what the model adds becomes
    visible. Both are reported, because reporting only the flattering one would
    be picking the stratum after seeing the answer.
    """
    train, test, feat = fake._frames()
    if min_chars:
        train = train[train['n_chars'].astype(int) >= min_chars].reset_index(drop=True)
        test = test[test['n_chars'].astype(int) >= min_chars].reset_index(drop=True)
    ytr = (train['authenticity'] == 'Suspicious').astype(int).values
    yte = (test['authenticity'] == 'Suspicious').astype(int).values
    pstats = train.assign(s=train['star_rating'].astype(float)) \
                  .groupby('product_name')['s'].mean()

    rows = []
    kept = {'y': yte}

    def add(name, score, note):
        kept[name] = np.asarray(score, dtype=float)
        auc = roc_auc_score(yte, score)
        lo, hi = _boot_ci(yte, score, roc_auc_score)
        rows.append({'Approach': name, 'ROC-AUC': round(auc, 3),
                     '95% CI': f'{lo:.3f}-{hi:.3f}', 'Needs a model': note})

    # 1. the star rating on its own - a five-star rating as the only suspicion
    add('Star rating alone (5 stars = suspicious)',
        (test['star_rating'].astype(float) >= 5).astype(float).values, 'no')

    # 2. character count on its own - the confound documented in Chapter 4
    add('Review length alone (character count)',
        -test['review_text'].astype(str).str.len().values, 'no')

    # 3-5. the pipeline, with families removed
    for label, fams in [
            ('Behavioural + lexical features only',
             ['lexical', 'behavioural']),
            ('Full feature set without CII',
             [f for f in fake.FAMILIES if f != 'cii']),
            ('Full feature set with CII (this study)', fake.FAMILIES)]:
        Xtr = fake.build_matrix(train, feat, fams, pstats)
        Xte = fake.build_matrix(test, feat, fams, pstats).reindex(columns=Xtr.columns)
        m = fake._rf().fit(Xtr, ytr)
        add(label, m.predict_proba(Xte)[:, 1], 'yes')

    t = pd.DataFrame(rows)
    if verbose:
        print(f'\nSuspicious-review detection, test split '
              f'(n = {len(test)}, {int(yte.sum())} suspicious)\n')
        print(t.to_string(index=False))
        print('\n  Length alone is the baseline to beat, not a strawman: the '
              'annotation is\n  close to a rule about length, so a model that '
              'merely matches it has added\n  nothing. See the length-stratified '
              'ablation in Chapter 4.')
    return t, kept


def detection_significance(verbose: bool = True, min_chars: int = 0) -> pd.DataFrame:
    """The two comparisons Chapter 5.9 actually turns on, tested pairwise.

    RQ3 asks whether adding CII improves detection. That is a question about a
    difference, so it is tested on the difference, not by eyeballing whether two
    separate intervals overlap.
    """
    _t, kept = detection_comparison(verbose=False, min_chars=min_chars)
    yte = kept['y']
    pairs = [
        ('Adding CII to the full feature set',
         kept['Full feature set without CII'],
         kept['Full feature set with CII (this study)']),
        ('The full model over review length alone',
         kept['Review length alone (character count)'],
         kept['Full feature set with CII (this study)']),
        ('Any model at all over the star rating',
         kept['Star rating alone (5 stars = suspicious)'],
         kept['Full feature set with CII (this study)']),
    ]
    rows = []
    for label, a, b in pairs:
        r = paired_delta(yte, a, b)
        rows.append({'Comparison': label, 'Change in AUC': f"{r['delta']:+.3f}",
                     '95% CI': f"{r['ci'][0]:+.3f} to {r['ci'][1]:+.3f}",
                     'Resamples favouring it': f"{100*r['p_better']:.1f}%",
                     'Clear at 95%?': 'yes' if r['ci'][0] > 0 else 'no'})
    npos = int(np.asarray(yte).sum())
    t = pd.DataFrame(rows)
    t.insert(0, 'Reviews', f'{min_chars}+ chars (n={len(yte)}, {npos} susp.)')
    if verbose:
        print(f'\nPaired bootstrap on the differences, reviews of {min_chars}+ '
              f'characters (2000 resamples)\n')
        print(t.to_string(index=False))
        if npos < 25:
            print(f'\n  CAUTION: only {npos} suspicious reviews in this stratum. '
                  f'An AUC estimated\n  from {npos} positives is fragile however '
                  'wide the interval looks, and the\n  interval is already wide. '
                  'Report this figure with its n, never alone.')
    return t


# ------------------------------------------------------------------- reporting

def fig_sentiment_by_language(t: pd.DataFrame):
    """Where the off-the-shelf tool actually breaks."""
    import matplotlib.pyplot as plt
    eda.setup_style()
    langs = [c for c in t.columns if c.split(' (')[0]
             in ('English', 'Sinhala', 'Singlish')]
    d = t.dropna(subset=langs)
    x = np.arange(len(langs))
    width = 0.8/len(d)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    shades = [eda.BASE, eda.ORANGE, eda.AQUA, eda.BLUE]
    for i, (_, r) in enumerate(d.iterrows()):
        ax.bar(x + i*width - 0.4 + width/2, [r[l] for l in langs], width*0.88,
               label=r['Approach'], color=shades[i % len(shades)])
    ax.set_xticks(x, [l.replace(' (', '\n(') for l in langs])
    ax.set_ylabel('Macro-F1 within the language')
    ax.set_title('Sentiment accuracy by language of the review')
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True); ax.set_axisbelow(True); ax.xaxis.grid(False)
    ax.legend(fontsize=6.6, loc='upper right', framealpha=0.95)
    return eda._save(fig, '15_sentiment_vs_baselines')


def fig_detection_strata(d_all: pd.DataFrame, d20: pd.DataFrame):
    """Why the headline number and the real finding differ.

    On every review, length already explains the label, so nothing can add much.
    On reviews with something written in them, length is worth nothing and the
    model carries the whole signal.
    """
    import matplotlib.pyplot as plt
    eda.setup_style()
    order = ['Review length alone (character count)',
             'Full feature set without CII',
             'Full feature set with CII (this study)']
    short = ['Length alone', 'Model without CII', 'Model with CII']
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for d, off, colour, lab in ((d_all, +0.19, eda.BASE, 'All reviews'),
                                (d20, -0.19, eda.BLUE, 'Reviews of 20+ characters')):
        v = [float(d.loc[d.Approach == a, 'ROC-AUC'].iloc[0]) for a in order]
        b = ax.barh(y + off, v, height=0.34, color=colour, label=lab)
        ax.bar_label(b, fmt='%.3f', fontsize=7, padding=3,
                     color=eda.INK2 if colour == eda.BASE else eda.BLUE)
    ax.axvline(0.5, color=eda.ORANGE, linewidth=1.2, linestyle='--')
    ax.text(0.49, -0.62, 'chance', fontsize=7, color=eda.ORANGE, ha='right')
    ax.set_yticks(y, short, fontsize=8)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(-0.75, len(order)-0.25)
    ax.set_xlabel('ROC-AUC on the held-out test split')
    ax.set_title('What the model adds, once length stops explaining the label')
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.yaxis.grid(False)
    ax.legend(fontsize=7, loc='lower right')
    return eda._save(fig, '16_detection_strata')


def sentiment_significance(verbose: bool = True) -> pd.DataFrame:
    """Character n-grams against word features, tested rather than asserted.

    Overall the two are a tie. The design claim is specifically about code-mixed
    text, so it is tested there - on 30 Singlish reviews, which is few enough
    that the interval is the finding.
    """
    df = splits.attach(data.sentiment_data())
    train = df[df.split == 'train'].reset_index(drop=True)
    test = df[df.split == 'test'].reset_index(drop=True)
    ytr, yte = train['sentiment_polarity'].values, test['sentiment_polarity'].values
    word = _word_tfidf_model().fit(train['review_text'], ytr).predict(test['review_text'])
    ours = sentiment.load().predict(list(test['review_text']))
    rng = np.random.default_rng(config.SEED)

    rows = []
    for label, mask in [('All test reviews', np.ones(len(test), bool)),
                        ('Singlish only', test['language_type'].values == 'Singlish'),
                        ('Sinhala only', test['language_type'].values == 'Sinhala'),
                        ('Non-English', test['language_type'].values != 'English')]:
        yt, a, b = yte[mask], word[mask], ours[mask]
        f = lambda Y, P: f1_score(Y, P, average='macro', zero_division=0)
        obs = f(yt, b) - f(yt, a)
        ds = []
        for _ in range(BOOT):
            i = rng.integers(0, len(yt), len(yt))
            if len(np.unique(yt[i])) < 2:
                continue
            ds.append(f(yt[i], b[i]) - f(yt[i], a[i]))
        ds = np.array(ds)
        rows.append({'Subset': f'{label} (n={int(mask.sum())})',
                     'Char n-gram minus word': f'{obs:+.3f}',
                     '95% CI': f'{np.percentile(ds, 2.5):+.3f} to '
                               f'{np.percentile(ds, 97.5):+.3f}',
                     'Clear at 95%?': 'yes' if np.percentile(ds, 2.5) > 0 else 'no'})
    t = pd.DataFrame(rows)
    if verbose:
        print('\nCharacter n-grams against word features, paired bootstrap\n')
        print(t.to_string(index=False))
        print('\n  Read this as a tie that the data cannot resolve, not as a win. '
              'The\n  character n-gram was chosen for the reasons in Chapter 3; '
              'this test says\n  the test set is too small to confirm the '
              'advantage, not that none exists.')
    return t


def report_all():
    s = sentiment_comparison()
    v = vader_failure_detail()
    d, _ = detection_comparison()
    d20, _ = detection_comparison(min_chars=20)
    sig = pd.concat([detection_significance(min_chars=0),
                     detection_significance(min_chars=20)], ignore_index=True)
    ssig = sentiment_significance()
    eda._save_table(s, '23_sentiment_comparison')
    eda._save_table(v, '24_vader_zero_scores')
    eda._save_table(pd.concat(
        [d.assign(Reviews='all'), d20.assign(Reviews='20+ chars')],
        ignore_index=True), '25_detection_comparison')
    eda._save_table(sig, '26_detection_significance')
    eda._save_table(ssig, '27_sentiment_significance')
    fig_sentiment_by_language(s)
    fig_detection_strata(d, d20)
    return {'sentiment': s, 'vader': v, 'detection': d, 'detection20': d20,
            'significance': sig, 'sentiment_significance': ssig}


if __name__ == '__main__':
    from trustlens import compare as _c
    _c.report_all()
