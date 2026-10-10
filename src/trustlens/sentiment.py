"""Whole-review sentiment: character n-gram TF-IDF + calibrated Logistic Regression.

This model exists for two jobs:

1. Predict Positive / Neutral / Negative (thesis 4.6, 5.6).
2. Produce the continuous score SSG needs: P(positive) - P(negative), on [-1, 1].

Job 2 is why calibration matters. Logistic Regression on sparse TF-IDF tends to be
overconfident, which pushes that score toward the extremes and makes SSG bimodal
instead of continuous. SSG is one third of CII, so a miscalibrated probability
would bias the whole index.

Character n-grams rather than words: one vectorizer then covers Sinhala script,
Singlish transliteration and English without a tokeniser per language.

Train and evaluate:   python -m trustlens.sentiment
Use:                  from trustlens import sentiment
                      m = sentiment.load()
                      m.predict(['good product'])
                      sentiment.score(m, ['good product'])     # -> [-1, 1]
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             brier_score_loss)
from . import config, data, splits

MODEL_PATH = config.MODELS/'sentiment.joblib'
META_PATH = config.MODELS/'sentiment_meta.json'

GRID = {
    'tfidf__ngram_range':  [(2, 4), (2, 5), (3, 5)],
    'tfidf__min_df':       [1, 2, 3],
    'tfidf__sublinear_tf': [True, False],
    'clf__C':              [0.1, 1.0, 10.0],
}


def _pipeline() -> Pipeline:
    return Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', lowercase=True)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=4000,
                                   random_state=config.SEED)),
    ])


def _frames():
    """Training and test frames, using the project's fixed split."""
    df = splits.attach(data.sentiment_data())
    return df[df['split'] == 'train'].reset_index(drop=True), \
           df[df['split'] == 'test'].reset_index(drop=True)


def _predefined(train: pd.DataFrame) -> PredefinedSplit:
    """Grid search must use the project's own folds, not a fresh random CV."""
    return PredefinedSplit(train['fold'].astype(int).values)


def score(model, texts) -> np.ndarray:
    """Continuous sentiment on [-1, 1]: P(positive) - P(negative). Feeds SSG."""
    proba = model.predict_proba(list(texts))
    cls = list(model.classes_)
    return proba[:, cls.index('Positive')] - proba[:, cls.index('Negative')]


def _ece(y_true_bin: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error - mean gap between confidence and accuracy."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum():
            total += m.mean()*abs(y_true_bin[m].mean() - p[m].mean())
    return float(total)


def _calibration_report(train: pd.DataFrame, best_params: dict) -> dict:
    """Out-of-fold comparison of raw vs calibrated probabilities.

    Measured on the TRAINING folds only. The test set is not consulted for this
    decision - it is looked at once, at the end.
    """
    y = train['sentiment_polarity'].values
    X = train['review_text'].values
    folds = train['fold'].astype(int).values
    out = {}
    for name in ['raw', 'calibrated']:
        p_pos = np.zeros(len(train))
        for f in sorted(set(folds)):
            tr, va = folds != f, folds == f
            est = _pipeline().set_params(**best_params)
            if name == 'calibrated':
                est = CalibratedClassifierCV(est, method='sigmoid', cv=3)
            est.fit(X[tr], y[tr])
            cls = list(est.classes_)
            p_pos[va] = est.predict_proba(X[va])[:, cls.index('Positive')]
        ybin = (y == 'Positive').astype(int)
        out[name] = {'ece': round(_ece(ybin, p_pos), 4),
                     'brier': round(brier_score_loss(ybin, p_pos), 4)}
    out['chosen'] = 'calibrated' if out['calibrated']['ece'] < out['raw']['ece'] else 'raw'
    return out


def train_model(verbose: bool = True) -> dict:
    train, test = _frames()
    if verbose:
        print(f'train {len(train)} | test {len(test)}')
        print('classes:', train['sentiment_polarity'].value_counts().to_dict())

    gs = GridSearchCV(_pipeline(), GRID, scoring='f1_macro',
                      cv=_predefined(train), n_jobs=-1, refit=True, verbose=0)
    gs.fit(train['review_text'].values, train['sentiment_polarity'].values)
    best = {k: v for k, v in gs.best_params_.items()}
    if verbose:
        print(f'\nbest CV macro-F1 {gs.best_score_:.4f}')
        for k, v in best.items():
            print(f'  {k} = {v}')

    cal = _calibration_report(train, best)
    if verbose:
        print(f"\ncalibration (out-of-fold): raw ECE {cal['raw']['ece']}, "
              f"calibrated ECE {cal['calibrated']['ece']} -> using {cal['chosen']}")

    final = _pipeline().set_params(**best)
    if cal['chosen'] == 'calibrated':
        final = CalibratedClassifierCV(final, method='sigmoid', cv=5)
    final.fit(train['review_text'].values, train['sentiment_polarity'].values)

    y_true = test['sentiment_polarity'].values
    y_pred = final.predict(test['review_text'].values)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    meta = {'best_params': {k: list(v) if isinstance(v, tuple) else v
                            for k, v in best.items()},
            'cv_macro_f1': round(gs.best_score_, 4),
            'calibration': cal,
            'test_macro_f1': round(f1_score(y_true, y_pred, average='macro'), 4),
            'test_accuracy': round(rep['accuracy'], 4),
            'n_train': len(train), 'n_test': len(test),
            'grid_size': int(len(gs.cv_results_['params']))}
    META_PATH.write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"\ntest macro-F1 {meta['test_macro_f1']}  accuracy {meta['test_accuracy']}")
        print(f'saved {MODEL_PATH.name}')
    return {'model': final, 'meta': meta, 'gs': gs,
            'train': train, 'test': test, 'y_pred': y_pred}


def load():
    return config.load_model(MODEL_PATH, 'python -m trustlens.sentiment')


# ------------------------------------------------------------------ reporting

def per_class_table(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = [{'Class': c, 'Precision': round(rep[c]['precision'], 3),
             'Recall': round(rep[c]['recall'], 3), 'F1': round(rep[c]['f1-score'], 3),
             'Support': int(rep[c]['support'])}
            for c in config.SENTIMENTS if c in rep]
    rows.append({'Class': 'Macro average', 'Precision': round(rep['macro avg']['precision'], 3),
                 'Recall': round(rep['macro avg']['recall'], 3),
                 'F1': round(rep['macro avg']['f1-score'], 3),
                 'Support': int(rep['macro avg']['support'])})
    return pd.DataFrame(rows)


def per_language_table(test: pd.DataFrame, y_pred) -> pd.DataFrame:
    t = test.assign(pred=y_pred)
    rows = []
    for lang, g in t.groupby('language_type'):
        rows.append({'Language': lang, 'Reviews': len(g),
                     'Accuracy': round((g['sentiment_polarity'] == g['pred']).mean(), 3),
                     'Macro F1': round(f1_score(g['sentiment_polarity'], g['pred'],
                                                average='macro', zero_division=0), 3)})
    return pd.DataFrame(rows).sort_values('Reviews', ascending=False)




# ------------------------------------------------------------------ figures

def fig_confusion(y_true, y_pred):
    import matplotlib.pyplot as plt
    from . import eda
    eda.setup_style()
    cm = confusion_matrix(y_true, y_pred, labels=config.SENTIMENTS)
    pct = cm/cm.sum(axis=1, keepdims=True)*100
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
    for i in range(len(config.SENTIMENTS)):
        for j in range(len(config.SENTIMENTS)):
            ax.text(j, i, f'{cm[i, j]}\n{pct[i, j]:.0f}%', ha='center', va='center',
                    fontsize=7.5, color='white' if pct[i, j] > 55 else eda.INK)
    ax.set_xticks(range(3), config.SENTIMENTS, fontsize=7.5)
    ax.set_yticks(range(3), config.SENTIMENTS, fontsize=7.5)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Annotated')
    ax.set_title('Sentiment confusion matrix (test set)')
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return eda._save(fig, '07_sentiment_confusion')


def fig_per_language(test, y_pred):
    import matplotlib.pyplot as plt
    from . import eda
    eda.setup_style()
    t = per_language_table(test, y_pred)
    t = t[t['Reviews'] >= 10].sort_values('Reviews')
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    y = np.arange(len(t)); h = 0.34
    b1 = ax.barh(y + h/2, t['Accuracy'], height=h, color=eda.BLUE, label='Accuracy')
    b2 = ax.barh(y - h/2, t['Macro F1'], height=h, color=eda.ORANGE, label='Macro F1')
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.2f}', va='center', fontsize=7, color=eda.INK2)
    ax.set_yticks(y, [f"{r.Language}\n(n={r.Reviews})" for r in t.itertuples()], fontsize=7.5)
    ax.set_xlim(0, 1.12); ax.set_xlabel('Score')
    ax.set_title('Sentiment performance by language (test set)')
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.yaxis.grid(False)
    ax.legend(loc='lower right')
    return eda._save(fig, '08_sentiment_by_language')


def fig_score_by_star(df, scores):
    """The figure that previews SSG: predicted sentiment against star rating."""
    import matplotlib.pyplot as plt
    from . import eda
    eda.setup_style()
    t = df.assign(s=scores)
    stars = sorted(t['star_rating'].astype(int).unique())
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    bp = ax.boxplot([t[t['star_rating'].astype(int) == s]['s'].values for s in stars],
                    positions=stars, widths=0.55, patch_artist=True,
                    medianprops=dict(color=eda.INK, linewidth=1.4),
                    flierprops=dict(marker='o', markersize=2.5, markerfacecolor=eda.MUTED,
                                    markeredgecolor='none', alpha=0.5))
    for patch in bp['boxes']:
        patch.set(facecolor=eda.BLUE, alpha=0.30, edgecolor=eda.BLUE, linewidth=1.2)
    for part in ('whiskers', 'caps'):
        for ln in bp[part]:
            ln.set(color=eda.BASE, linewidth=1.0)
    ax.plot(stars, [(s - 3)/2 for s in stars], color=eda.ORANGE, linewidth=1.8,
            marker='o', markersize=4, label='Star-implied value, (r−3)/2')
    ax.axhline(0, color=eda.BASE, linewidth=0.8)
    ax.set_xlabel('Star rating'); ax.set_ylabel('Predicted sentiment,  P(pos) − P(neg)')
    ax.set_title('Predicted sentiment against star rating')
    ax.set_ylim(-1.15, 1.35); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc='upper left')          # lower right sits on the 5-star outliers
    return eda._save(fig, '09_score_vs_star')


def table_grid(gs, top: int = 10) -> pd.DataFrame:
    r = pd.DataFrame(gs.cv_results_)
    keep = [c for c in r.columns if c.startswith('param_')]
    out = r[keep + ['mean_test_score', 'std_test_score']].copy()
    out.columns = [c.replace('param_', '').replace('tfidf__', '').replace('clf__', '')
                   for c in keep] + ['CV macro F1', 'SD']
    out['CV macro F1'] = out['CV macro F1'].round(4)
    out['SD'] = out['SD'].round(4)
    return out.sort_values('CV macro F1', ascending=False).head(top).reset_index(drop=True)


def table_score_vs_star(train, scores) -> pd.DataFrame:
    t = train.assign(s=scores)
    g = t.groupby('star_rating')['s'].agg(Reviews='count', Mean='mean', SD='std').round(3)
    g['Star-implied'] = [(int(i) - 3)/2 for i in g.index]
    g['Offset'] = (g['Mean'] - g['Star-implied']).round(3)
    return g.reset_index()


PROBES = ['good product', 'fast delivery', 'ok', 'average', 'not working',
          'do not buy', 'very bad quality', 'broke after two days',
          'this is terrible', 'worst product', 'stopped working']


def table_probes(model, train) -> pd.DataFrame:
    """Qualitative probes, with how often each key word occurs in training.

    Shows concretely where the model is strong and where it is not. Common
    negative phrasing is handled well; rare words are not, because a word seen
    three times cannot be learned. That is a corpus-size limit, not a modelling
    fault, and it is the clearest argument for the pretrained-embedding
    benchmark - a pretrained model knows "worst" without having seen it here.
    """
    low = train['review_text'].str.lower()
    cls = list(model.classes_)
    proba = model.predict_proba(PROBES)
    rows = []
    for t, p, pr, sc in zip(PROBES, model.predict(PROBES), proba, score(model, PROBES)):
        key = max(t.split(), key=len).strip('.,')
        rows.append({'Probe': t, 'Predicted': p, 'Score': round(float(sc), 3),
                     'P(neg)': round(pr[cls.index('Negative')], 2),
                     'Key word': key,
                     'Times in training': int(low.str.contains(key, regex=False).sum())})
    return pd.DataFrame(rows)


def report_all():
    """Train, evaluate, and write every figure and table for the thesis."""
    from . import eda
    res = train_model()
    model, test, y_pred = res['model'], res['test'], res['y_pred']
    y_true = test['sentiment_polarity'].values
    print('\nfigures ->', config.FIGURES)
    fig_confusion(y_true, y_pred)
    fig_per_language(test, y_pred)
    fig_score_by_star(res['train'], score(model, res['train']['review_text'].values))
    print('tables ->', config.TABLES)
    eda._save_table(per_class_table(y_true, y_pred), '06_sentiment_per_class')
    eda._save_table(per_language_table(test, y_pred), '07_sentiment_per_language')
    eda._save_table(table_grid(res['gs']), '08_sentiment_grid_search')
    eda._save_table(table_score_vs_star(res['train'],
                    score(model, res['train']['review_text'].values)), '09_score_vs_star')
    eda._save_table(table_probes(model, res['train']), '10_sentiment_probes')
    return res


if __name__ == '__main__':
    report_all()
