"""Aspect-based sentiment: which aspects a review mentions, and how it feels about each.

Two stages, seven aspects:

  1. Detection - a binary classifier per aspect: is this aspect mentioned?
  2. Polarity  - given it is mentioned, Positive / Neutral / Negative.

One TF-IDF vectorizer is shared by all fourteen heads, using the configuration the
sentiment grid search selected in step 3. That is what the architecture means by a
shared vectorizer: fit the text representation once, hang many small models off it.
The Flask app then loads one vectorizer, not fifteen.

Why this runs BEFORE the index is built: ASCS needs to know which aspects a review
mentions. Reading them from the annotation would use labels that do not exist at
deployment and would inflate every number downstream, so ASCS consumes the
predictions from here.

Train and evaluate:  python -m trustlens.absa
Use:                 from trustlens import absa
                     m = absa.load()
                     m.predict(['fast delivery but poor quality'])
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import f1_score, precision_recall_fscore_support
from . import config, data, splits

MODEL_PATH = config.MODELS/'absa.joblib'
META_PATH = config.MODELS/'absa_meta.json'

# The representation chosen by the step-3 grid search, reused here.
VECTORIZER = dict(analyzer='char_wb', ngram_range=(2, 4), min_df=3,
                  sublinear_tf=True, lowercase=True)
C_GRID = {'C': [0.1, 1.0, 10.0]}
MIN_CLASS = 5        # a polarity class needs this many training examples to be kept
MIN_TEST_SUPPORT = 30    # below this, test metrics are reported as indicative only


class AspectModel:
    """Shared vectorizer plus per-aspect detection and polarity heads."""

    def __init__(self, vectorizer, detectors, polarity, meta):
        self.vectorizer = vectorizer
        self.detectors = detectors        # aspect -> fitted classifier
        self.polarity = polarity          # aspect -> fitted classifier, or None
        self.meta = meta

    def predict(self, texts) -> list[dict[str, str]]:
        """[{aspect: polarity, ...}, ...] - one dict per text. What ASCS consumes."""
        X = self.vectorizer.transform(list(texts))
        out = [{} for _ in range(X.shape[0])]
        for aspect, det in self.detectors.items():
            present = det.predict(X).astype(bool)
            if not present.any():
                continue
            pol = self.polarity.get(aspect)
            if pol is None:
                continue            # no polarity head: omit, never guess a label
            labels = pol.predict(X[present])
            for i, lab in zip(np.flatnonzero(present), labels):
                out[i][aspect] = str(lab)
        return out

    def predict_frame(self, texts) -> pd.DataFrame:
        """One column per aspect, '' where not mentioned."""
        preds = self.predict(texts)
        return pd.DataFrame([{a: p.get(a, '') for a in config.ASPECTS} for p in preds])

    def to_strings(self, texts) -> list[str]:
        """Same format as the `aspects_merged` column: 'Aspect:Polarity; ...'."""
        return ['; '.join(f'{a}:{p}' for a, p in d.items()) for d in self.predict(texts)]


def _frames():
    df = splits.attach(data.absa_data())
    return df[df.split == 'train'].reset_index(drop=True), \
           df[df.split == 'test'].reset_index(drop=True)


def train_model(verbose: bool = True) -> dict:
    train, test = _frames()
    Mtr, Mte = data.aspect_matrix(train), data.aspect_matrix(test)

    vec = TfidfVectorizer(**VECTORIZER).fit(train['review_text'].values)
    Xtr = vec.transform(train['review_text'].values)
    Xte = vec.transform(test['review_text'].values)
    cv = PredefinedSplit(train['fold'].astype(int).values)
    if verbose:
        print(f'train {len(train)} | test {len(test)} | vocabulary {len(vec.vocabulary_):,}')

    detectors, polarity, rows_d, rows_p = {}, {}, [], []
    for aspect in config.ASPECTS:
        # ---- detection
        y = (Mtr[aspect].values != '').astype(int)
        gs = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=4000,
                                             random_state=config.SEED),
                          C_GRID, scoring='f1', cv=cv, n_jobs=-1)
        gs.fit(Xtr, y)
        detectors[aspect] = gs.best_estimator_
        yte = (Mte[aspect].values != '').astype(int)
        p, r, f, _ = precision_recall_fscore_support(yte, gs.predict(Xte),
                                                     average='binary', zero_division=0)
        rows_d.append({'Aspect': aspect, 'Train mentions': int(y.sum()),
                       'Test mentions': int(yte.sum()), 'C': gs.best_params_['C'],
                       'CV F1': round(gs.best_score_, 3), 'Precision': round(p, 3),
                       'Recall': round(r, 3), 'F1': round(f, 3),
                       'Reliable': 'yes' if yte.sum() >= MIN_TEST_SUPPORT else 'indicative'})

        # ---- polarity, on the reviews that mention this aspect
        mask = Mtr[aspect].values != ''
        ypol = Mtr[aspect].values[mask]
        keep = [c for c, n in pd.Series(ypol).value_counts().items() if n >= MIN_CLASS]
        sub = np.isin(ypol, keep)
        note = ''
        if len(keep) < 2 or sub.sum() < 20:
            polarity[aspect] = None
            note = 'not trained - too few examples'
            rows_p.append({'Aspect': aspect, 'Train examples': int(mask.sum()),
                           'Classes': ', '.join(keep) or '-', 'CV macro F1': None,
                           'Test macro F1': None, 'Test examples': int((Mte[aspect] != '').sum()),
                           'Note': note})
            continue
        if len(keep) < 3:
            note = f"Neutral dropped ({MIN_CLASS} examples minimum)"
        gsp = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=4000,
                                              random_state=config.SEED),
                           C_GRID, scoring='f1_macro',
                           cv=PredefinedSplit(train['fold'].astype(int).values[mask][sub]),
                           n_jobs=-1)
        gsp.fit(Xtr[mask][sub], ypol[sub])
        polarity[aspect] = gsp.best_estimator_

        mte = Mte[aspect].values != ''
        yte_pol = Mte[aspect].values[mte]
        keep_te = np.isin(yte_pol, keep)
        n_classes_seen = len(set(yte_pol[keep_te])) if keep_te.sum() else 0
        # A macro F1 over one observed class is not a score. Packaging's test set
        # holds only Positive examples, which would otherwise read as F1 = 1.000.
        test_f1 = (round(f1_score(yte_pol[keep_te], gsp.predict(Xte[mte][keep_te]),
                                  average='macro', zero_division=0), 3)
                   if n_classes_seen >= 2 else None)
        rows_p.append({'Aspect': aspect, 'Train examples': int(sub.sum()),
                       'Classes': ', '.join(keep), 'CV macro F1': round(gsp.best_score_, 3),
                       'Test macro F1': test_f1, 'Test examples': int(keep_te.sum()),
                       'Note': '; '.join(filter(None, [
                           note,
                           'only one class present in test - not measurable'
                           if n_classes_seen < 2 else
                           'indicative - small test set'
                           if keep_te.sum() < MIN_TEST_SUPPORT else '']))})

    det_tbl = pd.DataFrame(rows_d).sort_values('Train mentions', ascending=False)
    pol_tbl = pd.DataFrame(rows_p).sort_values('Train examples', ascending=False)
    model = AspectModel(vec, detectors, polarity,
                        {'vectorizer': {k: (list(v) if isinstance(v, tuple) else v)
                                        for k, v in VECTORIZER.items()},
                         'n_train': len(train), 'n_test': len(test),
                         'vocabulary': len(vec.vocabulary_),
                         'detection_macro_f1': round(det_tbl['F1'].mean(), 4),
                         'polarity_macro_f1': round(pol_tbl['Test macro F1'].dropna().mean(), 4)})
    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps(model.meta, indent=2))

    if verbose:
        print('\n=== DETECTION ===');  print(det_tbl.to_string(index=False))
        print('\n=== POLARITY ===');   print(pol_tbl.to_string(index=False))
        print(f'\nsaved {MODEL_PATH.name}')
    return {'model': model, 'detection': det_tbl, 'polarity': pol_tbl,
            'train': train, 'test': test}


def polarity_detail(model, test) -> pd.DataFrame:
    """Per-aspect, per-class polarity performance with support beside every figure.

    A macro F1 over three classes is misleading when one of them has five
    examples, and a perfect score on eight test cases is noise, not achievement.
    Report this table rather than the macro column alone.
    """
    from sklearn.metrics import classification_report
    Mte = data.aspect_matrix(test)
    X = model.vectorizer.transform(test['review_text'].values)
    rows = []
    for aspect, pol in model.polarity.items():
        if pol is None:
            continue
        mask = Mte[aspect].values != ''
        y = Mte[aspect].values[mask]
        keep = np.isin(y, list(pol.classes_))
        if keep.sum() == 0:
            continue
        y, pred = y[keep], pol.predict(X[mask][keep])
        rep = classification_report(y, pred, output_dict=True, zero_division=0)
        for c in pol.classes_:
            if c in rep:
                rows.append({'Aspect': aspect, 'Polarity': c,
                             'Precision': round(rep[c]['precision'], 3),
                             'Recall': round(rep[c]['recall'], 3),
                             'F1': round(rep[c]['f1-score'], 3),
                             'Test support': int(rep[c]['support']),
                             'Trust': 'yes' if rep[c]['support'] >= 20 else 'too few to judge'})
    return pd.DataFrame(rows)


def fig_aspect_performance(detection, polarity):
    """Detection F1 per aspect, with test support and a reliability marker."""
    import matplotlib.pyplot as plt
    from . import eda
    eda.setup_style()
    d = detection.sort_values('Test mentions')
    pol = polarity.set_index('Aspect')['Test macro F1'].to_dict()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True,
                                   gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.08})
    fig.subplots_adjust(top=0.80)
    y = np.arange(len(d))

    colors = [eda.BLUE if r == 'yes' else eda.MUTED for r in d['Reliable']]
    b = ax1.barh(y, d['F1'], color=colors, height=0.6)
    for bar, f in zip(b, d['F1']):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{f:.2f}', va='center', fontsize=7, color=eda.INK2)
    ax1.set_yticks(y, [f"{r.Aspect}\n(n={r._3})" for r in d.itertuples()], fontsize=7)
    ax1.set_xlim(0, 1.15); ax1.set_title('Detection F1', fontsize=8, pad=6)
    ax1.xaxis.grid(True); ax1.set_axisbelow(True); ax1.yaxis.grid(False)

    vals = [pol.get(a) for a in d['Aspect']]
    for i, (a, v) in enumerate(zip(d['Aspect'], vals)):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            ax2.text(0.04, i, 'not measurable', va='center', fontsize=7, color=eda.MUTED)
        else:
            ax2.barh(i, v, color=eda.ORANGE, height=0.6)
            ax2.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=7, color=eda.INK2)
    ax2.set_xlim(0, 1.15); ax2.set_title('Polarity macro F1', fontsize=8, pad=6)
    ax2.set_xlabel('Score')
    ax2.xaxis.grid(True); ax2.set_axisbelow(True); ax2.yaxis.grid(False)

    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(facecolor=eda.BLUE, label='30+ test mentions'),
                        Patch(facecolor=eda.MUTED, label='fewer - indicative')],
               loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7)
    fig.suptitle('Aspect model performance by aspect', x=0.055, y=1.04, ha='left',
                 fontsize=9.5, fontweight='bold')
    return eda._save(fig, '10_absa_performance')


def load() -> AspectModel:
    return config.load_model(MODEL_PATH, 'python -m trustlens.absa')


def report_all():
    from . import eda
    res = train_model()
    print('\ntables ->', config.TABLES)
    eda._save_table(res['detection'], '11_absa_detection')
    eda._save_table(res['polarity'], '12_absa_polarity')
    eda._save_table(polarity_detail(res['model'], res['test']), '13_absa_polarity_detail')
    print('figures ->', config.FIGURES)
    fig_aspect_performance(res['detection'], res['polarity'])
    return res


if __name__ == '__main__':
    # Import through the package name before training. Run as `python -m
    # trustlens.absa`, this module is also called __main__, and AspectModel would
    # be pickled as __main__.AspectModel - unloadable from anywhere else.
    from trustlens import absa as _absa
    _absa.report_all()
