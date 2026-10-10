"""Fake review detection: feature families plus CII, into a Random Forest.

Features are grouped into named FAMILIES so the ablation in step 7 can switch
them on and off by name rather than by commenting out code.

    length        n_chars, n_words, mean word length
    readability   Flesch reading ease, Flesch-Kincaid grade, syllables per word
    lexical       type-token ratio, case, digits, punctuation, emoji, repeats
    behavioural   deviation from the product's mean rating, helpfulness, duplicates
    cii           SSG, ASCS, ISG and the composite

Families from Hajek et al. (2023) that this corpus cannot support
-----------------------------------------------------------------
VP-based behavioural - `verified_purchase` is True for all 1,977 rows. Constant,
    so it carries no information. Dropped.
Word embedding - Skip-Gram trained on 1,557 training reviews averaging 53
    characters produces noise. Dropped rather than reported as working.
POS tagging - taggers exist for English but not for Sinhala or Singlish, so the
    family would be missing for 22% of the corpus. More decisively, the median
    review is 32 characters and 708 are under 20: a part-of-speech distribution
    over five tokens is not a measurement. Dropped, with that reasoning.

`length` is kept as its own family precisely so it can be removed. The Suspicious
label is close to a length rule (AUC 0.841 on its own), so any claim about CII has
to be made against a model that already has length.

Readability is implemented here rather than pulled from a library: it needs no
network, no corpus download, and no dependency for a reader to reproduce.

Train and evaluate:  python -m trustlens.fake
"""
from __future__ import annotations
import json
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, confusion_matrix)
from . import config, data, splits, cii

MODEL_PATH = config.MODELS/'fake.joblib'
META_PATH = config.MODELS/'fake_meta.json'

FAMILIES = ['length', 'readability', 'lexical', 'behavioural', 'cii']
VOWELS = re.compile(r'[aeiouy]+')
WORD = re.compile(r"[A-Za-z඀-෿']+")


def _syllables(word: str) -> int:
    """Heuristic syllable count. Vowel groups, less a silent e, minimum one."""
    w = word.lower()
    n = len(VOWELS.findall(w))
    if w.endswith('e') and not w.endswith(('le', 'ee')) and n > 1:
        n -= 1
    return max(n, 1)


def f_length(df) -> pd.DataFrame:
    t = df['review_text'].astype(str)
    words = t.map(lambda s: WORD.findall(s))
    return pd.DataFrame({
        'len_chars': t.str.len(),
        'len_words': words.map(len),
        'len_mean_word': words.map(lambda w: np.mean([len(x) for x in w]) if w else 0.0),
    }, index=df.index)


def f_readability(df) -> pd.DataFrame:
    """English only. NaN elsewhere - the tree models handle that natively."""
    out = []
    for text, lang in zip(df['review_text'].astype(str), df['language_type']):
        words = WORD.findall(text)
        if lang != 'English' or len(words) < 3:
            out.append({'read_flesch': np.nan, 'read_fk_grade': np.nan,
                        'read_syll_per_word': np.nan})
            continue
        sents = max(len([s for s in re.split(r'[.!?]+', text) if s.strip()]), 1)
        syl = sum(_syllables(w) for w in words)
        wps, spw = len(words)/sents, syl/len(words)
        out.append({'read_flesch': 206.835 - 1.015*wps - 84.6*spw,
                    'read_fk_grade': 0.39*wps + 11.8*spw - 15.59,
                    'read_syll_per_word': spw})
    return pd.DataFrame(out, index=df.index)


def f_lexical(df) -> pd.DataFrame:
    t = df['review_text'].astype(str)
    words = t.map(lambda s: WORD.findall(s.lower()))
    n = t.str.len().clip(lower=1)
    return pd.DataFrame({
        'lex_type_token': words.map(lambda w: len(set(w))/len(w) if w else 0.0),
        'lex_upper_ratio': t.map(lambda s: sum(c.isupper() for c in s))/n,
        'lex_digit_ratio': t.map(lambda s: sum(c.isdigit() for c in s))/n,
        'lex_punct_ratio': t.map(lambda s: sum(not c.isalnum() and not c.isspace()
                                               for c in s))/n,
        'lex_emoji': t.map(lambda s: len(cii.EMOJI_RE.findall(s))),
        'lex_repeat_runs': t.map(lambda s: len(cii.ELONGATION_RE.findall(s))),
        'lex_exclaim': t.map(lambda s: len(cii.PUNCT_RE.findall(s))),
    }, index=df.index)


def f_behavioural(df, product_stats=None) -> pd.DataFrame:
    """Rating-based behaviour. Fitted stats come from TRAINING rows only."""
    star = df['star_rating'].astype(float)
    if product_stats is None:
        product_stats = df.groupby('product_name')['star_rating'].astype(float).mean()
    pmean = df['product_name'].map(product_stats)
    return pd.DataFrame({
        'beh_star': star,
        'beh_star_dev': star - pmean.fillna(star.mean()),
        'beh_helpful': df['helpful_count'].astype(float),
        'beh_dup_count': df['dup_count'].astype(float),
        'beh_is_extreme': star.isin([1, 5]).astype(int),
    }, index=df.index)


def f_cii(df, feat: pd.DataFrame) -> pd.DataFrame:
    m = feat.set_index('review_id')
    cols = ['ssg_centred', 'ascs', 'isg', 'cii']
    out = m.reindex(df['review_id'])[cols]
    out.index = df.index
    return out.rename(columns={c: f'cii_{c}' for c in cols})


def build_matrix(df, feat, families=None, product_stats=None) -> pd.DataFrame:
    families = FAMILIES if families is None else list(families)
    parts = []
    if 'length' in families:      parts.append(f_length(df))
    if 'readability' in families: parts.append(f_readability(df))
    if 'lexical' in families:     parts.append(f_lexical(df))
    if 'behavioural' in families: parts.append(f_behavioural(df, product_stats))
    if 'cii' in families:         parts.append(f_cii(df, feat))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def _models():
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight='balanced',
            random_state=config.SEED, n_jobs=-1),
        'Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, random_state=config.SEED,
            class_weight='balanced'),
        'Logistic Regression': Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=4000,
                                       random_state=config.SEED))]),
    }


def _frames():
    df = splits.attach(data.fake_data())
    feat = pd.read_csv(config.PROCESSED/'cii_features.csv')
    return df[df.split == 'train'].reset_index(drop=True), \
           df[df.split == 'test'].reset_index(drop=True), feat


def evaluate_models(families=None, verbose=True) -> dict:
    train, test, feat = _frames()
    pstats = train.assign(s=train['star_rating'].astype(float)).groupby('product_name')['s'].mean()
    Xtr = build_matrix(train, feat, families, pstats)
    Xte = build_matrix(test, feat, families, pstats)
    ytr = (train['authenticity'] == 'Suspicious').astype(int).values
    yte = (test['authenticity'] == 'Suspicious').astype(int).values

    rows, fitted = [], {}
    for name, model in _models().items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]
        p, r, f, _ = precision_recall_fscore_support(yte, pred, average='binary',
                                                     zero_division=0)
        rows.append({'Model': name, 'Precision': round(p, 3), 'Recall': round(r, 3),
                     'F1': round(f, 3), 'ROC-AUC': round(roc_auc_score(yte, proba), 3),
                     'PR-AUC': round(average_precision_score(yte, proba), 3),
                     'Accuracy': round((pred == yte).mean(), 3)})
        fitted[name] = model
    return {'table': pd.DataFrame(rows), 'models': fitted, 'Xtr': Xtr, 'Xte': Xte,
            'ytr': ytr, 'yte': yte, 'train': train, 'test': test,
            'feat': feat, 'pstats': pstats}


def importance_table(model, X, top=15) -> pd.DataFrame:
    imp = getattr(model, 'feature_importances_', None)
    if imp is None:
        return pd.DataFrame()
    t = pd.DataFrame({'Feature': X.columns, 'Importance': imp.round(4)})
    t['Family'] = [c.split('_')[0] for c in X.columns]
    t['Family'] = t['Family'].replace({'len': 'length', 'read': 'readability',
                                       'lex': 'lexical', 'beh': 'behavioural',
                                       'cii': 'cii'})
    return t.sort_values('Importance', ascending=False).head(top).reset_index(drop=True)


def family_importance(model, X) -> pd.DataFrame:
    t = importance_table(model, X, top=len(X.columns))
    g = t.groupby('Family')['Importance'].sum().sort_values(ascending=False)
    return g.reset_index().rename(columns={'Importance': 'Total importance'})


# ------------------------------------------------------------------ ablation

BASE_FAMILIES = ['length', 'readability', 'lexical', 'behavioural']

ABLATION = [
    ('Baseline: no CII',              BASE_FAMILIES, None),
    ('+ SSG only',                    BASE_FAMILIES, ['cii_ssg_centred']),
    ('+ ASCS only',                   BASE_FAMILIES, ['cii_ascs']),
    ('+ ISG only',                    BASE_FAMILIES, ['cii_isg']),
    ('+ CII composite',               BASE_FAMILIES, ['cii_cii']),
    ('+ all three components',        BASE_FAMILIES, ['cii_ssg_centred', 'cii_ascs',
                                                      'cii_isg']),
    ('Without length, no CII',        ['readability', 'lexical', 'behavioural'], None),
    ('Without length, + all three',   ['readability', 'lexical', 'behavioural'],
                                      ['cii_ssg_centred', 'cii_ascs', 'cii_isg']),
]


def _rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                  class_weight='balanced', random_state=config.SEED,
                                  n_jobs=-1)


def ablation(min_chars: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Does CII earn its place? Each row adds something to the same base model.

    Run at min_chars=0 for the whole test set and at min_chars=20 for the reviews
    where length carries no signal. The second is the one that matters: the label
    is close to a length rule, so on the full set the base model already has the
    answer and nothing can add much.
    """
    train, test, feat = _frames()
    pstats = train.assign(s=train['star_rating'].astype(float)
                          ).groupby('product_name')['s'].mean()
    keep_tr = train['n_chars'].astype(int) >= min_chars
    keep_te = test['n_chars'].astype(int) >= min_chars
    train, test = train[keep_tr], test[keep_te]
    ytr = (train['authenticity'] == 'Suspicious').astype(int).values
    yte = (test['authenticity'] == 'Suspicious').astype(int).values

    full_tr = build_matrix(train, feat, FAMILIES, pstats)
    full_te = build_matrix(test, feat, FAMILIES, pstats)

    rows, base_auc, base_f1 = [], None, None
    for label, fams, extra in ABLATION:
        cols = [c for c in build_matrix(train.head(1), feat, fams, pstats).columns]
        cols += list(extra or [])
        m = _rf().fit(full_tr[cols], ytr)
        proba = m.predict_proba(full_te[cols])[:, 1]
        pred = m.predict(full_te[cols])
        auc = roc_auc_score(yte, proba)
        _, _, f1, _ = precision_recall_fscore_support(yte, pred, average='binary',
                                                      zero_division=0)
        lo = hi = np.nan
        b = cii._boot_auc(yte, proba)
        if b:
            _, lo, hi = b
        if label == 'Baseline: no CII':
            base_auc, base_f1 = auc, f1
        rows.append({'Configuration': label, 'Features': len(cols),
                     'ROC-AUC': round(auc, 3), 'CI low': round(lo, 3),
                     'CI high': round(hi, 3), 'F1': round(f1, 3),
                     'delta AUC': round(auc - base_auc, 3),
                     'delta F1': round(f1 - base_f1, 3)})
    t = pd.DataFrame(rows)
    t.insert(0, 'Reviews', f'{min_chars}+ chars (n={len(test)}, {int(yte.sum())} susp.)')
    if verbose:
        print(t.to_string(index=False))
    return t


def fig_ablation(ab: pd.DataFrame):
    """The Chapter 5.10 figure: what CII adds, in each length regime."""
    import matplotlib.pyplot as plt
    from . import eda
    eda.setup_style()
    strata = list(dict.fromkeys(ab['Reviews']))
    order = [c for c, _, _ in ABLATION]
    fig, axes = plt.subplots(1, len(strata), figsize=(7.4, 3.4), sharey=True,
                             gridspec_kw={'wspace': 0.06})
    fig.subplots_adjust(top=0.80)
    for ax, st in zip(np.atleast_1d(axes), strata):
        d = ab[ab['Reviews'] == st].set_index('Configuration').reindex(order)
        y = np.arange(len(order))[::-1]
        base = d.loc['Baseline: no CII', 'ROC-AUC']
        for yy, (name, r) in zip(y, d.iterrows()):
            is_base = name == 'Baseline: no CII'
            colour = eda.MUTED if is_base else (
                eda.BLUE if r['ROC-AUC'] >= base else eda.ORANGE)
            ax.plot([r['CI low'], r['CI high']], [yy, yy], color=colour, linewidth=1.6,
                    solid_capstyle='round', zorder=2)
            ax.plot(r['ROC-AUC'], yy, 'o', color=colour, markersize=7,
                    markeredgecolor=eda.SURFACE, markeredgewidth=1.5, zorder=3)
            ax.text(r['CI high'] + 0.015, yy, f"{r['ROC-AUC']:.2f}", va='center',
                    fontsize=7, color=eda.INK2)
        ax.axvline(base, color=eda.BASE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.axvline(0.5, color=eda.GRID, linewidth=1.0, zorder=0)
        ax.set_yticks(y, order, fontsize=7.5)
        ax.set_xlim(0.35, 1.06); ax.set_xlabel('ROC-AUC')
        ax.set_title(st.replace(' chars', ' characters'), fontsize=8, pad=6)
        ax.xaxis.grid(True); ax.set_axisbelow(True); ax.yaxis.grid(False)
    fig.suptitle('What the index adds to fake-review detection', x=0.055, y=1.03,
                 ha='left', fontsize=9.5, fontweight='bold')
    return eda._save(fig, '13_ablation')


def train_final():
    """Fit the primary detector on every family and save it for the Flask app."""
    res = evaluate_models()
    model = res['models']['Random Forest']
    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'columns': list(res['Xtr'].columns),
                 'product_stats': res['pstats'].to_dict()}, MODEL_PATH)
    return res


def load():
    return config.load_model(MODEL_PATH, 'python -m trustlens.fake')


def report_all():
    from . import eda
    print('=== model comparison, all families ===')
    res = train_final()
    print(res['table'].to_string(index=False))

    rf, Xte = res['models']['Random Forest'], res['Xte']
    fam = family_importance(rf, Xte)
    print('\n=== importance by family ===')
    print(fam.to_string(index=False))
    top = importance_table(rf, Xte)
    print('\n=== top features ===')
    print(top.to_string(index=False))

    print('\n=== ABLATION: whole test set ===')
    ab0 = ablation(0)
    print('\n=== ABLATION: reviews of 20+ characters ===')
    ab20 = ablation(20)
    ab = pd.concat([ab0, ab20], ignore_index=True)

    print('\ntables ->', config.TABLES)
    eda._save_table(res['table'], '18_fake_model_comparison')
    eda._save_table(fam, '19_fake_family_importance')
    eda._save_table(top, '20_fake_top_features')
    eda._save_table(ab, '21_ablation')
    print('figures ->', config.FIGURES)
    fig_ablation(ab)

    meta = {'families': FAMILIES, 'n_features': Xte.shape[1],
            'dropped_families': {
                'VP-based behavioural': 'verified_purchase is constant True',
                'word embedding': 'Skip-Gram needs far more than 1,557 short reviews',
                'POS tagging': 'no tagger for Sinhala or Singlish; median review is '
                               '32 characters, too short for a POS distribution'},
            'test_metrics': res['table'].to_dict('records')}
    META_PATH.write_text(json.dumps(meta, indent=2))
    res.update({'ablation': ab, 'family_importance': fam, 'top_features': top})
    return res


if __name__ == '__main__':
    from trustlens import fake as _fake
    _fake.report_all()
