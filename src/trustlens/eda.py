"""Corpus analysis: the figures and tables for thesis section 4.7.

Everything is written to outputs/figures (300 DPI PNG) and outputs/tables (CSV).
Regenerate the lot with:

    python -m trustlens.eda
"""
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from . import config, data, splits

# Validated palette. Categorical hues are assigned in fixed order, never cycled.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
# Sentiment is polarity, so it takes the diverging pair with a neutral midpoint.
POS, NEU, NEG = '#2a78d6', '#c3c2b7', '#e34948'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, BASE, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'

SENTIMENT_COLORS = {'Positive': POS, 'Neutral': NEU, 'Negative': NEG}
LENGTH_BANDS = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 10**6)]
BAND_LABELS = ['<10', '10–19', '20–39', '40–79', '80+']


def setup_style():
    plt.rcParams.update({
        'figure.dpi': 300, 'savefig.dpi': 300,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 10,
        'text.color': INK, 'axes.labelcolor': INK2,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'axes.edgecolor': BASE, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': GRID, 'grid.linewidth': 0.6,
        'legend.frameon': False, 'legend.fontsize': 7.5,
    })


def _save(fig, name):
    path = config.FIGURES/f'{name}.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  {path.name}')
    return path


def _bar_labels(ax, bars, fmt='{:,.0f}', horizontal=False):
    """Direct labels on every bar. Offset is a fraction of the axis range, so
    it stays visually constant whatever the data scale."""
    vals = [b.get_width() if horizontal else b.get_height() for b in bars]
    pad = max(vals) * 0.018 if vals else 0
    for b, v in zip(bars, vals):
        if horizontal:
            ax.text(v + pad, b.get_y() + b.get_height()/2, fmt.format(v),
                    va='center', ha='left', fontsize=7, color=INK2)
        else:
            ax.text(b.get_x() + b.get_width()/2, v + pad, fmt.format(v),
                    ha='center', va='bottom', fontsize=7, color=INK2)


# ------------------------------------------------------------------ figures

def fig_language(df):
    s = df['language_type'].replace('', 'Unlabelled').value_counts()
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    bars = ax.barh(s.index[::-1], s.values[::-1], color=BLUE, height=0.62)
    _bar_labels(ax, bars, horizontal=True)
    ax.set_title('Reviews by language')
    ax.set_xlabel('Reviews')
    ax.set_xlim(0, s.max()*1.12)
    ax.xaxis.grid(True); ax.set_axisbelow(True); ax.yaxis.grid(False)
    return _save(fig, '01_language_distribution')


def fig_stars(df):
    s = df['star_rating'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    bars = ax.bar([str(i) for i in s.index], s.values, color=BLUE, width=0.62)
    _bar_labels(ax, bars)
    ax.set_title('Reviews by star rating')
    ax.set_xlabel('Stars'); ax.set_ylabel('Reviews')
    ax.set_ylim(0, s.max()*1.14)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    return _save(fig, '02_star_distribution')


def fig_sentiment_by_star(df):
    d = df[df['sentiment_polarity'] != '']
    ct = pd.crosstab(d['star_rating'], d['sentiment_polarity'])
    ct = ct.reindex(columns=['Negative', 'Neutral', 'Positive'], fill_value=0)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    left = np.zeros(len(pct))
    for col in ['Negative', 'Neutral', 'Positive']:
        v = pct[col].values
        ax.barh([str(i) for i in pct.index], v, left=left,
                color=SENTIMENT_COLORS[col], height=0.62, label=col,
                edgecolor=SURFACE, linewidth=1.4)          # 2px surface gap
        for i, (val, l) in enumerate(zip(v, left)):
            if val >= 9:
                ax.text(l + val/2, i, f'{val:.0f}%', ha='center', va='center',
                        fontsize=7, color=INK if col == 'Neutral' else 'white')
        left += v
    ax.set_title('Annotated sentiment within each star rating')
    ax.set_xlabel('Share of reviews (%)'); ax.set_ylabel('Stars')
    ax.set_xlim(0, 100); ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3)
    return _save(fig, '03_sentiment_by_star')


def fig_length(df):
    L = df['n_chars'].astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.hist(L.clip(upper=300), bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
    med = int(L.median())
    ax.axvline(med, color=ORANGE, linewidth=1.6)
    ax.text(med + 6, ax.get_ylim()[1]*0.88, f'median {med} chars',
            fontsize=7.5, color=ORANGE)
    ax.set_title('Review length')
    ax.set_xlabel('Characters (clipped at 300)'); ax.set_ylabel('Reviews')
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    return _save(fig, '04_review_length')


def fig_authenticity_by_length(df):
    """The limitation figure. Authenticity is close to a length threshold."""
    d = df[df['authenticity'] != ''].copy()
    d['band'] = pd.cut(d['n_chars'].astype(int),
                       bins=[b[0] for b in LENGTH_BANDS] + [10**6],
                       labels=BAND_LABELS, right=False)
    g = d.groupby('band', observed=True)['authenticity'].agg(
        pct=lambda s: 100*(s == 'Suspicious').mean(), n='size')

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    labels = [f"{b}\n(n={int(n)})" for b, n in zip(g.index.astype(str), g['n'])]
    bars = ax.bar(labels, g['pct'], color=BLUE, width=0.6)
    for b, v in zip(bars, g['pct']):
        ax.text(b.get_x() + b.get_width()/2, v + 2.5, f'{v:.1f}%',
                ha='center', fontsize=7.5, color=INK2)
    ax.set_title('Share labelled Suspicious, by review length')
    ax.set_xlabel('Review length (characters)')
    ax.set_ylabel('Suspicious (%)')
    ax.set_ylim(0, 105); ax.yaxis.grid(True); ax.set_axisbelow(True)
    return _save(fig, '05_authenticity_by_length')


def fig_aspects(df):
    """Two panels: how often each aspect is mentioned, and how it splits.

    One panel cannot do both - Product Quality is 1,107 mentions against
    Packaging's 67, so on a shared count axis the small aspects' polarity mix
    is invisible.
    """
    a = data.aspect_frame(df)
    a = a[a['polarity'].isin(['Positive', 'Neutral', 'Negative'])]
    ct = pd.crosstab(a['aspect'], a['polarity'])
    ct = ct.reindex(columns=['Negative', 'Neutral', 'Positive'], fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    totals = ct.sum(axis=1)
    pct = ct.div(totals, axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True,
                                   gridspec_kw={'width_ratios': [1, 1.15], 'wspace': 0.08})
    fig.subplots_adjust(top=0.80)

    bars = ax1.barh(ct.index, totals.values, color=BLUE, height=0.62)
    _bar_labels(ax1, bars, horizontal=True)
    ax1.set_title('Mentions', fontsize=8, pad=6)
    ax1.set_xlim(0, totals.max()*1.22)
    ax1.xaxis.grid(True); ax1.set_axisbelow(True); ax1.yaxis.grid(False)

    left = np.zeros(len(pct))
    for col in ['Negative', 'Neutral', 'Positive']:
        v = pct[col].values
        ax2.barh(pct.index, v, left=left, color=SENTIMENT_COLORS[col], height=0.62,
                 label=col, edgecolor=SURFACE, linewidth=1.4)
        for i, (val, l) in enumerate(zip(v, left)):
            if val >= 11:
                ax2.text(l + val/2, i, f'{val:.0f}%', ha='center', va='center',
                         fontsize=6.8, color=INK if col == 'Neutral' else 'white')
        left += v
    ax2.set_title('Polarity mix', fontsize=8, pad=6)
    ax2.set_xlabel('Share of mentions (%)')
    ax2.set_xlim(0, 100); ax2.xaxis.grid(True); ax2.set_axisbelow(True); ax2.yaxis.grid(False)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3)

    fig.suptitle('Aspect mentions and their annotated polarity', x=0.055, y=1.04,
                 ha='left', fontsize=9.5, fontweight='bold')
    return _save(fig, '06_aspect_mentions')


# ------------------------------------------------------------------ tables

def table_corpus(df):
    L = df['n_chars'].astype(int)
    rows = [('Reviews', len(df)),
            ('Products', df['product_name'].nunique()),
            ('Categories', df['category'].nunique()),
            ('Distinct review texts', df['group_key'].nunique()),
            ('Fully labelled reviews', int((df['has_labels'] == 'yes').sum())),
            ('Non-verbal reviews', int((df['text_type'] == 'non-verbal').sum())),
            ('Median length (characters)', int(L.median())),
            ('Mean length (characters)', int(L.mean())),
            ('Reviews under 20 characters', int((L < 20).sum())),
            ('Date range', f"{df['date'].min()} to {df['date'].max()}")]
    return pd.DataFrame(rows, columns=['Measure', 'Value'])


def table_labels(df):
    out = []
    for col in ['language_type', 'sentiment_polarity', 'authenticity', 'category']:
        vc = df[col].replace('', 'Unlabelled').value_counts()
        for val, n in vc.items():
            out.append({'Field': col, 'Value': val, 'n': n,
                        'Percent': round(100*n/len(df), 1)})
    return pd.DataFrame(out)


def table_aspects(df):
    a = data.aspect_frame(df)
    ct = pd.crosstab(a['aspect'], a['polarity'])
    for c in ['Positive', 'Neutral', 'Negative']:
        if c not in ct:
            ct[c] = 0
    ct = ct[['Positive', 'Neutral', 'Negative']]
    ct['Total'] = ct.sum(axis=1)
    return ct.sort_values('Total', ascending=False).reset_index()


def table_length_authenticity(df):
    d = df[df['authenticity'] != ''].copy()
    d['band'] = pd.cut(d['n_chars'].astype(int),
                       bins=[b[0] for b in LENGTH_BANDS] + [10**6],
                       labels=BAND_LABELS, right=False)
    g = d.groupby('band', observed=True)['authenticity'].agg(
        Reviews='size', Suspicious=lambda s: int((s == 'Suspicious').sum()))
    g['Suspicious %'] = (100*g['Suspicious']/g['Reviews']).round(1)
    return g.reset_index().rename(columns={'band': 'Length (characters)'})


def _save_table(df, name):
    path = config.TABLES/f'{name}.csv'
    df.to_csv(path, index=False)
    print(f'  {path.name}')
    return path


def run_all():
    setup_style()
    df = data.load_gold()
    print('figures ->', config.FIGURES)
    fig_language(df); fig_stars(df); fig_sentiment_by_star(df)
    fig_length(df); fig_authenticity_by_length(df); fig_aspects(df)
    print('tables ->', config.TABLES)
    _save_table(table_corpus(df), '01_corpus_summary')
    _save_table(table_labels(df), '02_label_distributions')
    _save_table(table_aspects(df), '03_aspect_support')
    _save_table(table_length_authenticity(df), '04_length_vs_authenticity')
    try:
        _save_table(splits.report(), '05_train_test_balance')
    except FileNotFoundError:
        print('  (skipped train/test balance - run: python -m trustlens.splits)')
    return df


if __name__ == '__main__':
    run_all()
