"""Load the gold dataset and expose clean, task-specific views.

Nothing else in the project reads the CSV directly - always go through here,
so filtering rules live in one place and every notebook sees the same data.

    from trustlens import data
    df = data.load_gold()
"""
from __future__ import annotations
import pandas as pd
from . import config

INT_COLS = ['star_rating', 'helpful_count', 'n_chars', 'dup_count']


def load_gold(linguistic_only: bool = False) -> pd.DataFrame:
    """The full gold dataset.

    linguistic_only=True drops the 27 non-verbal rows (emoji, keyboard mash).
    Use it for the language and sentiment models; leave it False for fake
    detection, where those rows are legitimate data.
    """
    df = pd.read_csv(config.GOLD, encoding='utf-8-sig', dtype=str).fillna('')
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    if linguistic_only:
        df = df[df['text_type'] == 'linguistic']
    return df.reset_index(drop=True)


def parse_aspects(s: str) -> list[tuple[str, str]]:
    """'Product Quality:Positive; Packaging:Negative' -> [(cat, polarity), ...]"""
    out = []
    for part in str(s).split(';'):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            cat, pol = part.rsplit(':', 1)
            out.append((cat.strip(), pol.strip()))
        else:
            out.append((part, ''))
    return out


def aspect_frame(df: pd.DataFrame, column: str = 'aspects_merged') -> pd.DataFrame:
    """One row per (review, aspect) pair. Long format, easy to group and count."""
    rows = []
    for rid, s in zip(df['review_id'], df[column]):
        for cat, pol in parse_aspects(s):
            rows.append({'review_id': rid, 'aspect': cat, 'polarity': pol})
    return pd.DataFrame(rows, columns=['review_id', 'aspect', 'polarity'])


def aspect_matrix(df: pd.DataFrame, column: str = 'aspects_merged') -> pd.DataFrame:
    """One column per aspect: '' if not mentioned, else its polarity.

    Indexed by review_id. This is what the ABSA models train against.
    """
    m = pd.DataFrame('', index=df['review_id'], columns=config.ASPECTS)
    for rid, s in zip(df['review_id'], df[column]):
        for cat, pol in parse_aspects(s):
            if cat in config.ASPECTS:
                m.at[rid, cat] = pol
    return m


# ---------------------------------------------------------------- task views
# Each returns only the rows that task can actually learn from.

def sentiment_data() -> pd.DataFrame:
    """Rows the sentiment model can learn from.

    Excludes non-verbal text and rows annotated 'Not applicable' - reviews whose
    text carries no expressible sentiment. Those are a deliberate annotation
    decision, not missing data, so they are excluded here but kept in the corpus.
    """
    df = load_gold(linguistic_only=True)
    keep = ~df['sentiment_polarity'].isin(['', config.SENTIMENT_NA])
    return df[keep].reset_index(drop=True)


def absa_data() -> pd.DataFrame:
    """Linguistic rows. Reviews with no aspect are valid negatives for detection."""
    return load_gold(linguistic_only=True)


def fake_data() -> pd.DataFrame:
    """Rows with an authenticity label. Non-verbal rows are kept on purpose."""
    df = load_gold()
    return df[df['authenticity'] != ''].reset_index(drop=True)


def summary(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Quick label counts - handy at the top of a notebook."""
    df = load_gold() if df is None else df
    out = []
    for col in ['language_type', 'sentiment_polarity', 'authenticity', 'text_type']:
        for val, n in df[col].value_counts(dropna=False).items():
            out.append({'field': col, 'value': val if val != '' else '(blank)', 'n': n})
    return pd.DataFrame(out)
