"""The split must not leak. If any of these fail, every downstream number is wrong."""
import pandas as pd
from trustlens import data, splits


def _df():
    return splits.attach(data.load_gold())


def test_every_row_is_assigned():
    df = _df()
    assert df['split'].isin(['train', 'test']).all()
    assert df[df['split'] == 'train']['fold'].notna().all()
    assert df[df['split'] == 'test']['fold'].isna().all()


def test_no_group_spans_train_and_test():
    df = _df()
    per_group = df.groupby('group_key')['split'].nunique()
    assert (per_group == 1).all(), 'a product+text group appears on both sides'


def test_no_identical_text_spans_train_and_test():
    df = _df()
    norm = df['review_text'].str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
    per_text = df.assign(t=norm).groupby('t')['split'].nunique()
    assert (per_text == 1).all(), 'identical review text appears on both sides'


def test_no_group_spans_two_folds():
    df = _df()
    tr = df[df['split'] == 'train']
    assert (tr.groupby('group_key')['fold'].nunique() == 1).all()


def test_test_set_is_about_twenty_percent():
    df = _df()
    assert 0.17 <= (df['split'] == 'test').mean() <= 0.23


def test_minority_classes_present_in_test():
    df = _df()
    te = df[df['split'] == 'test']
    assert (te['authenticity'] == 'Suspicious').sum() >= 50
    assert (te['sentiment_polarity'] == 'Negative').sum() >= 25
    assert (te['sentiment_polarity'] == 'Neutral').sum() >= 10
