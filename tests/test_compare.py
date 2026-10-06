"""The statistics behind Chapter 5.9.

A bug in the bootstrap does not crash - it produces a confidence interval that
looks perfectly reasonable and is wrong, and that interval goes into the thesis
as a claim. So the machinery is tested against cases whose answer is known in
advance.
"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from trustlens import compare


def test_a_scorer_compared_with_itself_shows_no_difference():
    """The null case. Any paired test that fails this is measuring noise."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    r = compare.paired_delta(y, s, s, n=300)
    assert r['delta'] == 0.0
    assert r['ci'] == (0.0, 0.0)


def test_a_clearly_better_scorer_is_detected():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    weak = rng.normal(size=400) + 0.15*y
    strong = rng.normal(size=400) + 3.0*y
    r = compare.paired_delta(y, weak, strong, n=400)
    assert r['delta'] > 0.2
    assert r['ci'][0] > 0, 'a large real difference must be clear at 95%'
    assert r['p_better'] > 0.99


def test_a_worse_scorer_gives_a_negative_delta():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    good = rng.normal(size=400) + 2.0*y
    r = compare.paired_delta(y, good, -good, n=300)
    assert r['delta'] < 0 and r['ci'][1] < 0


def test_the_pairing_is_real_not_two_independent_draws():
    """Paired resampling must reuse the same rows for both scorers.

    If it did not, two identical scorers would show spurious variation. This is
    the bug that would make every comparison look significant.
    """
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200) + y
    jitter = s + 1e-12
    r = compare.paired_delta(y, s, jitter, n=200)
    assert abs(r['ci'][0]) < 1e-6 and abs(r['ci'][1]) < 1e-6


def test_delta_matches_the_observed_metric_difference():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 250)
    a, b = rng.normal(size=250), rng.normal(size=250) + y
    r = compare.paired_delta(y, a, b, n=50)
    assert r['delta'] == pytest.approx(
        roc_auc_score(y, b) - roc_auc_score(y, a), abs=1e-4)


def test_vader_maps_compound_scores_to_the_three_classes():
    got = compare.vader_labels(['This is wonderful, I love it',
                                'This is terrible and broken',
                                'chair'])
    assert got[0] == 'Positive' and got[1] == 'Negative'


def test_vader_returns_neutral_for_text_it_has_no_words_for():
    """The point of the comparison: it abstains rather than failing loudly."""
    assert compare.vader_labels(['මෙය හොන්ද නැහැ'])[0] == 'Neutral'
