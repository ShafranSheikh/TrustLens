"""Closed-form tests for the index. If these break, the contribution is wrong."""
import numpy as np
import pytest
from trustlens import cii, config


# ------------------------------------------------------------------ star mapping

def test_stars_map_onto_minus_one_to_one():
    assert list(cii.star_value([1, 2, 3, 4, 5])) == [-1.0, -0.5, 0.0, 0.5, 1.0]


# ------------------------------------------------------------------ SSG

@pytest.mark.parametrize('score,star,expected', [
    (1.0, 5, 0.0),     # glowing text, five stars - no gap
    (-1.0, 1, 0.0),    # scathing text, one star - no gap
    (1.0, 1, 1.0),     # glowing text, one star - maximum gap
    (-1.0, 5, 1.0),    # scathing text, five stars - maximum gap
    (0.0, 3, 0.0),     # neutral text, three stars
    (0.0, 5, 0.5),     # neutral text, five stars - half the range
])
def test_ssg_closed_form(score, star, expected):
    assert cii.ssg([score], [star])[0] == pytest.approx(expected)


def test_ssg_stays_in_range():
    rng = np.random.default_rng(0)
    v = cii.ssg(rng.uniform(-1, 1, 500), rng.integers(1, 6, 500))
    assert v.min() >= 0 and v.max() <= 1


# ------------------------------------------------------------------ ASCS

def test_ascs_counts_only_contradicting_aspects():
    # five stars: a Negative aspect contradicts, a Positive one does not
    assert cii.ascs([{'A': 'Negative'}], [5])[0] == 1.0
    assert cii.ascs([{'A': 'Positive'}], [5])[0] == 0.0
    assert cii.ascs([{'A': 'Negative', 'B': 'Positive'}], [5])[0] == 0.5
    # one star: the other way round
    assert cii.ascs([{'A': 'Positive'}], [1])[0] == 1.0
    assert cii.ascs([{'A': 'Negative'}], [1])[0] == 0.0


def test_ascs_neutral_aspects_never_contradict():
    assert cii.ascs([{'A': 'Neutral'}], [5])[0] == 0.0
    assert cii.ascs([{'A': 'Neutral', 'B': 'Negative'}], [5])[0] == 0.5


def test_three_star_reviews_are_excluded_as_nan_not_zero():
    """A skipped row must not read as a consistent one."""
    v = cii.ascs([{'A': 'Negative'}], [3])[0]
    assert np.isnan(v), 'three-star reviews must be NaN, never 0.0'


def test_no_aspects_is_nan():
    assert np.isnan(cii.ascs([{}], [5])[0])


# ------------------------------------------------------------------ ISG markers

def test_intensifiers_match_on_word_boundaries():
    """'very' must not match inside 'delivery' - the bug this guards against."""
    assert cii.marker_counts(['delivery was fast'])['intensifier'][0] == 0
    assert cii.marker_counts(['very fast'])['intensifier'][0] == 1
    assert cii.marker_counts(['sober'])['intensifier'][0] == 0        # 'so' inside a word
    assert cii.marker_counts(['so good'])['intensifier'][0] == 1


def test_intensifiers_cover_all_three_languages():
    for text in ['very good', 'godak hodai', 'ගොඩක් හොඳයි']:
        assert cii.marker_counts([text])['intensifier'][0] >= 1, text


def test_markers_count_what_they_claim():
    c = cii.marker_counts(['sooo goooood!! 👍👍'])
    assert c['elongation'][0] == 2        # 'sooo', 'goooood'
    assert c['punctuation'][0] == 2       # two '!'
    assert c['emoji'][0] == 2


def test_isg_stays_in_range():
    texts = ['good', 'sooo amazing!!! 😍😍😍', '', 'ok']
    scale = cii.fit_intensity(texts)
    v = cii.isg_linear(texts, [5, 5, 3, 1], scale)
    assert v.min() >= 0 and v.max() <= 1


# ------------------------------------------------------------------ combination

def test_weights_renormalise_when_a_component_is_missing():
    """With ASCS absent, CII must be the weighted mean of what remains."""
    v = cii.combine([0.4], [np.nan], [0.8], {'ssg': 1/3, 'ascs': 1/3, 'isg': 1/3})
    assert v[0] == pytest.approx(0.6)          # (0.4 + 0.8) / 2, not / 3


def test_combine_matches_the_formula_when_all_present():
    w = {'ssg': 0.5, 'ascs': 0.3, 'isg': 0.2}
    v = cii.combine([1.0], [0.5], [0.0], w)
    assert v[0] == pytest.approx(0.5*1.0 + 0.3*0.5 + 0.2*0.0)


def test_combine_stays_in_range():
    rng = np.random.default_rng(1)
    v = cii.combine(rng.uniform(0, 1, 300), rng.uniform(0, 1, 300), rng.uniform(0, 1, 300))
    assert np.nanmin(v) >= 0 and np.nanmax(v) <= 1


def test_all_nan_components_give_nan():
    assert np.isnan(cii.combine([np.nan], [np.nan], [np.nan])[0])


# ------------------------------------------------------------------ centring

def test_centred_isg_is_near_zero_for_a_typical_review():
    """The pre-specified form scored plain five-star reviews as maximally
    inconsistent. The centred form must not."""
    texts = ['good product']*20 + ['great']*20
    stars = [5]*40
    scale = cii.fit_intensity(texts)
    expected = cii.fit_expected(texts, stars, [0.9]*40, scale)
    assert cii.isg(texts, stars, scale, expected)[0] < 0.2
    assert cii.isg_linear(texts, stars, scale)[0] > 0.8      # the old behaviour
