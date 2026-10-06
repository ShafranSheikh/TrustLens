"""The chart builders, checked for the failures that a byte count hides.

Every bug these cover was found by rendering the page and looking at it: a
label pushed outside the canvas, and a label column too narrow for its longest
entry. Both produced perfectly valid SVG of the right size.
"""
import re

import pytest

from app import charts


def _texts(svg):
    return re.findall(r'<text[^>]*>([^<]*)</text>', svg)


def _viewbox_width(svg):
    return float(re.search(r'viewBox="0 0 ([\d.]+)', svg).group(1))


def _max_x(svg):
    """Rightmost x attribute used by any element in the drawing."""
    xs = [float(m) for m in re.findall(r'\sx="([\d.]+)"', svg)]
    xs += [float(a) + float(b) for a, b in
           re.findall(r'\sx="([\d.]+)"[^>]*?\swidth="([\d.]+)"', svg)]
    return max(xs) if xs else 0.0


def test_star_distribution_keeps_its_labels_inside_the_canvas():
    """A full-width bar used to push its count label off the right edge."""
    svg = charts.star_distribution({5: 1580, 4: 17, 3: 13, 2: 1, 1: 10})
    assert _max_x(svg) <= _viewbox_width(svg)
    assert '1580' in ' '.join(_texts(svg))


def test_star_distribution_totals_are_shown():
    svg = charts.star_distribution({5: 3, 1: 1})
    joined = ' '.join(_texts(svg))
    assert '3  (75%)' in joined and '1  (25%)' in joined


def test_label_column_grows_for_a_long_label():
    """"Delivery/Shipping" was silently clipped by a fixed-width label column."""
    short = charts.diverging_stacked([('Fit', 1, 0, 2)], 't')
    long = charts.diverging_stacked([('Delivery/Shipping', 1, 0, 2)], 't')
    assert _max_x(long) <= _viewbox_width(long)
    # the long-labelled chart must reserve more room than the short one
    assert long.index('Delivery/Shipping') > 0
    assert float(re.search(r'<text x="([\d.]+)"', long).group(1)) > \
           float(re.search(r'<text x="([\d.]+)"', short).group(1))


def test_empty_rows_produce_no_chart_rather_than_an_empty_frame():
    assert charts.diverging_stacked([], 't') == ''
    assert charts.diverging_stacked([('a', 0, 0, 0)], 't') == ''


def test_every_segment_carries_a_hover_title():
    """The palette check allows the pale neutral only if the value is reachable."""
    svg = charts.diverging_stacked([('5 stars', 3, 2, 9)], 't')
    assert svg.count('<title>') >= 3


def test_rating_comparison_handles_three_identical_values():
    """A zero span must not divide by zero or collapse the axis."""
    svg = charts.rating_comparison(4.5, 4.5, 4.5)
    assert '<svg' in svg and _max_x(svg) <= _viewbox_width(svg)


def test_rating_comparison_without_the_filtered_figure():
    svg = charts.rating_comparison(4.2, 4.0, None)
    assert 'Doubtful ones dropped' not in svg
    assert 'Weighted by trust' in svg


def test_a_label_cannot_open_a_tag():
    """Product names come off a third-party page, so they are untrusted text."""
    svg = charts.diverging_stacked([('x<script>alert(1)</script>y', 1, 0, 1)], 't')
    assert '<script>' not in svg
    assert '&lt;script&gt;' in svg


def test_a_title_cannot_break_out_of_an_attribute():
    """The figure title is interpolated into aria-label, so quotes must escape."""
    svg = charts.diverging_stacked([('a', 1, 0, 1)], 'He said "go" now')
    # the quotes that would close aria-label are gone; the text survives
    assert '&quot;go&quot;' in svg
    assert 'aria-label="He said &quot;go&quot; now"' in svg


def test_untrusted_text_never_reaches_a_hover_title_raw():
    svg = charts.diverging_stacked([('<b>x</b>', 2, 0, 1)], 't')
    assert '<b>' not in svg
