"""Charts for the web application, drawn as inline SVG on the server.

No charting library and no CDN. Two reasons, both about the demonstration:
a script tag pointing at a CDN fails on bad conference wifi and takes the whole
results page with it, and a canvas chart cannot be read by a screen reader or
copied into a document. Inline SVG needs nothing but the page itself.

Colours are the ones the thesis figures use (trustlens.eda), so a chart on
screen matches the corresponding figure in Chapter 4.

The diverging neutral is deliberately a pale low-chroma grey - that is what a
midpoint is for - which leaves it under 3:1 against the page. The palette check
allows that only where something else carries the value, so every segment here
is directly labelled and the full table sits below the charts.
"""
from __future__ import annotations

from html import escape

# Matches trustlens.eda so the app and the thesis figures agree.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
POS, NEU, NEG = '#2a78d6', '#c3c2b7', '#e34948'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, BASE, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'

R = 4          # rounded data-end radius
GAP = 2        # surface gap between adjacent fills


def _t(x, y, s, size=11, fill=INK2, anchor='start', weight='400'):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(str(s))}</text>')


def _frame(w, h, body, title, desc=''):
    return (f'<figure class="chart"><figcaption>{escape(title)}'
            + (f'<span class="sub">{escape(desc)}</span>' if desc else '')
            + '</figcaption>'
            f'<svg viewBox="0 0 {w} {h}" width="100%" height="auto" '
            f'role="img" aria-label="{escape(title)}" '
            f'preserveAspectRatio="xMinYMin meet">{body}</svg></figure>')


def star_distribution(counts: dict[int, int]) -> str:
    """How many reviews sit at each star rating. One hue: this is magnitude."""
    rows = [(s, counts.get(s, 0)) for s in (5, 4, 3, 2, 1)]
    total = sum(c for _, c in rows) or 1
    top = max(c for _, c in rows) or 1
    padL, padR, rowH, y0 = 34, 74, 26, 26
    w, h = 520, y0 + rowH*len(rows) + 8
    plot = w - padL - padR
    out = [f'<rect x="0" y="0" width="{w}" height="{h}" fill="none"/>']
    for i, (star, c) in enumerate(rows):
        y = y0 + i*rowH
        bar = plot * (c/top)
        out.append(_t(padL-8, y+13, f'{star}★', 11, MUTED, 'end'))
        out.append(f'<rect x="{padL}" y="{y+3}" width="{plot}" height="15" rx="3" '
                   f'fill="{GRID}" opacity="0.5"/>')
        if bar > 0:
            out.append(f'<rect x="{padL}" y="{y+3}" width="{max(bar, R):.1f}" '
                       f'height="15" rx="{R}" fill="{BLUE}">'
                       f'<title>{c} review{"" if c == 1 else "s"} at {star} stars '
                       f'({100*c/total:.0f}%)</title></rect>')
        # The longest bar reaches the plot edge, so its label goes inside it -
        # outside, it would be clipped by the viewBox.
        text = f'{c}  ({100*c/total:.0f}%)'
        if bar > plot*0.78:
            out.append(_t(padL+bar-8, y+14, text, 10.5, '#ffffff', 'end', '600'))
        else:
            out.append(_t(padL+bar+8, y+14, text, 10.5, INK2))
    return _frame(w, h, ''.join(out), 'What the stars say',
                  f'{total} reviews by rating')


def diverging_stacked(rows, title, desc='', label_w=None) -> str:
    """Negative left, positive right, neutral straddling the centre.

    The form for an ordered-scale share. Centring on neutral is what makes a
    five-star row with a long red arm readable at a glance - which is the whole
    claim the system makes.
    """
    rows = [r for r in rows if (r[1] + r[2] + r[3]) > 0]
    if not rows:
        return ''
    # Size the label column to the longest label rather than guessing: a fixed
    # width silently clipped "Delivery/Shipping" to "elivery/Shipping".
    if label_w is None:
        label_w = min(150, max(52, int(6.4*max(len(str(r[0])) for r in rows)) + 14))
    widest = max(max(neg + neu/2, pos + neu/2) for _, neg, neu, pos in rows) or 1
    padL, padR, rowH, y0 = label_w, 30, 30, 22
    w = 520
    h = y0 + rowH*len(rows) + 30
    half = (w - padL - padR) / 2
    mid = padL + half
    unit = half / widest
    out = [f'<line x1="{mid}" y1="{y0-8}" x2="{mid}" y2="{y0+rowH*len(rows)-4}" '
           f'stroke="{BASE}" stroke-width="1"/>']
    for i, (label, neg, neu, pos) in enumerate(rows):
        y = y0 + i*rowH
        total = neg + neu + pos
        out.append(_t(padL-10, y+14, label, 11, INK2, 'end'))
        # neutral straddles the centre line
        nl, nr = mid - (neu/2)*unit, mid + (neu/2)*unit
        segs = [(nl - neg*unit, neg*unit, NEG, 'Negative', neg),
                (nl, neu*unit, NEU, 'Neutral', neu),
                (nr, pos*unit, POS, 'Positive', pos)]
        for x, bw, colour, name, n in segs:
            if n <= 0:
                continue
            out.append(
                f'<rect x="{x + GAP/2:.1f}" y="{y+2}" width="{max(bw-GAP, 1):.1f}" '
                f'height="17" rx="2" fill="{colour}">'
                f'<title>{escape(str(label))}: {n} {name} '
                f'({100*n/total:.0f}% of {total})</title></rect>')
            if bw > 26:                       # direct label where it fits
                out.append(_t(x + bw/2, y+14.5, n, 10,
                              '#ffffff' if colour != NEU else INK2, 'middle', '600'))
        out.append(_t(mid + half + 6, y+14, total, 10.5, MUTED, 'start'))
    legend = [('Negative', NEG), ('Neutral', NEU), ('Positive', POS)]
    lx = padL
    ly = y0 + rowH*len(rows) + 12
    for name, colour in legend:
        out.append(f'<rect x="{lx}" y="{ly-8}" width="9" height="9" rx="2" '
                   f'fill="{colour}"/>')
        out.append(_t(lx+13, ly, name, 10, MUTED))
        lx += 22 + 6.1*len(name)
    return _frame(w, h, ''.join(out), title, desc)


def rating_comparison(plain, weighted, filtered=None) -> str:
    """The three readings of the same product, on one axis.

    One axis, not two: they are all ratings on the same 1-5 scale, so they
    belong on the same scale or the comparison means nothing.
    """
    items = [('Shown on the listing', plain, BASE),
             ('Weighted by trust', weighted, BLUE)]
    if filtered is not None:
        items.append(('Doubtful ones dropped', filtered, ORANGE))
    lo = min(v for _, v, _ in items)
    hi = max(v for _, v, _ in items)
    span = max(hi - lo, 0.2)
    lo, hi = lo - span*0.16, hi + span*0.16
    padL, padR, rowH, y0 = 146, 46, 30, 22
    w = 520
    h = y0 + rowH*len(items) + 16
    plot = w - padL - padR
    out = []
    for gl in range(6):
        v = lo + (hi-lo)*gl/5
        x = padL + plot*gl/5
        out.append(f'<line x1="{x:.1f}" y1="{y0-6}" x2="{x:.1f}" '
                   f'y2="{y0+rowH*len(items)-8}" stroke="{GRID}" stroke-width="1"/>')
        out.append(_t(x, h-2, f'{v:.2f}', 9, MUTED, 'middle'))
    for i, (label, v, colour) in enumerate(items):
        y = y0 + i*rowH
        x = padL + plot*(v-lo)/(hi-lo)
        out.append(_t(padL-10, y+9, label, 11, INK2, 'end'))
        out.append(f'<line x1="{padL}" y1="{y+5}" x2="{x:.1f}" y2="{y+5}" '
                   f'stroke="{colour}" stroke-width="2" stroke-linecap="round"/>')
        out.append(f'<circle cx="{x:.1f}" cy="{y+5}" r="5.5" fill="{colour}" '
                   f'stroke="{SURFACE}" stroke-width="2">'
                   f'<title>{escape(label)}: {v:.2f} stars</title></circle>')
        out.append(_t(x+11, y+9, f'{v:.2f}', 11, INK, 'start', '600'))
    return _frame(w, h, ''.join(out), 'The same product, three ways',
                  'all on the 1-5 star scale')
