"""Design diagrams for Chapter 4, Section 4.5: use case, data model, interface.

Companion to ch4_diagrams.py, whose Sheet helper is reused. Every box here
sizes itself from its text, in both directions: height from the line count and
font size, width checked by `fits_width` with the font shrunk until the longest
line fits. The first version of all three of these figures had text spilling
out of its boxes, and none of it was visible in the code.

A note on the data model. This system has no database: the corpus is a single
delimited file and the fitted models are serialised objects on disk. The entity
relationship diagram therefore describes the LOGICAL structure of the data the
pipeline operates on, not a physical schema. It is worth drawing even so,
because one of the relationships it shows is the reason a stored column has to
be decomposed before the aspect models can use it.

    python scripts/ch4_design_diagrams.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from ch4_diagrams import (Sheet, LINE_SP, shrink_to_fit, BLUE, ORANGE, AQUA,
                          INK, INK2, MUTED, GRID, BASE, SURFACE, PALE, MINT,
                          PEACH, DEEP_BLUE, DEEP_GREEN, DEEP_ORANGE)


def actor(ax, x, y, label, colour=INK2, s=0.028):
    ax.add_patch(plt.Circle((x, y + s*2.1), s*0.62, fill=False,
                            edgecolor=colour, linewidth=1.2, zorder=3))
    ax.plot([x, x], [y + s*1.45, y - s*0.55], color=colour, lw=1.2, zorder=3)
    ax.plot([x - s*0.95, x + s*0.95], [y + s*0.95, y + s*0.95],
            color=colour, lw=1.2, zorder=3)
    ax.plot([x, x - s*0.8], [y - s*0.55, y - s*2.0], color=colour, lw=1.2, zorder=3)
    ax.plot([x, x + s*0.8], [y - s*0.55, y - s*2.0], color=colour, lw=1.2, zorder=3)
    ax.text(x, y - s*2.8, label, ha='center', va='top', fontsize=8.2,
            color=colour, fontweight='600', linespacing=1.3)


# ------------------------------------------------------------------ use case
def fig_use_case():
    s = Sheet(7.0, 4.3)
    ax = s.ax
    EW, EH = 0.285, 0.104

    def oval(x, y, text, research=False):
        fs = shrink_to_fit(text, EW - 0.030, 8.2, s.W)
        el = ax.add_patch(Ellipse((x, y), EW, EH,
                                  facecolor=MINT if research else PALE,
                                  edgecolor=AQUA if research else BLUE,
                                  linewidth=1.0, zorder=2))
        tx = ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                     color=DEEP_GREEN if research else DEEP_BLUE, zorder=3,
                     linespacing=1.3)
        s.track(el, tx)

    def link(x1, y1, x2, y2, colour=MUTED):
        ax.plot([x1, x2], [y1, y2], color=colour, lw=0.9, zorder=1)

    def include(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=7, color=ORANGE, lw=0.9,
                                     linestyle=(0, (3, 2.2)), shrinkA=1,
                                     shrinkB=1, zorder=1))
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.018, '«include»', ha='center',
                fontsize=6.6, color=ORANGE, style='italic', zorder=3)

    ax.add_patch(Rectangle((0.205, 0.050), 0.650, 0.880, fill=False,
                           edgecolor=BASE, linewidth=1.2, zorder=1))
    ax.text(0.530, 0.948, 'TrustLens', ha='center', fontsize=8.6,
            fontweight='700', color=INK)
    ax.plot([0.205, 0.855], [0.455, 0.455], color=GRID, lw=0.9,
            linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.843, 0.472, 'shopper-facing', ha='right', fontsize=7,
            color=MUTED, style='italic')
    ax.text(0.843, 0.432, 'research pipeline', ha='right', va='top',
            fontsize=7, color=MUTED, style='italic')

    LC, RC = 0.370, 0.700
    shopper = [(LC, 0.855, 'Analyse a product\nfrom its link'),
               (LC, 0.750, 'Inspect the evidence\nfor each review'),
               (LC, 0.645, 'View the\naspect breakdown'),
               (LC, 0.540, 'Run the offline\ndemonstration')]
    incl = [(RC, 0.855, 'Fetch the reviews\nfrom the page'),
            (RC, 0.700, 'Score reviews and\nweight by trust')]
    research = [(LC, 0.355, 'Build and\nannotate the corpus'),
                (LC, 0.245, 'Train and\npersist the models'),
                (LC, 0.135, 'Evaluate and compare')]
    for x, y, t in shopper + incl:
        oval(x, y, t)
    for x, y, t in research:
        oval(x, y, t, research=True)

    actor(ax, 0.085, 0.730, 'Shopper')
    actor(ax, 0.085, 0.260, 'Researcher')
    actor(ax, 0.930, 0.830, 'E-commerce\nplatform', colour=DEEP_ORANGE)

    for x, y, _ in shopper:
        link(0.120, 0.715, x - EW/2, y)
    for x, y, _ in research:
        link(0.120, 0.245, x - EW/2, y)
    link(0.895, 0.822, RC + EW/2, 0.855, colour=DEEP_ORANGE)

    include(LC + EW/2, 0.855, RC - EW/2, 0.855)
    include(LC + EW/2 - 0.012, 0.820, RC - EW/2 + 0.004, 0.728)
    s.verify('use case')
    return s.save('21_use_case')


# ----------------------------------------------------------------------- ERD
def fig_erd():
    s = Sheet(7.0, 5.0)
    ax = s.ax
    HFS, AFS = 9.0, 7.8

    def entity(x, top, w, name, attrs, face=PALE, edge=BLUE, colour=DEEP_BLUE):
        body = '\n'.join(attrs)
        afs = shrink_to_fit(body, w - 0.024, AFS, s.W)
        hh = (1 + 0.65) * LINE_SP * HFS / (72.0 * s.H)
        bh = (len(attrs) + 1.35) * LINE_SP * afs / (72.0 * s.H)
        body_patch = ax.add_patch(FancyBboxPatch(
            (x, top - hh - bh), w, hh + bh,
            boxstyle='round,pad=0.003,rounding_size=0.012',
            facecolor=SURFACE, edgecolor=edge, linewidth=1.1, zorder=2))
        ax.add_patch(FancyBboxPatch((x, top - hh), w, hh,
                                    boxstyle='round,pad=0.003,rounding_size=0.012',
                                    facecolor=face, edgecolor=edge,
                                    linewidth=1.1, zorder=2))
        ax.text(x + w/2, top - hh/2, name, ha='center', va='center',
                fontsize=HFS, fontweight='700', color=colour, zorder=3)
        bt = ax.text(x + 0.012, top - hh - bh/2, body, ha='left', va='center',
                     fontsize=afs, color=INK2, zorder=3, linespacing=LINE_SP)
        s.track(body_patch, bt)
        return top - hh - bh

    def rel(x1, y1, x2, y2, left, right):
        ax.plot([x1, x2], [y1, y2], color=MUTED, lw=1.0, zorder=1)
        dx, dy = x2 - x1, y2 - y1
        vertical = abs(dx) < 1e-6
        ox, oy = (0.016, 0.0) if vertical else (0.0, 0.020)
        ax.text(x1 + dx*0.16 + ox, y1 + dy*0.16 + oy, left, ha='center',
                fontsize=7.4, color=MUTED, fontweight='700', zorder=3)
        ax.text(x1 + dx*0.84 + ox, y1 + dy*0.84 + oy, right, ha='center',
                fontsize=7.4, color=MUTED, fontweight='700', zorder=3)

    C1X, C1W = 0.025, 0.225
    C2X, C2W = 0.330, 0.300
    C3X, C3W = 0.700, 0.280

    entity(C1X, 0.930, C1W, 'PRODUCT', ['product_name  PK', 'category'])
    entity(C1X, 0.700, C1W, 'TEXT_GROUP', ['group_key  PK', 'normalised_text'],
           face=PEACH, edge=ORANGE, colour=DEEP_ORANGE)
    rb = entity(C2X, 0.930, C2W, 'REVIEW',
                ['review_id  PK', 'product_name  FK', 'group_key  FK', 'date',
                 'star_rating', 'review_text', 'variant', 'verified_purchase',
                 'helpful_count', 'n_chars', 'dup_count'])
    entity(C3X, 0.930, C3W, 'ANNOTATION',
           ['review_id  PK FK', 'language_type', 'sentiment_polarity',
            'authenticity', 'text_type'], face=MINT, edge=AQUA, colour=DEEP_GREEN)
    mb = entity(C3X, 0.660, C3W, 'ASPECT_MENTION',
                ['review_id  PK FK', 'aspect_name  PK FK', 'polarity'],
                face=MINT, edge=AQUA, colour=DEEP_GREEN)
    entity(C3X, 0.430, C3W, 'ASPECT',
           ['aspect_name  PK', '(seven fixed categories)'])
    sb = entity(C2X, rb - 0.070, C2W, 'DERIVED_SCORE',
                ['review_id  PK FK', 'sentiment_score', 'ssg,  ascs,  isg',
                 'cii', 'p_suspicious', 'trust_weight'],
                face=PEACH, edge=ORANGE, colour=DEEP_ORANGE)

    rel(C1X + C1W, 0.868, C2X, 0.868, '1', 'N')
    rel(C1X + C1W, 0.638, C2X, 0.638, '1', 'N')
    rel(C2X + C2W, 0.868, C3X, 0.868, '1', '1')
    rel(C2X + C2W, 0.640, C3X, 0.600, '1', 'N')
    rel(C2X + C2W*0.84, rb, C2X + C2W*0.84, rb - 0.070, '1', '1')
    rel(C3X + C3W*0.86, mb, C3X + C3W*0.86, 0.430, 'N', '1')

    s.note(C1X, sb - 0.085,
           'No relational database is used: the corpus is held in one delimited file and the\n'
           'fitted models as serialised objects on disk. What is shown is the logical\n'
           'structure the pipeline operates on. ASPECT_MENTION is the decomposition of a\n'
           'multi-valued stored column, performed in code before any aspect model\n'
           'sees the data.',
           fs=7.2, ha='left')
    s.verify('ERD')
    return s.save('22_erd')


# ----------------------------------------------------------------- UI design
def fig_ui_design():
    s = Sheet(7.0, 3.85)
    ax = s.ax
    CHROME = 0.050

    def region_h(label, note, fs):
        n = 1 + (note.count('\n') + 1 if note else 0)
        return (n + 1.05) * LINE_SP * fs / (72.0 * s.H)

    def region(x, top, w, label, note=None, face=SURFACE, edge=GRID,
               colour=INK2, fs=8.0, weight='600'):
        fs = shrink_to_fit(label, w - 0.020, fs, s.W)
        if note:
            fs = min(fs, shrink_to_fit(note, w - 0.020, fs, s.W) + 0.0)
        h = region_h(label, note, fs)
        rp = ax.add_patch(FancyBboxPatch((x, top - h), w, h,
                                         boxstyle='round,pad=0.003,rounding_size=0.010',
                                         facecolor=face, edgecolor=edge,
                                         linewidth=0.9, zorder=3))
        ly = top - h/2 + (h*0.20 if note else 0)
        lt = ax.text(x + w/2, ly, label, ha='center', va='center', fontsize=fs,
                     color=colour, fontweight=weight, zorder=4)
        if face != 'none':
            s.track(rp, lt)
        if note:
            nt = ax.text(x + w/2, top - h/2 - h*0.17, note, ha='center',
                         va='center', fontsize=max(fs - 1.0, 6.2), color=MUTED,
                         zorder=4, linespacing=1.35)
            if face != 'none':
                s.track(rp, nt)
        return top - h

    def chrome(x, top, w, h, title):
        ax.add_patch(FancyBboxPatch((x, top - h), w, h,
                                    boxstyle='round,pad=0.004,rounding_size=0.014',
                                    facecolor='#fbfbf9', edgecolor=BASE,
                                    linewidth=1.3, zorder=1))
        ax.add_patch(FancyBboxPatch((x, top - CHROME), w, CHROME,
                                    boxstyle='round,pad=0.003,rounding_size=0.012',
                                    facecolor='#eeeeea', edgecolor=BASE,
                                    linewidth=1.0, zorder=2))
        for i in range(3):
            ax.add_patch(plt.Circle((x + 0.020 + i*0.016, top - CHROME/2), 0.0055,
                                    facecolor='#cfcec7', edgecolor='none', zorder=3))
        ax.text(x + w/2, top - CHROME/2, title, ha='center', va='center',
                fontsize=7.6, color=MUTED, zorder=3)

    TOP = 0.945
    # ---------- entry screen ----------
    EX, EW = 0.030, 0.425
    rows_e = [
        dict(label='TrustLens', note='one line saying what the page does',
             face='none', edge='none', colour=INK, fs=9.8, weight='700'),
        dict(label='Product link', note='a single text field, autofocused',
             face=PALE, edge=BLUE, colour=DEEP_BLUE),
        dict(label='Check', note=None, face=BLUE, edge=BLUE,
             colour='#ffffff', fs=7.8),
        dict(label='Progress', note='stage, pages read, reviews found, elapsed',
             face=PEACH, edge=ORANGE, colour=DEEP_ORANGE),
    ]
    he = CHROME + 0.030 + sum(region_h(r['label'], r.get('note'),
                                       r.get('fs', 8.0)) + 0.014 for r in rows_e)
    chrome(EX, TOP, EW, he, 'Entry screen')
    y = TOP - CHROME - 0.018
    for r in rows_e:
        w = 0.120 if r['label'] == 'Check' else EW - 0.040
        y = region(EX + 0.020, y, w, **r) - 0.014
    s.note(EX + EW/2, TOP - he - 0.035,
           'One input and one action; nothing else competes for attention.', fs=7.2)

    # ---------- result screen ----------
    RX, RW = 0.515, 0.455
    rows_r = [
        dict(label='4.57  →  4.47',
             note='the listed rating beside its trust-weighted value',
             face=PALE, edge=BLUE, colour=DEEP_BLUE, fs=10.4, weight='700'),
        dict(label='Why it moved, or did not',
             note='how many were flagged, how many contradict'),
        None,                                     # the side-by-side pair
        dict(label='Sentiment at each star rating',
             note='the chart that carries the contribution',
             face=MINT, edge=AQUA, colour=DEEP_GREEN),
        dict(label='Aspect breakdown', note='sentiment per aspect, all reviews'),
        dict(label='Review table',
             note='least trusted first: stars, text, sentiment,\nmatch, trust'),
    ]
    pair_h = region_h('Rating', 'three ways', 8.0)
    hr = CHROME + 0.030 + sum((pair_h if r is None else
                               region_h(r['label'], r.get('note'), r.get('fs', 8.0)))
                              + 0.014 for r in rows_r)
    chrome(RX, TOP, RW, hr, 'Result screen')
    y = TOP - CHROME - 0.018
    half = (RW - 0.040 - 0.014) / 2
    for r in rows_r:
        if r is None:
            region(RX + 0.020, y, half, 'Rating', 'three ways')
            y = region(RX + 0.020 + half + 0.014, y, half, 'Star', 'distribution') - 0.014
        else:
            y = region(RX + 0.020, y, RW - 0.040, **r) - 0.014
    s.note(RX + RW/2, TOP - hr - 0.035,
           'Verdict first, then the evidence for it, then the reviews themselves.', fs=7.2)
    s.verify('UI design')
    return s.save('23_ui_design')


if __name__ == '__main__':
    fig_use_case()
    fig_erd()
    fig_ui_design()
