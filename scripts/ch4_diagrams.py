"""Block diagrams for Chapter 4.

Same palette, font and resolution as the result figures in trustlens.eda, so a
reader turning from a diagram to a chart does not cross a visual seam. These
are not charts - nothing is encoded in position or length - so colour is used
only to separate layers of the system.

Box heights are COMPUTED from the line count and font size rather than chosen
by hand. Hand-set heights were the reason the first version of every one of
these figures had text spilling out of its box: the overflow is invisible in
the code and only shows once the figure is rendered and looked at.

    python scripts/ch4_diagrams.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from trustlens import eda

BLUE, ORANGE, AQUA = eda.BLUE, eda.ORANGE, eda.AQUA
INK, INK2, MUTED = eda.INK, eda.INK2, eda.MUTED
GRID, BASE, SURFACE = eda.GRID, eda.BASE, eda.SURFACE
PALE, MINT, PEACH, ROSE = '#eaf1fb', '#e9f7f1', '#fdf0e6', '#fdeceb'
DEEP_BLUE, DEEP_GREEN, DEEP_ORANGE, DEEP_RED = '#17508f', '#0e6b4a', '#8a3f14', '#8c2019'
LINE_SP = 1.42


def fits_width(text, w_axes, fs, w_inches, factor=0.68):
    """Does the longest line of `text` fit in `w_axes` at this font size?

    An estimate, not a measurement. The factor was calibrated against a rendered
    figure: a bold 17-character label at 6.4pt measured 0.658 em per character,
    so 0.68 leaves a margin. It is deliberately conservative, and it turns a
    silent overflow into something the build can catch. Every diagram in this
    chapter overflowed its boxes on the first attempt, and none of it was
    visible until the figure was rendered and looked at.
    """
    longest = max((len(l) for l in str(text).split('\n')), default=0)
    return (longest * factor * fs) / (72.0 * w_inches) <= w_axes


def shrink_to_fit(text, w_axes, fs, w_inches, min_fs=5.0):
    """The largest size at or below `fs` whose longest line fits."""
    while fs > min_fs and not fits_width(text, w_axes, fs, w_inches):
        fs -= 0.2
    return round(fs, 1)


class Sheet:
    """A diagram canvas in 0-1 coordinates that knows its own physical height.

    Knowing the height in inches is what lets a box size itself to its text.
    """

    def __init__(self, w_in, h_in):
        eda.setup_style()
        self.H = h_in
        self.W = w_in
        self.fig, self.ax = plt.subplots(figsize=(w_in, h_in))
        # Axes coordinates map to the AXES box, not the figure. With default
        # subplot margins the axes is about 77% of the figure height, so every
        # height computed as a fraction of the figure came out roughly a
        # quarter too small - which is why each of these diagrams overflowed
        # its boxes. Filling the figure makes the two coordinate systems agree.
        self.ax.set_position([0, 0, 1, 1])
        self._pairs = []
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')

    def text_h(self, text, fs, pad_lines=0.85):
        """Height in axes units that this text needs, padding included."""
        n = text.count('\n') + 1
        return (n + pad_lines) * LINE_SP * fs / (72.0 * self.H)

    def box(self, x, top, w, text, fs=7.4, face=SURFACE, edge=BASE,
            colour=INK2, weight='400', lw=1.0, h=None, pad_lines=0.85):
        """Draw a box whose TOP edge is at `top`. Returns its bottom edge."""
        fs = shrink_to_fit(text, w - 0.016, fs, self.W)
        h = h if h is not None else self.text_h(text, fs, pad_lines)
        pt = self.ax.add_patch(FancyBboxPatch(
            (x, top - h), w, h,
            boxstyle='round,pad=0.004,rounding_size=0.018',
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
        tx = self.ax.text(x + w/2, top - h/2, text, ha='center', va='center',
                          fontsize=fs, color=colour, fontweight=weight,
                          zorder=3, linespacing=LINE_SP)
        self.track(pt, tx)
        return top - h

    def arrow(self, x1, y1, x2, y2, colour=MUTED, lw=1.1):
        self.ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8,
            color=colour, linewidth=lw, shrinkA=0, shrinkB=0, zorder=1))

    def line(self, x1, y1, x2, y2, colour=MUTED, lw=1.1):
        """A connector segment with no head, for elbows and buses."""
        self.ax.plot([x1, x2], [y1, y2], color=colour, linewidth=lw,
                     solid_capstyle='round', zorder=1)

    def crop(self, bottom, pad=0.018):
        """Discard unused canvas below the content.

        bbox_inches='tight' crops to the drawn artists, but the axes fills the
        figure, so an over-tall sheet keeps its empty lower half. Narrowing the
        y range and shrinking the figure by the SAME factor leaves
        inches-per-axes-unit unchanged, so every box already sized against
        self.H stays physically the same and the text keeps fitting.
        """
        lo = max(0.0, bottom - pad)
        frac = 1.0 - lo
        self.ax.set_ylim(lo, 1.0)
        self.fig.set_size_inches(self.W, self.H*frac)
        self.H *= frac

    def note(self, x, y, text, fs=6.6, ha='center', colour=MUTED):
        self.ax.text(x, y, text, ha=ha, va='center', fontsize=fs,
                     color=colour, style='italic', linespacing=LINE_SP)

    def track(self, patch, text):
        """Record a box and the text inside it, for verify() to check."""
        self._pairs.append((patch, text))

    def verify(self, name='figure'):
        """Measure, with the renderer, whether any text escapes its box.

        The width estimate in fits_width is a guess; this is the ground truth.
        Called before saving, so an overflow fails the build instead of being
        something that has to be spotted by eye in the rendered image.
        """
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        bad = []
        for patch, text in self._pairs:
            tb, pb = text.get_window_extent(r), patch.get_window_extent(r)
            over = max(pb.x0 - tb.x0, tb.x1 - pb.x1,
                       pb.y0 - tb.y0, tb.y1 - pb.y1)
            if over > 0.5:
                label = text.get_text().replace('\n', ' / ')[:38]
                bad.append((label, round(over, 1)))
        if bad:
            lines = '\n'.join(f'    +{o}px  {t}' for t, o in bad)
            raise AssertionError(f'{name}: text outside its box\n{lines}')
        return True

    def save(self, name, tight=True):
        """Save, cropping unused canvas.

        The layout flows downward from the top and stops where the content
        stops, so the amount of blank canvas below it depends on how much text
        each box turned out to need. Cropping to the drawn content avoids
        tuning figure heights by hand every time a label changes.
        """
        from trustlens import config
        import matplotlib.pyplot as _plt
        if not tight:
            return eda._save(self.fig, name)
        path = config.FIGURES/f'{name}.png'
        self.fig.savefig(path, bbox_inches='tight', pad_inches=0.04)
        _plt.close(self.fig)
        print(f'  {path.name}')
        return path


GAP = 0.030          # vertical breathing room, and the space an arrow lives in


# ------------------------------------------------------------- architecture
def fig_architecture():
    """The pipeline with its internal data flow, not one box per phase.

    Colour encodes role rather than phase, and the numbered chips carry phase
    identity: five distinct hues fails the colourblind separation check (green
    against red reaches only 6.9 deltaE under deuteranopia), and the thesis
    figures already use one validated three-colour set.

    Every element is the system as built, which in five places differs from the
    design as proposed: the two sentiment branches fit separate vectorisers,
    the detector uses five feature families rather than seven, SSG is centred
    on the observed per-rating mean rather than a linear mapping, the corpus
    narrows from 2,506 to 1,977, and the application fetches reviews live.

    The edges are the real dependencies, which is why they are not a single
    column of arrows. SSG consumes the review-level sentiment score and ASCS
    the aspect polarities, so those are separate edges rather than one fan;
    ISG consumes the review text through the intensity lexicon and touches
    neither model (cii.isg takes texts and star, not a sentiment score), so it
    is fed from the corpus down the right-hand rail. An arrow drawn between
    two parallel branches, or into a box that does not consume the source,
    asserts an order the system does not have.
    """
    s = Sheet(7.0, 6.3)
    CX, CW = 0.145, 0.815          # content column
    PX, PW = 0.020, 0.105          # phase chip gutter
    RAIL = 0.980                   # right-hand rail, outside the content column
    GX = 0.022                     # horizontal gap between boxes
    ROLE = {'data': (SURFACE, BASE, INK2),
            'model': (PALE, BLUE, DEEP_BLUE),
            'index': (PEACH, ORANGE, DEEP_ORANGE),
            'out': (MINT, AQUA, DEEP_GREEN)}

    def colw(n):
        return (CW - GX*(n - 1)) / n

    def col(i, n):
        """Centre of column i of n."""
        w = colw(n)
        return CX + i*(w + GX) + w/2

    def row(top, items, role, fs=6.9, seq=True):
        """One rank of equal-height boxes.

        seq=True joins them left to right, for a genuine sequence. seq=False
        leaves them unjoined, for branches that run in parallel.
        """
        face, edge, colr = ROLE[role]
        n, w = len(items), colw(len(items))
        h = max(s.text_h(t, fs) for t in items)
        for i, t in enumerate(items):
            x = CX + i*(w + GX)
            s.box(x, top, w, t, fs=fs, face=face, edge=edge, colour=colr, h=h)
            if seq and i < n - 1:
                s.arrow(x + w, top - h/2, x + w + GX, top - h/2, colour=edge)
        return top - h

    def elbow(x1, y1, x2, y2, colour=MUTED):
        """One edge between offset columns: down, across, down."""
        mid = (y1 + y2)/2
        if abs(x1 - x2) < 1e-9:
            s.arrow(x1, y1, x2, y2, colour=colour)
            return
        s.line(x1, y1, x1, mid, colour=colour)
        s.line(x1, mid, x2, mid, colour=colour)
        s.arrow(x2, mid, x2, y2, colour=colour)

    def bus(sources, targets, y1, y2, colour=MUTED):
        """A fan-out or a join, through one shared horizontal line."""
        mid = (y1 + y2)/2
        for x in sources:
            s.line(x, y1, x, mid, colour=colour)
        xs = list(sources) + list(targets)
        s.line(min(xs), mid, max(xs), mid, colour=colour)
        for x in targets:
            s.arrow(x, mid, x, y2, colour=colour)

    def chip(top, bottom, n, label, role):
        face, edge, colr = ROLE[role]
        hh = s.text_h(f'Phase {n}\n{label}', 7.4)
        s.box(PX, (top + bottom)/2 + hh/2, PW, f'Phase {n}\n{label}', fs=7.4,
              face=face, edge=edge, colour=colr, weight='700', lw=1.2)

    # ---- Phase 1: collection -------------------------------------------
    t1 = 0.985
    b1 = row(t1, ['A Sri Lankan\ne-commerce platform\nelectronics, fashion,\nhealth and personal care',
                  'Selenium WebDriver\nscraper',
                  '2,506 reviews\nscraped',
                  '2,017 annotated\n1,977 analysed'], 'data', fs=6.6)
    chip(t1, b1, 1, 'Collection and\nannotation', 'data')

    # ---- Phase 2: the two sentiment branches ----------------------------
    t2 = b1 - GAP
    b2 = row(t2, ['Review-level sentiment\nits own TF-IDF, character n-grams 2\u20134,\n'
                  'logistic regression \u2192 sentiment score',
                  'Aspect-level sentiment\na second TF-IDF, shared across\n'
                  '7 detectors and 7 polarity heads'], 'model', seq=False)
    chip(t2, b2, 2, 'Cross-lingual\nsentiment', 'model')
    bus([col(3, 4)], [col(0, 2), col(1, 2)], b1, t2)

    # ---- Phase 3: the three components, then the index ------------------
    t3 = b2 - GAP
    b3 = row(t3, ['SSG\nSentiment\u2013Star Gap\ndeparture from the mean\n'
                  'sentiment at that rating',
                  'ASCS\nAspect\u2013Star Conflict\nshare of aspects opposing\n'
                  'the rating direction',
                  'ISG\nIntensity\u2013Star Gap\ndeparture from the emphasis\n'
                  'usual at that rating'], 'index', fs=6.5, seq=False)
    elbow(col(0, 2), b2, col(0, 3), t3, colour=BLUE)      # sentiment score -> SSG
    elbow(col(1, 2), b2, col(1, 3), t3, colour=BLUE)      # aspect polarity -> ASCS
    # ISG reads the review text, not either model, so it is fed from the corpus.
    y_src, y_isg = (t1 + b1)/2, (t3 + b3)/2
    s.line(CX + CW, y_src, RAIL, y_src, colour=BASE)
    s.line(RAIL, y_src, RAIL, y_isg, colour=BASE)
    s.arrow(RAIL, y_isg, CX + CW, y_isg, colour=BASE)

    t_cii = b3 - 0.034
    bus([col(0, 3), col(1, 3), col(2, 3)], [CX + CW/2], b3, t_cii, colour=ORANGE)
    b_cii = s.box(CX + 0.112, t_cii, CW - 0.224,
                  'Composite Inconsistency Index\n'
                  'CII = w\u2081\u00b7SSG + w\u2082\u00b7ASCS + w\u2083\u00b7ISG\n'
                  'weights fitted by grid search to 0.05, 0.95 and 0.00',
                  fs=6.9, face=PEACH, edge=ORANGE, colour=DEEP_ORANGE)
    chip(t3, b_cii, 3, 'Composite\nInconsistency\nIndex', 'index')

    # ---- Phase 4: detection ---------------------------------------------
    t4 = b_cii - GAP
    b4 = row(t4, ['Random Forest\n22 features in five families \u2014 length,\n'
                  'readability, lexical, behavioural,\nand the index with its components',
                  'P(suspicious)\nper review'], 'model')
    chip(t4, b4, 4, 'Authenticity\ndetection', 'model')
    elbow(CX + CW/2, b_cii, col(0, 2), t4, colour=ORANGE)

    # ---- Phase 5: aggregation and delivery -------------------------------
    t5 = b4 - GAP
    b5 = row(t5, ['Trust-weighted aggregation\nweight = 1 \u2212 P(suspicious), held above 0.02,\n'
                  'applied overall and per aspect',
                  'Web application\na product link, its reviews fetched live,\n'
                  'the rating beside its trust-weighted value'], 'out')
    chip(t5, b5, 5, 'Trust-weighted\naggregation', 'out')
    elbow(col(1, 2), b4, col(0, 2), t5, colour=AQUA)

    y_note = b5 - 0.042
    s.note(CX + CW/2, y_note,
           'Each phase consumes the output of the one above it; the intensity '
           'signal alone is read directly from the review text.')
    s.crop(y_note - 0.014)
    s.verify('architecture')
    return s.save('17_architecture')


# ----------------------------------------------------------------- CII design
def fig_cii_design():
    s = Sheet(7.0, 3.5)
    s.ax.text(0.5, 0.955, 'Composite Inconsistency Index', ha='center',
              fontsize=10.5, fontweight='700', color=INK)
    s.ax.text(0.5, 0.890, 'CII  =  w₁·SSG  +  w₂·ASCS  +  w₃·ISG',
              ha='center', fontsize=9, color=INK2, family='monospace')

    comps = [
        ('SSG\nSentiment–Star Gap',
         'how far the review’s overall\nsentiment sits from the\nsentiment that star rating\nusually carries',
         'w₁ = 0.05', False),
        ('ASCS\nAspect–Star Conflict',
         'the share of detected aspects\nwhose polarity opposes the\ndirection of the star rating',
         'w₂ = 0.95', False),
        ('ISG\nIntensity–Star Gap',
         'how far the emphasis in the\nwording sits from the\nemphasis usual at that rating',
         'w₃ = 0.00', True),
    ]
    W, GX = 0.300, 0.020
    top = 0.820
    head_h = max(s.text_h(c[0], 8.2) for c in comps)
    body_h = max(s.text_h(c[1], 6.9) for c in comps)
    for i, (title, detail, w, dim) in enumerate(comps):
        x = 0.030 + i*(W + GX)
        edge = BASE if dim else BLUE
        s.box(x, top, W, title, fs=8.2, face=SURFACE if dim else PALE,
              edge=edge, colour=MUTED if dim else DEEP_BLUE, weight='600', h=head_h)
        s.box(x, top - head_h - 0.012, W, detail, fs=6.9,
              colour=MUTED if dim else INK2, h=body_h)
        wy = top - head_h - body_h - 0.055
        s.ax.text(x + W/2, wy, w, ha='center', va='center', fontsize=8.2,
                  fontweight='700', family='monospace',
                  color=MUTED if dim else ORANGE)
        s.arrow(x + W/2, wy - 0.028, x + W/2, wy - 0.028 - GAP,
                colour=GRID if dim else MUTED)

    top = wy - 0.028 - GAP
    bot = s.box(0.175, top, 0.650,
                'weighted sum, renormalised over\nwhichever components a review has',
                fs=7.4, face=PEACH, edge=ORANGE, colour=DEEP_ORANGE)
    s.arrow(0.5, bot, 0.5, bot - GAP, colour=ORANGE)
    bot = s.box(0.085, bot - GAP, 0.830,
                'Random Forest\nthe index and its three components enter as four of 22 features',
                fs=7.6, face=PALE, edge=BLUE, colour=DEEP_BLUE, weight='600')
    s.note(0.5, bot/2, 'The weights were fitted by grid search. ISG was fitted to zero.')
    return s.save('18_cii_design')


# ------------------------------------------------------------ data preparation
def fig_data_prep():
    s = Sheet(7.0, 2.9)
    stages = [
        ('2,506\n\nreviews scraped from three\nproduct categories', BASE, SURFACE, INK2),
        ('2,017\n\nselected for annotation and\nlabelled on four dimensions', BLUE, PALE, DEEP_BLUE),
        ('1,977\n\nanalysed, after exclusions', AQUA, MINT, DEEP_GREEN),
    ]
    W, GX = 0.290, 0.045
    top = 0.960
    h = max(s.text_h(t[0], 7.6) for t in stages)
    for i, (text, edge, face, colour) in enumerate(stages):
        x = 0.035 + i*(W + GX)
        s.box(x, top, W, text, fs=7.6, face=face, edge=edge, colour=colour,
              lw=1.2, h=h)
        if i < len(stages) - 1:
            s.arrow(x + W, top - h/2, x + W + GX, top - h/2)
    bot = top - h

    s.arrow(0.515, bot, 0.515, bot - GAP)        # corrections act on the 2,017
    s.arrow(0.700, bot, 0.700, bot - GAP, colour='#c47b74')
    left = s.box(0.035, bot - GAP, 0.380,
                 'Corrections applied before use\n\n'
                 'two aspect names respelled, two\n“Fake” labels mapped to\n'
                 '“Suspicious”, 18 blanks filled',
                 fs=7.0)
    right = s.box(0.450, bot - GAP, 0.515,
                  'Excluded (40)\n\n'
                  '34 with no review text  ·  3 in Tamil script\n'
                  '2 romanised Tamil  ·  1 whose text a\nspreadsheet turned into a date',
                  fs=7.0, face=ROSE, edge='#f5c2be', colour=DEEP_RED)
    bot = min(left, right)

    s.arrow(0.230, bot, 0.230, bot - GAP)
    s.arrow(0.740, bot, 0.740, bot - GAP, colour=ORANGE)
    a = s.box(0.035, bot - GAP, 0.420,
              'Grouped stratified split\n\n'
              'reviews sharing a normalised text stay\ntogether, so no wording appears on\n'
              'both sides  ·  1,581 train, 396 test',
              fs=7.0, face=PALE, edge=BLUE, colour=DEEP_BLUE)
    b = s.box(0.490, bot - GAP, 0.475,
              'A confound carried forward\n\n'
              'the authenticity label tracks length\n(0.951 AUC from character count alone),\n'
              'so length is a baseline to beat, never\nevidence for the index',
              fs=7.0, face=PEACH, edge=ORANGE, colour=DEEP_ORANGE)
    return s.save('19_data_preparation')


if __name__ == '__main__':
    fig_architecture()
    fig_cii_design()
    fig_data_prep()
