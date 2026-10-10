# TrustLens

Trust-aware, cross-lingual review analysis for Sri Lankan e-commerce.
BSc (Hons) Computing final-year project.

## Layout

| Folder | Holds |
|---|---|
| `Dataset/RawDataset/` | Scraped reviews, unmodified |
| `Dataset/LabelledDataset/` | Manual annotation, as annotated |
| `Dataset/Gold/` | **The dataset everything uses.** See `Dataset/Gold/GOLD_DATASET.md` |
| `Dataset/Processed/` | Feature matrices and splits (generated, not committed) |
| `src/trustlens/` | All real logic. Notebooks and the app import from here |
| `notebooks/` | One per phase — the evidence trail for the viva |
| `app/` | Flask web application (Phase 5) |
| `models/` | Trained models (generated, not committed) |
| `outputs/figures/` | 300 DPI PNGs for the thesis |
| `outputs/tables/` | Results tables as CSV, ready to paste into the thesis |
| `scripts/` | One-off build scripts |
| `tests/` | Unit tests, mainly for the CII components |

## Setup — macOS, once

Run these in Terminal from the `TrustLens` folder.

```bash
cd ~/Personal/"Lecture notes"/"Bsc(HONS) Computing- 4th Year"/"Project Dissertation"/Development/TrustLens

python3 -m venv .venv                       # create the environment
source .venv/bin/activate                   # you should now see (.venv) in your prompt
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .                            # makes `from trustlens import ...` work anywhere
```

Check it worked:

```bash
python -c "import trustlens, sklearn; print(trustlens.__version__, sklearn.__version__)"
```

`sklearn` must be 1.4 or higher — below that the tree models cannot handle missing values,
which the Sinhala/Singlish feature strategy depends on.

## Every session

```bash
cd .../TrustLens
source .venv/bin/activate
```

Everything below assumes the environment is active. `deactivate` when you are done.

## Commands

| What | Command |
|---|---|
| Rebuild the gold dataset from the annotation | `python scripts/build_gold_dataset.py` |
| Build the train/test split | `python -m trustlens.splits` |
| Run the tests | `pytest -q` |
| Start Jupyter (once installed) | `pip install jupyterlab && jupyter lab` |

Both build commands are safe to re-run — they overwrite their own output and nothing else.

## Running the web application

```bash
source .venv/bin/activate
pip install selenium webdriver-manager      # once, for live fetching
python app/app.py
```

Then open <http://127.0.0.1:5000>. Models load once at startup, about ten seconds;
nothing retrains on a request.

The page asks for one thing: a product link. Every review on that product is read,
page by page, then run through the pipeline. A scrape runs on a background thread
and the page polls it, so the progress indicator shows real pages and counts rather
than an indefinite spinner - and a long scrape cannot time the request out.

| Route | What it is |
|---|---|
| `/` | paste a product link, fetch live |
| `/demo` | offline fallback: a product from the annotated corpus, or reviews typed by hand |
| `/health` | readiness check |

`/demo` is not linked from the front page, because the front page is meant to show
one idea. Keep it in your pocket for a demonstration: it needs neither the network
nor Chrome, so it cannot fail on someone else's wifi.

### Reading more than one platform

Extraction is layered, and the result page names the layer that produced the rows:

1. **A site profile** - hand-written selectors for a platform we have read, driving
   real pagination. One profile ships, for the platform the corpus came from.
2. **Structured data** - schema.org `Product.review`, as JSON-LD or microdata. This
   is what a search engine reads to show review stars beside a shop's result, so a
   shop that wants those stars publishes it whatever it is built with. This is the
   layer that reads sites nobody wrote code for.
3. **A generic reader** - repeated blocks pairing a star widget with a paragraph.
   Least reliable; the page says so when it is used.

Reviews are often absent from the page until a **Reviews** tab is clicked, and
only partly there until **Load more** is pressed. Both happen automatically before
the readers run - matching by whole label and skipping links that would navigate
away, so it cannot wander into "Buy now".

Before trusting a link in a demonstration, probe it:

```bash
python -m trustlens.review_scraper "https://…/some-product"
python -m trustlens.review_scraper URL1 URL2 URL3          # one browser, many links
python -m trustlens.review_scraper --file candidates.txt   # a list, with notes
python -m trustlens.review_scraper "https://…" --show      # watch Chrome work
```

It reports which layer fires, how many reviews it would return, which tab it
opened, and prints the first review so the extraction can be eyeballed.
`candidates.txt` holds the shortlist; edit it as you find links that work.

"Nothing readable" usually means the product genuinely has no reviews, or they
arrive from a separate request after load. Try a second, more popular product on
the same site before writing the site off.

### If a fetch fails

* Set `TRUSTLENS_SHOW_BROWSER=1` to watch Chrome work. A site that refuses a
  headless browser often accepts a visible one, and it demonstrates well.
* The first run downloads a matching ChromeDriver. Chrome itself must be installed.
* Safety stops: 200 pages, 3000 reviews, five minutes. The result says when one was
  hit.

**The application applies one thing the measured system does not:** a deployment-time
sentiment lexicon (`app/lexicon.py`). See that file for the measured trade-off. Nothing
in `trustlens/` uses it and no reported result includes it.

The sentiment column shows a class, but the model produces a score. The cut points
(>0.100 Positive, <-0.275 Negative) were chosen to reproduce the model's own argmax
label, which they do on 98.9% of the corpus.

### Charts

Drawn as inline SVG on the server (`app/charts.py`) - no charting library and no CDN,
so a results page cannot be broken by bad conference wifi. Colours match
`trustlens.eda`, so a chart on screen is the same chart as the corresponding figure
in Chapter 4. `outputs/app_example.html` is a saved result page, openable offline.

## Comparison with existing solutions (Chapter 5.9)

```bash
pip install vaderSentiment
python -m trustlens.compare
```

Writes tables 23-27 and figures 15-16. Everything is scored on the same held-out
test split as the rest of the evaluation; nothing is retrained.

**Sentiment.** Against the majority class, VADER (the standard English lexicon
baseline) and the same classifier on word features. Reported per language,
because that is where the answer is: VADER returns a compound of exactly zero -
no lexicon match at all - on 57.5% of Sinhala and 66.7% of Singlish reviews, so
on those languages it scores at or below the majority class. Character n-grams
against word features is a tie the test set cannot resolve (+0.002 overall,
+0.059 on Singlish, both intervals straddling zero); the choice is defended on
the grounds in Chapter 3, not on this result.

**Detection.** Against the star rating alone and review length alone, and
against the pipeline with the CII family removed. Differences are tested by
*paired* bootstrap - resampling the same rows for both scorers - because the two
scores are predictions about the very same reviews, and comparing two
independent intervals for overlap is far too conservative here.

The headline and the finding differ, and both are reported:

| Reviews | Length alone | Without CII | With CII | Adding CII |
|---|---|---|---|---|
| All (n=396, 88 susp.) | 0.841 | 0.851 | 0.869 | +0.018, CI −0.006 to +0.043 |
| 20+ chars (n=200, 10 susp.) | 0.508 | 0.608 | 0.863 | +0.255, CI +0.095 to +0.429 |

Across every review the improvement is not clear at 95%, because short reviews
dominate and the label is close to a length rule there, so nothing can add much.
On reviews with something actually written in them, length is worth no more than
a coin toss and the index carries the signal. **The 20+ stratum holds only 10
suspicious reviews, so that figure is fragile and must never be quoted without
its n.** Both strata are reported because choosing the flattering one after
seeing the answer would not be a result.

## Using it in a notebook or script

```python
from trustlens import config, data, splits

df = splits.attach(data.load_gold())        # every row tagged 'train'/'test' and a fold
train = df[df.split == 'train']
test  = df[df.split == 'test']

data.sentiment_data()                        # rows usable for the sentiment model
data.fake_data()                             # rows usable for fake detection
data.aspect_matrix(df)                       # one column per aspect
splits.report()                              # train/test label balance, for the thesis
```

Never read the CSV directly — always go through `data`, so filtering rules stay in one place.

## Which output goes into which chapter

| Notebook | Produces | Thesis section |
|---|---|---|
| `01_eda` | Corpus statistics, distribution figures | 4.7 Data Sets / Data Preparation |
| `02_sentiment` | Sentiment model, per-language scores | 4.6 Algorithms/Models · 5.6 Accuracy/Precision/Recall |
| `03_absa` | Per-aspect F1 with support counts | 4.6 · 5.6 Technical Measures |
| `04_cii` | SSG / ASCS / ISG distributions, weight tuning | 4.6 Algorithms/Models · 5.4 Experimental Results |
| `05_fake_detection` | Detector results, feature importances | 5.4 Experimental Results · 5.6 Technical Measures |
| `06_ablation` | CII contribution over the length baseline | **5.10 Results Related to Research Objectives** |
| `app/` | Working prototype | 4.9 Implementation · 5.7–5.8 User & Usability Evaluation |
