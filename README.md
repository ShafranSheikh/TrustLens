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
| `03_cii` | SSG / ASCS / ISG distributions, weight tuning | 4.6 Algorithms/Models · 5.4 Experimental Results |
| `04_fake_detection` | Detector results, feature importances | 5.4 Experimental Results · 5.6 Technical Measures |
| `05_absa` | Per-aspect F1 with support counts | 5.6 Technical Measures |
| `06_ablation` | CII contribution over the length baseline | **5.10 Results Related to Research Objectives** |
| `app/` | Working prototype | 4.9 Implementation · 5.7–5.8 User & Usability Evaluation |
