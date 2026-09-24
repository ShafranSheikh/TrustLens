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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuilding the dataset

```bash
python3 scripts/build_gold_dataset.py
```
Reproduces `Dataset/Gold/annotated_reviews_gold_v1.csv` exactly from the manual annotation.

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
