# TrustLens gold dataset — v1

**File:** `annotated_reviews_gold_v1.csv` · **Rows:** 1,977
**Built by:** `scripts/build_gold_dataset.py` — re-run it to reproduce this file exactly.

## Provenance

All label values come from **`Labelled_Verified Dataset.xlsx`**, the manual annotation.

`annotated_reviews_v2.csv` was **not** used as a label source. It was produced by an AI assistant
from that spreadsheet and altered 147 authenticity labels, 56 language, 49 aspect and 12 sentiment
values. Those are model output, not annotation. Only v2's *structure* was adopted: the `review_id`
scheme and the column layout.

`review_id` is the row's own number in the source spreadsheet (header = row 1), so `R0299` is
spreadsheet row 299. Every row traces back to the annotation by eye.

**The script never overwrites an annotated value.** It removes out-of-scope rows, fixes spelling
and case, fills blanks where a structural rule applies, and adds derived columns.

## What the build changed

| Change | Count | Why |
|---|---|---|
| Removed: empty `review_text` | 34 | No text to analyse |
| Removed: Tamil script | 3 | Outside the English / Sinhala / Singlish scope |
| Removed: romanised Tamil (R1142, R1311) | 2 | Same reason; Latin script so the rule missed them |
| Removed: R1845 | 1 | Not a review — a date value leaked into the cell |
| `authenticity`: `Fake` → `Suspicious` | 1 | One row used a different word for the same class |
| Aspect spelling `color`→`Color/Design`, `size`→`Size/Fit` | 10 | Same label, inconsistent case |
| `aspects_merged` column added | — | 11 categories collapsed to 7, in a separate column |
| `Other` aspect dropped, merged column only | 2 | Not a usable category |
| `language_type` set to `Non-verbal` where blank | 23 | Emoji-only reviews carry no language |
| Manual label fixes applied from `manual_label_fixes.csv` | 42 | Completes every remaining blank |

Removed rows and their reasons are in `removed_rows.csv`.

## The `Non-verbal` decision

27 reviews carry no linguistic content — 22 emoji only (`👍`, `❤️`), the rest keyboard mash. They
are kept, not deleted, and marked two ways: `text_type = non-verbal`, and `language_type =
Non-verbal` **where that field was blank**. Four rows already had a language annotated (R0029 as
Mixed; R1399, R1590, R1829 as English) and were left untouched.

Filter on `text_type == 'linguistic'` (1,950 rows) for the language and sentiment models. Keep all
1,977 for fake-review detection.

## The `Not applicable` sentiment decision

33 reviews were annotated `sentiment_polarity = Not applicable`: 26 non-verbal (emoji, keyboard
mash) and 7 linguistic but contentless (`Amazon`, `#win`, `free gift`, `Colors`, `A +++`, `bnn`,
`Gd`). Their text carries no sentiment that can be assigned, so this is a deliberate annotation
decision, not missing data — and recording it explicitly keeps it distinguishable from work not
yet done.

They are excluded from sentiment training (`data.sentiment_data()` drops them) and retained for
fake-review detection, where they are legitimate examples. `Not applicable` is a data value, never
a model class: `config.SENTIMENTS` still holds only Positive, Neutral and Negative.

**The dataset now has no blank labels.** All 1,977 rows are fully annotated.

## Columns

| Column | Meaning |
|---|---|
| `review_id` | `R` + source spreadsheet row number. Stable, unique |
| `category` | electronics / fashion / grocery |
| `product_name`, `date`, `variant`, `helpful_count` | As scraped |
| `verified_purchase` | **Constant `True`.** Unusable as a feature |
| `star_rating` | 1–5 |
| `review_text` | As scraped, unmodified |
| `language_type` | English / Sinhala / Singlish / Mixed / Non-verbal |
| `sentiment_polarity` | Positive / Negative / Neutral (37 blank) |
| `aspects` | Annotated aspects, spelling normalised |
| `aspects_merged` | Collapsed to 7 categories. Use this for modelling |
| `authenticity` | Genuine / Suspicious |
| `text_type` | `linguistic` (1,950) or `non-verbal` (27) |
| `n_chars` | Review length. **Baseline to beat, never a feature** |
| `dup_count` | How many rows share this exact text |
| `group_key` | Hash of the normalised review text — the split groups on this |
| `has_labels` | `yes` for every row — kept as a guard for future edits |

## Distribution

| Field | Values |
|---|---|
| Language | English 1,536 · Sinhala 226 · Singlish 170 · Non-verbal 23 · Mixed 22 |
| Sentiment | Positive 1,545 · Negative 239 · Neutral 160 · Not applicable 33 |
| Authenticity | Genuine 1,540 · Suspicious 437 |
| Stars | 5★ 1,735 · 1★ 96 · 4★ 80 · 3★ 51 · 2★ 18 |
| Merged aspects | Product Quality 1,107 · Delivery/Shipping 279 · Color/Design 178 · Price/Value 163 · Seller Service 153 · Size/Fit 104 · Packaging 67 |

Fully labelled: **1,977 of 1,977**. Distinct text groups: 1,611.
Usable for sentiment training: **1,943** (linguistic, excluding Not applicable).

## Re-running with new labels

`manual_label_fixes.csv` holds the label completions, applied by the build script. It fills
**blanks only** and never overwrites an annotated value, and unknown values are rejected rather
than silently creating a new class. To correct something already annotated, edit
`Labelled_Verified Dataset.xlsx` and rebuild — that file remains the single source of truth.

```bash
python scripts/build_gold_dataset.py
python -m trustlens.splits
```

## Known limitations — carry these into the thesis

1. **`verified_purchase` is constant `True`.** The VP-based behavioural feature family from
   Hajek et al. (2023) cannot be computed. Drop the family and say so.
2. **`authenticity` correlates very strongly with review length** — AUC 0.949 from character count
   alone; a threshold near 11 characters reproduces 93.5% of the labels. Treat length as a
   **baseline to beat**, never as a feature, and report CII's contribution over and above it.
3. **`authenticity` carries two ideas** — very short generic reviews flagged for being generic,
   and a few long ones flagged for contradicting their star rating. The second group overlaps with
   what CII measures, which risks circularity in the ablation.
4. **33 products, appearing on both sides of the split.** Grouping on product as well as text is
   not possible: generic strings such as "Good" occur on nearly every product, so the product+text
   graph collapses into one component holding 95.5% of the corpus and no test set can be carved
   out. Text leakage is prevented; product overlap is not.
   `trustlens.splits.build_product_holdout()` gives a separate unseen-product test set
   (13 products, 366 rows) as a robustness check for Chapter 5.
5. **Median review length is 32 characters.** Readability, POS and complexity features will be
   weak by construction.
