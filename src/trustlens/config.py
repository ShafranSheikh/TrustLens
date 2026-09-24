"""Paths, random seed and label vocabularies. Import from here - never hardcode."""
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
DATA      = ROOT/'Dataset'
GOLD      = DATA/'Gold'/'annotated_reviews_gold_v1.csv'
PROCESSED = DATA/'Processed'          # feature matrices, splits (gitignored)
MODELS    = ROOT/'models'
FIGURES   = ROOT/'outputs'/'figures'  # 300 DPI PNGs for the thesis
TABLES    = ROOT/'outputs'/'tables'   # results tables as CSV
for _p in (PROCESSED, MODELS, FIGURES, TABLES): _p.mkdir(parents=True, exist_ok=True)

SEED = 42                 # the only seed in the project
TEST_SIZE = 0.20
N_FOLDS = 5

LANGUAGES  = ['English', 'Sinhala', 'Singlish', 'Mixed', 'Non-verbal']
SENTIMENTS = ['Positive', 'Neutral', 'Negative']   # the model's classes
# Reviews whose text carries no expressible sentiment (emoji only, keyboard mash,
# bare product words). Annotated as a deliberate decision, not a gap. Excluded
# from sentiment training; retained for fake detection.
SENTIMENT_NA = 'Not applicable'
AUTHENTICITY = ['Genuine', 'Suspicious']
ASPECTS = ['Product Quality', 'Delivery/Shipping', 'Price/Value',
           'Seller Service', 'Color/Design', 'Packaging', 'Size/Fit']

# CII weights - neutral baseline, to be tuned by grid search on the validation folds
CII_WEIGHTS = {'ssg': 1/3, 'ascs': 1/3, 'isg': 1/3}

# Languages for which POS, readability and pretrained embeddings are valid.
# Everything else is left as NaN and handled natively by the tree models.
POS_VALID_LANGUAGES = ['English']
