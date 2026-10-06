"""Paths, random seed and label vocabularies. Import from here - never hardcode."""
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[2]
DATA      = ROOT/'Dataset'
GOLD      = DATA/'Gold'/'annotated_reviews_gold_v1.csv'
PROCESSED = DATA/'Processed'          # feature matrices, splits (gitignored)
# Model artefacts are namespaced by scikit-learn version. Pickles are not portable
# between versions, and this project is worked on from more than one environment -
# without the namespace each machine silently invalidates the other's models.
# Everything under models/ is gitignored; these are build artefacts, regenerated
# in seconds from source.
import sklearn as _sklearn
SKLEARN_TAG = 'sk' + '.'.join(_sklearn.__version__.split('.')[:2])
MODELS    = ROOT/'models'/SKLEARN_TAG
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


def load_model(path, retrain_cmd: str):
    """joblib.load with a readable error when the pickle came from another sklearn.

    scikit-learn does not guarantee that a model pickled by one version loads in
    another; it usually fails deep inside predict() with a confusing
    AttributeError. Models are gitignored precisely because they are build
    artefacts - regenerate rather than ship them.
    """
    import warnings, joblib
    from sklearn.exceptions import InconsistentVersionWarning
    if not path.exists():
        raise FileNotFoundError(f'{path} missing - run: {retrain_cmd}')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', InconsistentVersionWarning)
        model = joblib.load(path)
    if any(issubclass(w.category, InconsistentVersionWarning) for w in caught):
        import sklearn
        raise RuntimeError(
            f'{path.name} was saved by a different scikit-learn version '
            f'(this environment has {sklearn.__version__}). Pickled models are not '
            f'portable across versions. Regenerate it with:  {retrain_cmd}')
    return model
