"""Deployment-time sentiment correction for the web application.

WHAT THIS IS AND IS NOT
-----------------------
This is NOT part of the trained system. Nothing in `trustlens/` uses it, and no
figure or table in the thesis is produced with it. It applies only inside the web
application, and the thesis must say so.

WHY IT EXISTS
-------------
The sentiment model learns from 1,554 reviews, and the strongest negative words
barely occur in them: "worst" appears five times, "stopped" three, "terrible"
once. So a review such as "Worst experience ever, never arrived and the seller
ignored me" attached to a five-star rating scores +0.79 - read as positive - and
the index therefore records no inconsistency at all. That is the single most
obvious case a person will try in a demonstration.

THE TRADE-OFF, MEASURED
-----------------------
On the 389-review test set, thresholding the raw score gives macro-F1 0.776;
adding this correction gives 0.747. It moves 12.6% of reviews and the model was
often right. Corpus accuracy is worse.

It is applied anyway, in the application only, because the corpus contains very
few five-star-on-a-complaint reviews - eight in total - so corpus accuracy
under-weights exactly the failure this corrects. Reported model performance
stays as measured, without the lexicon.

The approach follows Rizvi et al. (2025), arXiv 2504.13545, who report domain
lexicon correction adding 10.2% accuracy on Sinhala, Singlish and code-mixed
banking reviews. It does not transfer as a gain here, which is worth reporting.
"""
from __future__ import annotations
import re
import numpy as np

STRENGTH = 0.6

NEGATIVE = [
    'worst', 'terrible', 'horrible', 'awful', 'useless', 'rubbish', 'garbage',
    'trash', 'scam', 'fake', 'fraud', 'cheat', 'cheated', 'waste', 'refund',
    'damaged', 'broken', 'defective', 'faulty', 'disappointed', 'disappointing',
    'poor', 'worse', 'ruined', 'not working', 'stopped working', 'never arrived',
    'not received', 'not as described', 'do not buy', 'dont buy', 'never buy',
    'not recommend', 'waste of money',
    # Singlish and Sinhala
    'naraka', 'epa', 'kedila', 'boru', 'වැඩක් නෑ',
    'නරකයි', 'එපා', 'කැඩිලා', 'අවුලක්', 'බොරු',
]
POSITIVE = [
    'excellent', 'perfect', 'amazing', 'wonderful', 'fantastic', 'love it',
    'highly recommend', 'very good', 'great quality', 'superb',
    'hodai', 'niyamai', 'supiri', 'හොඳයි', 'සුපිරි', 'නියමයි',
]


def _compile(words):
    return re.compile(r'(?<!\w)(' + '|'.join(re.escape(w) for w in words) + r')(?!\w)',
                      re.IGNORECASE)


NEG_RE, POS_RE = _compile(NEGATIVE), _compile(POSITIVE)


def correct(scores, texts, strength: float = STRENGTH) -> np.ndarray:
    """Pull the sentiment score toward the lexicon when the model disagrees with it.

    Where the lexicon and the model point opposite ways the lexicon wins outright;
    where they agree it nudges. Where the lexicon finds nothing the score is
    untouched, which is most reviews.
    """
    s = np.asarray(scores, dtype=float)
    texts = [str(t) for t in texts]
    neg = np.array([len(NEG_RE.findall(t)) for t in texts], dtype=float)
    pos = np.array([len(POS_RE.findall(t)) for t in texts], dtype=float)
    pull = strength*(np.tanh(pos) - np.tanh(neg))
    out = np.where(pull == 0, s,
                   np.where(np.sign(pull) != np.sign(s), pull, s + 0.4*pull))
    return np.clip(out, -1.0, 1.0)


def hits(text: str) -> dict:
    """Which lexicon terms fired - so the interface can explain itself."""
    t = str(text)
    return {'negative': sorted({m.lower() for m in NEG_RE.findall(t)}),
            'positive': sorted({m.lower() for m in POS_RE.findall(t)})}
