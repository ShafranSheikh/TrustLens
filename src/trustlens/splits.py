"""Build the train/test split once, save it, and reuse it everywhere.

Three properties of this corpus break a plain random split:

  * 33 products, one with 200 reviews  -> the model learns the product
  * 366 rows share text with another   -> the same string lands on both sides
  * 87% five-star, 79% positive        -> a fold can end up with no negatives

So whole GROUPS of identical text move together, and the split is chosen to match
class proportions between the two sides.

Why this does not use sklearn's StratifiedGroupKFold
----------------------------------------------------
Two reasons. First, it gave materially different splits on scikit-learn 1.7 and
1.9 from the same seed, which makes a thesis result unreproducible across
machines. Second, it cannot balance this corpus well: "Good" is a single group of
132 rows, all Suspicious, against 437 Suspicious in total - one group holds 30% of
the class, so equal folds are impossible and the best it managed was a 13
percentage-point gap in the Suspicious rate.

Instead the test set is selected directly: repeated randomised greedy passes, each
adding the group that best preserves class proportions, keeping the best of many
restarts. The only randomness is numpy's own Generator, seeded here, so the result
is identical on any machine and any scikit-learn version. It reaches gaps of about
0.1 percentage points.

Build it:   python -m trustlens.splits
Use it:     from trustlens import splits
            df = splits.attach(df)          # adds 'split' and 'fold'
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from . import config, data

PATH = config.PROCESSED/'splits.json'
PRODUCT_PATH = config.PROCESSED/'product_holdout.json'

# Proportions held level between the two sides of every split.
BALANCE_TARGETS = [('authenticity', 'Suspicious'),
                   ('sentiment_polarity', 'Negative'),
                   ('sentiment_polarity', 'Neutral'),
                   ('sentiment_polarity', 'Positive')]
TEST_RESTARTS, FOLD_RESTARTS, CANDIDATES = 150, 25, 60


def _group_arrays(df):
    """Per-group row counts and per-group counts of each balanced class."""
    uniq, inv = np.unique(df['group_key'].values, return_inverse=True)
    gsize = np.bincount(inv, minlength=len(uniq)).astype(float)
    gflag = np.stack([np.bincount(inv, weights=(df[c] == v).values.astype(float),
                                  minlength=len(uniq))
                      for c, v in BALANCE_TARGETS])
    return uniq, inv, gsize, gflag


def _score(sel_rows, sel_flag, pool_rows, pool_flag, frac):
    """Lower is better: size error, then the class-proportion gap."""
    rest_rows = pool_rows - sel_rows
    if sel_rows <= 0 or rest_rows <= 0:
        return 1e9
    a = sel_flag/sel_rows
    b = (pool_flag - sel_flag)/rest_rows
    return 5*abs(sel_rows/pool_rows - frac) + float(np.abs(a - b).sum())


def _carve(pool, gsize, gflag, frac, seed_base, restarts):
    """Choose a subset of `pool` holding `frac` of its rows, matching the rest.

    Randomised greedy with restarts. Deterministic: numpy Generator only.
    """
    pool = np.asarray(pool)
    pool_rows = float(gsize[pool].sum())
    pool_flag = gflag[:, pool].sum(axis=1)
    target = frac*pool_rows
    best = None
    for r in range(restarts):
        rng = np.random.default_rng(seed_base + r)
        taken, rows, flag = [], 0.0, np.zeros(gflag.shape[0])
        avail = pool.copy()
        while rows < target and len(avail):
            room = max(target - rows, 1.0)*1.6
            ok = avail[gsize[avail] <= room]
            if not len(ok):
                break
            pick = rng.choice(ok, size=min(CANDIDATES, len(ok)), replace=False)
            costs = [_score(rows + gsize[i], flag + gflag[:, i],
                            pool_rows, pool_flag, frac) for i in pick]
            chosen = pick[int(np.argmin(costs))]
            taken.append(chosen); rows += gsize[chosen]; flag += gflag[:, chosen]
            avail = avail[avail != chosen]
        c = _score(rows, flag, pool_rows, pool_flag, frac)
        if best is None or c < best[0]:
            best = (c, r, np.array(taken, dtype=int))
    return best          # (score, restart index, group indices)


def _assign_folds(pool, gsize, gflag, k, seed_base, restarts):
    """Assign every group in `pool` to one of k folds, all folds at once.

    Carving folds off one at a time starves the last one: the largest group
    ("Good", 132 rows, every one Suspicious) lands in an early fold and the
    leftover ends up with almost no positives - a 40 point spread in the
    Suspicious rate. Assigning jointly keeps the spread to what the data forces.

    That floor is real: 132 rows in one group against 349 Suspicious in training
    means whichever fold holds it is about 42% Suspicious whatever we do. The
    others can still be held level with each other.
    """
    pool = np.asarray(pool)
    order_sizes = gsize[pool]
    class_total = np.maximum(gflag[:, pool].sum(axis=1), 1)
    best = None
    for r in range(restarts):
        rng = np.random.default_rng(seed_base + r)
        idx = rng.permutation(len(pool))
        idx = idx[np.argsort(-order_sizes[idx], kind='stable')]   # big first, shuffle breaks ties
        counts = np.zeros((k, gflag.shape[0]))
        sizes = np.zeros(k)
        assign = np.empty(len(pool), dtype=int)
        for j in idx:
            gi = pool[j]
            costs = []
            for f in range(k):
                c = counts.copy(); c[f] += gflag[:, gi]
                z = sizes.copy(); z[f] += gsize[gi]
                costs.append((np.std(c, axis=0)/class_total).sum() + 1e-3*np.std(z)/max(z.mean(), 1))
            f = int(np.argmin(costs))
            counts[f] += gflag[:, gi]; sizes[f] += gsize[gi]; assign[j] = f
        rates = counts/np.maximum(sizes[:, None], 1)
        spread = float((rates.max(axis=0) - rates.min(axis=0)).sum())
        if best is None or spread < best[0]:
            best = (spread, r, assign.copy())
    return best


def build(force: bool = False) -> dict:
    """Compute the split and write it to disk. Idempotent unless force=True."""
    if PATH.exists() and not force:
        print(f'{PATH.name} already exists - use build(force=True) to recompute')
        return load()

    df = data.load_gold()
    uniq, inv, gsize, gflag = _group_arrays(df)
    all_groups = np.arange(len(uniq))

    score, restart, test_groups = _carve(all_groups, gsize, gflag,
                                         config.TEST_SIZE, config.SEED, TEST_RESTARTS)
    is_test = np.isin(inv, test_groups)

    # cross-validation folds, all assigned together
    train_groups = np.setdiff1d(all_groups, test_groups)
    spread, _, fold_of = _assign_folds(train_groups, gsize, gflag, config.N_FOLDS,
                                       config.SEED + 9000, FOLD_RESTARTS)
    folds = {}
    for f in range(config.N_FOLDS):
        sel = train_groups[fold_of == f]
        folds[str(f)] = sorted(df['review_id'][np.isin(inv, sel)].tolist())

    out = {
        'method': 'randomised greedy group selection, best of N restarts; numpy '
                  'Generator only, so the result does not depend on the '
                  'scikit-learn version',
        'seed': config.SEED,
        'test_restarts': TEST_RESTARTS, 'fold_restarts': FOLD_RESTARTS,
        'best_restart': restart,
        'balance_score': round(score, 4),
        'fold_note': 'one group of 132 identical Suspicious reviews ("Good") is 38% of the '
                     'training Suspicious class, so the fold holding it is about 42% '
                     'Suspicious however folds are drawn. Report per-fold results, not just '
                     'the mean.',
        'fold_rate_spread': round(spread, 4),
        'balanced_on': [f'{c}={v}' for c, v in BALANCE_TARGETS],
        'grouped_on': 'group_key (normalised review_text)',
        'product_overlap': 'products appear on both sides - unavoidable, see '
                           'build_gold_dataset.py. Checked separately by build_product_holdout().',
        'test_size': config.TEST_SIZE, 'n_folds': config.N_FOLDS,
        'n_total': len(df), 'n_train': int((~is_test).sum()), 'n_test': int(is_test.sum()),
        'test': sorted(df['review_id'][is_test].tolist()),
        'folds': folds,
    }
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(out, indent=2))
    print(f'wrote {PATH.name}  (balance {score:.4f}, test {out["n_test"]} rows '
          f'= {100*out["n_test"]/len(df):.1f}%)')
    return out


def load() -> dict:
    if not PATH.exists():
        raise FileNotFoundError(f'{PATH} missing - run: python -m trustlens.splits')
    return json.loads(PATH.read_text())


def attach(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'split' ('train'/'test') and 'fold' (0-4, <NA> for test rows)."""
    s = load()
    test = set(s['test'])
    fold_of = {rid: int(k) for k, ids in s['folds'].items() for rid in ids}
    df = df.copy()
    df['split'] = np.where(df['review_id'].isin(test), 'test', 'train')
    df['fold'] = df['review_id'].map(fold_of).astype('Int64')
    return df


def report() -> pd.DataFrame:
    """Label balance across train/test - paste this into the thesis."""
    df = attach(data.load_gold())
    rows = []
    for col in ['authenticity', 'sentiment_polarity', 'language_type']:
        for split in ['train', 'test']:
            sub = df[df['split'] == split]
            for val, n in sub[col].value_counts().items():
                rows.append({'field': col, 'value': val or '(blank)', 'split': split,
                             'n': n, 'pct': round(100*n/len(sub), 1)})
    return pd.DataFrame(rows)


def build_product_holdout(target_frac: float = 0.15, force: bool = False) -> dict:
    """A second, separate split that holds out WHOLE products.

    The main split lets products appear on both sides - unavoidable, see the note
    in build_gold_dataset.py. This answers the other question: does the system work
    on a product it has never seen? That is what happens at deployment, when the
    app is handed an arbitrary product URL.

    A robustness check for Chapter 5, not the primary evaluation - it is small and
    its class balance cannot be controlled.
    """
    if PRODUCT_PATH.exists() and not force:
        return json.loads(PRODUCT_PATH.read_text())

    df = data.load_gold()
    rng = pd.Series(df['product_name'].unique()).sample(frac=1, random_state=config.SEED)
    sizes = df['product_name'].value_counts()
    cat_of = df.groupby('product_name')['category'].first()

    held, n, seen = [], 0, set()
    for prod in rng:
        if n >= target_frac*len(df) and seen >= set(df['category'].unique()):
            break
        if sizes[prod] > 0.5*target_frac*len(df) and n > 0:
            continue                      # skip the giants, they swamp the holdout
        held.append(prod); n += sizes[prod]; seen.add(cat_of[prod])

    ids = sorted(df[df['product_name'].isin(held)]['review_id'].tolist())
    out = {'seed': config.SEED, 'n_products_held': len(held), 'n_rows_held': len(ids),
           'frac': round(len(ids)/len(df), 3), 'products': sorted(held), 'test': ids}
    PRODUCT_PATH.write_text(json.dumps(out, indent=2))
    print(f'wrote {PRODUCT_PATH.name}  ({len(held)} products, {len(ids)} rows)')
    return out


if __name__ == '__main__':
    build(force=True)
    build_product_holdout(force=True)
