"""Train/test split and cross-fitting folds for the labeled clients."""

import numpy as np
import pandas as pd

from x5uplift.config import ID_COL, N_FOLDS, SEED, TARGET_COL, TEST_SHARE, TREATMENT_COL


def make_splits(labeled: pd.DataFrame, test_share=TEST_SHARE, n_folds=N_FOLDS, seed=SEED) -> pd.DataFrame:
    """Assign each labeled client to the test set or to one of n_folds training folds.

    The split is stratified on treatment x target, so every part keeps the same
    shares of the four cells. Rows are sorted by id first, so the result does not
    depend on the input order.
    Returns: id, treatment, target, is_test (bool), fold (0..n_folds-1, or -1 for test).
    """
    rng = np.random.default_rng(seed)
    df = labeled[[ID_COL, TREATMENT_COL, TARGET_COL]].sort_values(ID_COL).reset_index(drop=True)
    df["is_test"] = False
    df["fold"] = -1

    for _, idx in df.groupby([TREATMENT_COL, TARGET_COL]).indices.items():
        idx = rng.permutation(idx)
        n_test = int(round(len(idx) * test_share))
        df.loc[idx[:n_test], "is_test"] = True
        train_idx = idx[n_test:]
        df.loc[train_idx, "fold"] = np.arange(len(train_idx)) % n_folds

    return df