"""Shared helpers for the prediction and uplift notebooks (03 to 07)."""

import numpy as np
import pandas as pd


from x5uplift.config import FEATURES_PATH, ID_COL, SEED, SPLITS_PATH


def load_experiment() -> pd.DataFrame:
    """The 200,039 experiment clients with treatment, target, split and all features."""
    splits = pd.read_parquet(SPLITS_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    return splits.merge(features, on=ID_COL, how="left", validate="one_to_one")


def model_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Model inputs: the chosen feature columns, with gender one-hot encoded. Missing values stay NaN."""
    X = df[feature_cols].copy()
    if "gender" in X:
        X = pd.get_dummies(X, columns=["gender"], dtype=float)
    return X


def cross_fit_predict(make_model, X: pd.DataFrame, y: pd.Series, fold: pd.Series, fit_mask: pd.Series) -> np.ndarray:
    """Out-of-fold probabilities for every row with fold >= 0.

    For each fold k, a fresh model is fitted on the rows of the other folds where fit_mask is
    True (for example control clients only) and predicts all rows of fold k. Rows with fold -1
    (the test set) are never used and get NaN.
    """
    pred = np.full(len(X), np.nan)
    for k in sorted(fold[fold >= 0].unique()):
        fit_rows = (fold >= 0) & (fold != k) & fit_mask
        model = make_model().fit(X[fit_rows], y[fit_rows])
        pred[(fold == k).to_numpy()] = model.predict_proba(X[fold == k])[:, 1]
    return pred


def bootstrap_ci(metric, y_true, score, n_boot: int = 300, seed: int = SEED, alpha: float = 0.05):
    """Point estimate and percentile bootstrap interval of metric(y_true, score).

    The same seed gives the same resamples for every call, so intervals of different models are
    paired. score may be 2-dimensional (rows are resampled together), e.g. to bootstrap a
    difference between two models.
    """
    rng = np.random.default_rng(seed)
    y_true, score = np.asarray(y_true), np.asarray(score)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        stats.append(metric(y_true[i], score[i]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return metric(y_true, score), lo, hi