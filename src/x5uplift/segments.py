"""Customer segmentation on behaviour features (notebook 02).

The segments describe clients in business terms. They are fitted on all clients without the
target, so they can be used later for subgroup analysis (notebook 04) and to describe the
targeted audience (notebook 06).
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from x5uplift.config import SEED

# total_spend is left out on purpose: in logs it equals log(n_baskets) + log(avg_basket_spend),
# so including it would count frequency and basket size twice.
SEGMENT_FEATURES = [
    "days_since_last_purchase",
    "n_baskets",
    "avg_basket_spend",
    "n_distinct_products",
    "share_products_repeated",
    "share_baskets_points_spent",
    "share_baskets_express_received",
    "share_own_brand",
    "share_alcohol",
    "share_weekend_baskets",
]
LOG_FEATURES = ["n_baskets", "avg_basket_spend", "n_distinct_products"]


def prepare_matrix(features: pd.DataFrame) -> np.ndarray:
    """Log transform skewed counts and amounts, fill the few missing shares with the median, standardize."""
    X = features[SEGMENT_FEATURES].copy()
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
    X = X.fillna(X.median())
    return StandardScaler().fit_transform(X)


def choose_k(X: np.ndarray, ks=range(3, 9), n_runs: int = 5, fit_size: int = 100_000,
             eval_size: int = 10_000, seed: int = SEED) -> pd.DataFrame:
    """For each k, report how well separated and how stable the segments are.

    Each run fits k-means on a different random subsample with a different seed and labels the
    same evaluation sample. Stability is the adjusted Rand index (ARI) between the labels of every
    pair of runs: 1 means the runs found exactly the same segments.
    """
    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(len(X), eval_size, replace=False)
    rows = []
    for k in ks:
        labels = []
        for run in range(n_runs):
            fit_idx = rng.choice(len(X), fit_size, replace=False)
            km = KMeans(n_clusters=k, n_init=4, random_state=seed + run).fit(X[fit_idx])
            labels.append(km.predict(X[eval_idx]))
        aris = [adjusted_rand_score(labels[i], labels[j])
                for i in range(n_runs) for j in range(i + 1, n_runs)]
        rows.append({
            "k": k,
            "silhouette": silhouette_score(X[eval_idx], labels[0]),
            "stability_ari_mean": float(np.mean(aris)),
            "stability_ari_min": float(np.min(aris)),
        })
    return pd.DataFrame(rows).set_index("k")



def fit_segments(X: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """Final k-means on all clients. Segments are renumbered from largest (0) to smallest."""
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    order = pd.Series(labels).value_counts().index  # largest segment first
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[label] for label in labels])


def profile(features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Segment sizes and typical values: medians for counts and amounts, means for shares."""
    df = features[SEGMENT_FEATURES].assign(segment=labels)
    agg = {c: ("mean" if c.startswith("share_") else "median") for c in SEGMENT_FEATURES}
    prof = df.groupby("segment").agg(agg)
    prof.insert(0, "share_of_clients", df["segment"].value_counts(normalize=True).sort_index())
    return prof

# Names given after reading the k = 7 profiles in notebook 02. They hold for SEED and the
# current features; notebook 02 checks the profile against them before saving.
SEGMENT_NAMES = {
    0: "Daily top-up shoppers",
    1: "Regular small baskets",
    2: "Big basket shoppers",
    3: "Occasional shoppers",
    4: "Alcohol and own brand buyers",
    5: "Point redeemers",
    6: "Express point collectors",
}

FEATURE_LABELS = {
    "days_since_last_purchase": "Days since last purchase",
    "n_baskets": "Baskets (log)",
    "avg_basket_spend": "Average basket (log)",
    "n_distinct_products": "Distinct products (log)",
    "share_products_repeated": "Products bought again",
    "share_baskets_points_spent": "Baskets with points spent",
    "share_baskets_express_received": "Baskets with express points",
    "share_own_brand": "Own brand share",
    "share_alcohol": "Alcohol share",
    "share_weekend_baskets": "Weekend baskets",
}