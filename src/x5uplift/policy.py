"""Offline evaluation of targeting policies on the randomized test set (notebook 06).

A policy ranks clients and contacts the top share of the ranking (the budget). Because the SMS was
assigned at random, the extra purchases a policy creates can be estimated from the data: inside the
contacted share, the SMS purchase rate minus the control purchase rate (see uplift.uplift_curve).
"""

import numpy as np
import pandas as pd

from x5uplift.config import SEED
from x5uplift.uplift import FRACTIONS, ranking, uplift_curve


def bootstrap_curves(y, t, scores: dict, n_boot: int = 500, seed: int = SEED) -> dict:
    """Uplift curve of each ranking: row 0 on the sample, rows 1.. on Poisson bootstrap replicates.

    Every ranking uses the same weights in a replicate, so differences between policies are paired.
    "Random" is added as the expected curve of random targeting: share contacted x average effect.
    Values are extra purchases per client of the whole group, at each share in FRACTIONS.
    """
    orders = {name: ranking(s, seed) for name, s in scores.items()}
    rng = np.random.default_rng(seed)
    curves = {name: np.empty((n_boot + 1, len(FRACTIONS))) for name in [*scores, "Random"]}
    for rep in range(n_boot + 1):
        w = None if rep == 0 else rng.poisson(1.0, len(y))
        for name, order in orders.items():
            curves[name][rep] = uplift_curve(y, t, order, w)
        curves["Random"][rep] = FRACTIONS * curves[name][rep][-1]    # last point = average effect
    return curves


def budget_index(budget: float) -> int:
    """Position of a budget share (0.1 = 10%) in FRACTIONS."""
    return int(np.argmin(np.abs(FRACTIONS - budget)))


def interval(draws, alpha: float = 0.05) -> dict:
    """Estimate (row 0) and percentile interval (rows 1..) of bootstrap draws."""
    draws = np.asarray(draws)
    return {"estimate": draws[0], "ci_low": np.quantile(draws[1:], alpha / 2, axis=0),
            "ci_high": np.quantile(draws[1:], 1 - alpha / 2, axis=0)}


def per_sms(curves: dict, budgets, per: int = 1000, reference: str | None = None) -> pd.DataFrame:
    """Extra purchases per `per` messages at each budget, with intervals. With a reference policy,
    the paired difference policy minus reference instead."""
    rows = []
    for name, arr in curves.items():
        if name == reference:
            continue
        for b in budgets:
            i = budget_index(b)
            draws = arr[:, i] - (curves[reference][:, i] if reference else 0)
            rows.append({"policy": name, "budget": b, **interval(per * draws / FRACTIONS[i])})
    return pd.DataFrame(rows)


def profit(curve, cost_ratio: float) -> np.ndarray:
    """Profit per client of the whole group, in units of the margin of one purchase:
    extra purchases minus contact cost, where cost_ratio = cost of one SMS / margin of one purchase."""
    return np.asarray(curve) - cost_ratio * FRACTIONS
