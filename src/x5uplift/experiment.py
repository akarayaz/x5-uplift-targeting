"""Effect estimators for the randomized SMS experiment (notebook 04 onward).

All functions take the outcome y (0/1 purchase), the assignment t (1 = SMS) and return the
treatment minus control difference in purchase rate, in proportion units (0.033 = 3.3 points).
"""

import numpy as np
import pandas as pd
from scipy import stats

Z = stats.norm.ppf(0.975)


def diff_in_means(y, t) -> dict:
    """Difference in means with an unpooled standard error, 95% interval and two-sided p-value."""
    y, t = np.asarray(y, dtype=float), np.asarray(t).astype(bool)
    y1, y0 = y[t], y[~t]
    effect = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return {
        "effect": effect,
        "se": se,
        "ci_low": effect - Z * se,
        "ci_high": effect + Z * se,
        "p_value": 2 * stats.norm.sf(abs(effect) / se),
    }


def relative_lift(y, t) -> dict:
    """Effect relative to the control rate (0.055 = 5.5%), with a delta method 95% interval."""
    y, t = np.asarray(y, dtype=float), np.asarray(t).astype(bool)
    y1, y0 = y[t], y[~t]
    m1, m0 = y1.mean(), y0.mean()
    lift = m1 / m0 - 1
    se = np.sqrt(y1.var(ddof=1) / len(y1) / m0**2 + m1**2 * y0.var(ddof=1) / len(y0) / m0**4)
    return {"lift": lift, "ci_low": lift - Z * se, "ci_high": lift + Z * se}


def cuped(y, t, x) -> dict:
    """CUPED: remove the part of y explained by a pre-experiment covariate x, then diff_in_means.

    theta is the slope of y on x pooled over both groups. Because x is measured before the SMS,
    its mean is the same in both groups by randomization, so the adjustment leaves the expected
    effect unchanged and only removes noise.
    """
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    return diff_in_means(y - theta * (x - x.mean()), t)


def subgroup_effects(df: pd.DataFrame, by: str, y: str = "target", t: str = "treatment_flg",
                     x: str | None = None) -> pd.DataFrame:
    """Effect inside each group of column `by`: CUPAC with covariate x if given, plain difference otherwise.

    Groups must be defined by information from before the SMS, so each one is still a randomized
    experiment. theta is estimated inside each group.
    """
    rows = {}
    for group, d in df.groupby(by, observed=True, sort=True):
        res = cuped(d[y], d[t], d[x]) if x else diff_in_means(d[y], d[t])
        rows[group] = {"clients": len(d), "control_rate": d.loc[d[t] == 0, y].mean(), **res}
    out = pd.DataFrame(rows).T.astype(float)
    out["clients"] = out["clients"].astype(int)
    return out


def heterogeneity_test(effect, se) -> float:
    """Cochran's Q test: p-value for 'the group effects differ more than chance allows'."""
    e, w = np.asarray(effect, dtype=float), 1 / np.asarray(se, dtype=float) ** 2
    q = np.sum(w * (e - np.sum(w * e) / w.sum()) ** 2)
    return float(stats.chi2.sf(q, len(e) - 1))
