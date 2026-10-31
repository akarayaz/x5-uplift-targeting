"""Uplift models and their evaluation (notebooks 05 to 07).

The uplift of a client is P(buy | SMS) minus P(buy | no SMS). It is never observed for one client,
because each client either got the SMS or not, so models estimate it from both groups. Every
learner has fit(X, t, y) and predict(X), where predict returns the estimated uplift.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from x5uplift.config import SEED
from x5uplift.experiment import diff_in_means

FRACTIONS = np.linspace(0, 1, 101)   # share of clients contacted, 0% to 100%


def classifier(seed: int = SEED):
    """The base model of notebook 03, so that differences come from the uplift method, not the model."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, early_stopping=True, random_state=seed)


def regressor(seed: int = SEED):
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, early_stopping=True, random_state=seed)


class SLearner:
    """One model with the SMS flag as a feature; uplift = prediction with SMS minus without."""

    def __init__(self, seed: int = SEED):
        self.seed = seed

    def fit(self, X, t, y):
        self.model = classifier(self.seed).fit(X.assign(sms=t), y)
        return self

    def predict(self, X):
        p = lambda sms: self.model.predict_proba(X.assign(sms=sms))[:, 1]
        return p(1) - p(0)


class TLearner:
    """One model per group; uplift = P(buy) from the SMS model minus P(buy) from the control model."""

    def __init__(self, seed: int = SEED):
        self.seed = seed

    def fit(self, X, t, y):
        self.m1 = classifier(self.seed).fit(X[t == 1], y[t == 1])
        self.m0 = classifier(self.seed).fit(X[t == 0], y[t == 0])
        return self

    def predict(self, X):
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class XLearner:
    """X-learner (Kuenzel et al., 2019). Stage 1 is a T-learner. Stage 2 models each client's
    imputed effect (own outcome minus the other group's prediction) and combines the two models
    with the known share of treated clients e: uplift = e * tau0 + (1 - e) * tau1. In the X5
    experiment e = 0.5, so the two models are simply averaged."""

    def __init__(self, seed: int = SEED, treatment_share: float = 0.5):
        self.seed = seed
        self.e = treatment_share

    def fit(self, X, t, y):
        sms, ctl = t == 1, t == 0
        m1 = classifier(self.seed).fit(X[sms], y[sms])
        m0 = classifier(self.seed).fit(X[ctl], y[ctl])
        d1 = y[sms] - m0.predict_proba(X[sms])[:, 1]     # SMS clients: observed minus "without SMS"
        d0 = m1.predict_proba(X[ctl])[:, 1] - y[ctl]     # control clients: "with SMS" minus observed
        self.tau1 = regressor(self.seed).fit(X[sms], d1)
        self.tau0 = regressor(self.seed).fit(X[ctl], d0)
        return self

    def predict(self, X):
        return self.e * self.tau0.predict(X) + (1 - self.e) * self.tau1.predict(X)


class ClassTransformation:
    """Class transformation (Jaskowski and Jaroszewicz, 2012). The new label z is 1 for SMS buyers
    and control non-buyers. With half of the clients in each group, uplift = 2 P(z = 1) - 1, so a
    single classifier targets the uplift directly. If the treated share e is not 0.5, clients are
    reweighted (0.5 / e and 0.5 / (1 - e)) so that both groups count equally; with e = 0.5 all
    weights are 1."""

    def __init__(self, seed: int = SEED, treatment_share: float = 0.5):
        self.seed = seed
        self.e = treatment_share

    def fit(self, X, t, y):
        weight = np.where(t == 1, 0.5 / self.e, 0.5 / (1 - self.e))
        self.model = classifier(self.seed).fit(X, (y == t).astype(int), sample_weight=weight)
        return self

    def predict(self, X):
        return 2 * self.model.predict_proba(X)[:, 1] - 1


class UpliftForest:
    """causalml uplift random forest: trees split on differences in the SMS effect directly.
    causalml does not accept missing values, so they are filled with the median plus an indicator
    column, as for logistic regression in notebook 03."""

    def __init__(self, seed: int = SEED, **params):
        self.seed = seed
        self.params = {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 200,
                       "min_samples_treatment": 50, **params}

    def fit(self, X, t, y):
        from causalml.inference.tree import UpliftRandomForestClassifier
        self.imputer = SimpleImputer(strategy="median", add_indicator=True).fit(X)
        self.model = UpliftRandomForestClassifier(control_name="0", random_state=self.seed, **self.params)
        self.model.fit(self.imputer.transform(X), treatment=np.where(t == 1, "1", "0"), y=np.asarray(y))
        return self

    def predict(self, X):
        return self.model.predict(self.imputer.transform(X))[:, 0]


def cross_fit_uplift(make_model, X: pd.DataFrame, t, y, fold) -> np.ndarray:
    """Out-of-fold uplift for every row with fold >= 0: a fresh model fitted on the other folds
    (both groups) predicts each fold. Rows with fold -1 (test set) get NaN."""
    t, y, fold = np.asarray(t), np.asarray(y), np.asarray(fold)
    pred = np.full(len(X), np.nan)
    for k in np.unique(fold[fold >= 0]):
        fit = (fold >= 0) & (fold != k)
        pred[fold == k] = make_model().fit(X[fit], t[fit], y[fit]).predict(X[fold == k])
    return pred


def ranking(score, seed: int = SEED) -> np.ndarray:
    """Clients from highest to lowest score; ties are broken at random (but reproducibly)."""
    tiebreak = np.random.default_rng(seed).random(len(score))
    return np.lexsort((tiebreak, -np.asarray(score)))


def uplift_curve(y, t, order, weights=None, fractions=FRACTIONS) -> np.ndarray:
    """Extra purchases per client of the whole group if the top fraction of the ranking gets the SMS.

    Inside the top fraction, the effect is the SMS purchase rate minus the control purchase rate;
    it is multiplied by the fraction contacted. At fraction 1 the value is the average effect.
    weights (bootstrap counts) default to 1 for every client.
    """
    y, t = np.asarray(y, dtype=float)[order], np.asarray(t, dtype=float)[order]
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)[order]
    n_sms, n_ctl = np.cumsum(w * t), np.cumsum(w * (1 - t))
    buy_sms, buy_ctl = np.cumsum(w * y * t), np.cumsum(w * y * (1 - t))
    idx = np.clip(np.ceil(fractions * len(y)).astype(int) - 1, 0, len(y) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        effect = buy_sms[idx] / n_sms[idx] - buy_ctl[idx] / n_ctl[idx]
    curve = np.nan_to_num(effect) * (n_sms[idx] + n_ctl[idx]) / (n_sms[-1] + n_ctl[-1])
    return np.where(fractions == 0, 0.0, curve)


def curve_metrics(curve, fractions=FRACTIONS, ks=(0.1, 0.2, 0.3)) -> dict:
    """AUUC: average gap between the curve and random targeting over all budgets, in extra purchases
    per client (0 = no better than random). uplift_at_k: the effect among the top k contacted."""
    random_line = fractions * curve[-1]
    out = {"auuc": float(np.mean(curve - random_line))}
    for k in ks:
        i = int(np.argmin(np.abs(fractions - k)))
        out[f"uplift_top_{round(100 * k)}"] = float(curve[i] / fractions[i])
    return out


def bootstrap_metrics(y, t, scores: dict, n_boot: int = 300, seed: int = SEED) -> pd.DataFrame:
    """Metrics of every score on the original sample (replicate 0) and n_boot Poisson bootstrap
    replicates. All scores share the same weights in a replicate, so differences are paired."""
    orders = {name: ranking(s, seed) for name, s in scores.items()}
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(n_boot + 1):
        w = None if rep == 0 else rng.poisson(1.0, len(y))
        for name, order in orders.items():
            rows.append({"rep": rep, "model": name, **curve_metrics(uplift_curve(y, t, order, w))})
    return pd.DataFrame(rows)


def summarize(boot: pd.DataFrame, reference: str | None = None, alpha: float = 0.05) -> pd.DataFrame:
    """Estimate (replicate 0) and percentile interval for each model and metric. With a reference
    model, the paired difference model minus reference is summarized instead."""
    long = boot.melt(id_vars=["rep", "model"], var_name="metric")
    if reference is not None:
        ref = long[long["model"] == reference].set_index(["rep", "metric"])["value"]
        long = long.assign(value=long["value"].to_numpy() - ref.loc[list(zip(long["rep"], long["metric"]))].to_numpy())
    est = long[long["rep"] == 0].set_index(["model", "metric"])["value"]
    reps = long[long["rep"] > 0].groupby(["model", "metric"])["value"]
    return pd.DataFrame({"estimate": est, "ci_low": reps.quantile(alpha / 2), "ci_high": reps.quantile(1 - alpha / 2)})


def calibration_by_decile(y, t, score, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted uplift and observed effect (with 95% interval) in each decile of the score,
    decile 1 = highest predicted uplift."""
    rank = pd.Series(-np.asarray(score)).rank(method="first")
    decile = pd.qcut(rank, n_bins, labels=range(1, n_bins + 1)).to_numpy()
    rows = {}
    for d in range(1, n_bins + 1):
        m = decile == d
        rows[d] = {"predicted": float(np.mean(np.asarray(score)[m])), **diff_in_means(np.asarray(y)[m], np.asarray(t)[m])}
    return pd.DataFrame(rows).T
