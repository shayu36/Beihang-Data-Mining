"""Empirical joint-row transport and discriminative scoring. No evaluation or file access."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer, RobustScaler

SCENARIOS = tuple((side, strength) for side in ("second_only", "sixth_only", "joint")
                  for strength in ("weak", "strong"))
NAMES = ["_".join(pair) for pair in SCENARIOS]


def choose_rows(column, quantiles, size, rng):
    lo, hi = np.quantile(column, quantiles)
    return rng.choice(np.flatnonzero((column >= lo) & (column <= hi)), size=size, replace=True)


def transport(normals, side, strength, rng):
    n = len(normals)
    weak = strength == "weak"
    receiver_quantiles = ((.05, .25) if weak else (.0, .05)) if side != "second_only" else (.25, .75)
    donor_quantiles = ((.80, .96) if weak else (.95, 1.0)) if side != "sixth_only" else (.10, .90)
    receivers = choose_rows(normals[:, 5], receiver_quantiles, n, rng)
    donors = choose_rows(normals[:, 1], donor_quantiles, n, rng)
    changed = normals[receivers].copy()
    f2 = normals[donors, 1]
    if side != "sixth_only":
        # Interpolate/extrapolate the high donor measurement relative to the normal median.
        center = np.median(normals[:, 1])
        low, high = (1., 1.75) if weak else (1.75, 3.5)
        f2 = center + rng.uniform(low, high, n) * (f2 - center)
    changed[:, 1] = f2
    return changed


def simulations(normals, seed):
    return {name: transport(normals, *pair, np.random.default_rng(seed+i))
            for i, (name, pair) in enumerate(zip(NAMES, SCENARIOS, strict=True))}


class SimulatedClassifier:
    def __init__(self, kind):
        self.kind = kind

    def encode(self, x):
        z = self.scale.transform(self.power.transform(x))
        high2, low6 = np.maximum(z[:, 1], 0.), np.maximum(-z[:, 5], 0.)
        return np.column_stack((z, high2, low6, np.log1p(high2 * low6)))

    def fit(self, normals, pseudo):
        self.power = PowerTransformer(standardize=False).fit(normals)
        self.scale = RobustScaler(quantile_range=(10, 90)).fit(self.power.transform(normals))
        pos = np.vstack(list(pseudo.values()))
        x = self.encode(np.vstack([normals, pos]))
        y = np.r_[np.zeros(len(normals)), np.ones(len(pos))]
        w = np.r_[np.ones(len(normals)), np.full(len(pos), len(normals)/len(pos))]
        if self.kind == "regularized_logistic":
            self.model = LogisticRegression(C=.25, max_iter=1000)
        elif self.kind == "regularized_histogram":
            self.model = HistGradientBoostingClassifier(max_iter=90, learning_rate=.07, max_leaf_nodes=7,
                min_samples_leaf=25, l2_regularization=4, early_stopping=False, random_state=2026,
                monotonic_cst=[0,1,0,0,0,-1,1,1,1])
        else:
            raise ValueError(self.kind)
        self.model.fit(x, y, sample_weight=w)
        return self

    def scores(self, rows):
        return self.model.decision_function(self.encode(rows))
