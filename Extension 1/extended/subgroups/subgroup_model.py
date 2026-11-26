"""Conditional normal-score quantiles for the prespecified two-dimensional background."""
import math

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, RobustScaler


CONTROL = (0, 0)
CANDIDATES = ((128, 256), (256, 256), (256, 512))


def normal_threshold(scores):
    rank = math.ceil((len(scores) + 1) * 0.995)
    if rank > len(scores):
        raise ValueError("Normal calibration set too small at alpha 0.005")
    return float(np.sort(scores)[rank - 1])


def validate(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 6 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite six-column matrix")
    return x


class SubgroupReference:
    def __init__(self, k, prior):
        if (k, prior) not in (CONTROL, *CANDIDATES):
            raise ValueError("Unknown subgroup configuration")
        self.k, self.prior = k, prior

    def raw_score(self, x):
        z2, z3, z6 = self.scaler.transform(self.power.transform(validate(x)[:, [1, 2, 5]])).T
        return np.maximum(z2, 0) * np.maximum(-z6, 0) / (1 + np.maximum(z3, 0))

    def fit(self, x):
        x = validate(x)
        self.power = PowerTransformer(standardize=False).fit(x[:, [1, 2, 5]])
        self.scaler = RobustScaler(quantile_range=(10, 90)).fit(self.power.transform(x[:, [1, 2, 5]]))
        self.training_scores = self.raw_score(x)
        self.global_quantile = float(np.quantile(self.training_scores, 0.99, method="inverted_cdf"))
        if self.global_quantile <= 0 or np.median(self.training_scores) != 0:
            raise ValueError("Unexpected normal reference distribution")
        self.background = RobustScaler(quantile_range=(10, 90)).fit(x[:, [0, 4]])
        self.neighbors = NearestNeighbors(algorithm="kd_tree").fit(self.background.transform(x[:, [0, 4]]))
        return self

    def nearest(self, x, k):
        background = self.background.transform(validate(x)[:, [0, 4]])
        return self.neighbors.kneighbors(background, n_neighbors=k)

    def multipliers(self, x):
        if self.k == 0:
            return np.ones(len(x))
        distance, indices = self.nearest(x, self.k)
        bandwidth = distance[:, [-1]]
        ratio = np.divide(distance, bandwidth, out=np.zeros_like(distance), where=bandwidth > 0)
        weight = np.exp(-0.5 * ratio ** 2)
        values = self.training_scores[indices]
        order = np.argsort(values, axis=1)
        values = np.take_along_axis(values, order, axis=1)
        weight = np.take_along_axis(weight, order, axis=1)
        cumulative = np.cumsum(weight, axis=1)
        positions = np.argmax(cumulative >= 0.99 * cumulative[:, [-1]], axis=1)
        local = values[np.arange(len(x)), positions]
        effective = weight.sum(axis=1) ** 2 / (weight ** 2).sum(axis=1)
        shrunk = (effective * local + self.prior * self.global_quantile) / (effective + self.prior)
        return self.global_quantile / shrunk

    def scores(self, x):
        return self.raw_score(x) * self.multipliers(x)

    def calibrate(self, normal):
        self.threshold = normal_threshold(self.scores(normal))
        return self

    def predict(self, x):
        s = self.scores(x)
        return dict(score=s, prediction=(s > self.threshold).astype(int), threshold=self.threshold)
