"""Six-column, normal-fitted estimator for the prespecified thyroid stress study."""
import math

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, RobustScaler


CANDIDATES = ("product", "corrected", "conditional", "mix_15", "mix_35",
              "mix_50", "max_50", "max_75", "max_100")


def normal_threshold(scores, alpha=0.005):
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError("Calibration set too small for the chosen alpha")
    return float(np.sort(scores)[rank - 1])


def six_columns(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 6 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty, finite matrix with six original columns")
    return x


class ConditionalReference:
    """Fit each feature's conditional location and spread from local normal neighbors."""

    def __init__(self, method="conditional"):
        if method not in CANDIDATES:
            raise ValueError(f"Unknown method: {method}")
        self.method = method

    def fit(self, train):
        train = six_columns(train)
        power = PowerTransformer(standardize=False)
        transformed = power.fit_transform(train)
        self.power = power
        self.scaler = RobustScaler(quantile_range=(10, 90)).fit(transformed)
        z = self.scaler.transform(transformed)
        self.train_z = z
        self.k = max(10, round(math.sqrt(len(z))))
        self.contexts = [np.arange(6) != index for index in range(6)]
        self.references = [NearestNeighbors(algorithm="kd_tree").fit(z[:, mask])
                           for mask in self.contexts]
        q10, q90 = np.quantile(z, [0.1, 0.9], axis=0)
        self.minimum_spread = 0.25 * (q90 - q10)
        if np.any(self.minimum_spread <= 0):
            raise ValueError("A training feature has constant central spread")
        product = self._product(z)
        conditional = self._residual(z, reference_rows=True)
        self.product_scale = float(np.quantile(product, 0.995))
        self.conditional_scale = float(np.quantile(conditional, 0.995))
        if min(self.product_scale, self.conditional_scale) <= 0:
            raise ValueError("A component has no positive normal tail for scaling")
        return self

    @staticmethod
    def _product(z):
        return np.maximum(z[:, 1], 0) * np.maximum(-z[:, 5], 0)

    def _residual(self, z, reference_rows=False):
        residuals = np.empty_like(z)
        for j, (index, context) in enumerate(zip(self.references, self.contexts, strict=True)):
            neighbors = index.kneighbors(z[:, context], n_neighbors=self.k + int(reference_rows),
                                         return_distance=False)
            if reference_rows:
                # Among ties, our row can be absent. Keep exactly the first k other IDs.
                ids = np.arange(len(z))[:, None]
                keep = neighbors != ids
                keep &= np.cumsum(keep, axis=1) <= self.k
                neighbors = neighbors[keep].reshape(len(z), self.k)
            values = self.train_z[neighbors, j]
            lo, med, hi = np.quantile(values, [0.1, 0.5, 0.9], axis=1)
            residuals[:, j] = np.abs(z[:, j] - med) / np.maximum(hi - lo, self.minimum_spread[j])
        return np.partition(residuals, -2, axis=1)[:, -2:].mean(axis=1)

    def components(self, x):
        z = self.scaler.transform(self.power.transform(six_columns(x)))
        product = self._product(z)
        return {
            "product": product, "corrected": product / (1 + np.maximum(z[:, 2], 0)),
            "scaled_product": product / self.product_scale,
            "conditional": self._residual(z) / self.conditional_scale,
        }

    @staticmethod
    def combine(components, method):
        if method in ("product", "corrected", "conditional"):
            return components[method]
        kind, pct = method.split("_")
        weight = int(pct) / 100
        p, c = components["scaled_product"], components["conditional"]
        if kind == "mix":
            return (1 - weight) * p + weight * c
        return np.maximum(p, weight * c)

    def calibrate(self, calibration):
        self.threshold = normal_threshold(self.combine(self.components(calibration), self.method))
        return self

    def predict(self, x):
        scores = self.combine(self.components(x), self.method)
        return dict(score=scores, prediction=(scores > self.threshold).astype(int), threshold=self.threshold)
