"""One normal-reference detector for any fixed-length numeric representation."""
from math import ceil

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

METHODS = ("knn_residual", "knn_only", "center")


def conformal_threshold(scores, alpha=0.05):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration needs a finite, nonempty score vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    rank = ceil((len(scores) + 1) * (1 - alpha))
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float("inf")


def normal_p_values(scores, calibration):
    ordered = np.sort(calibration)
    # >= includes ties; a small value means unusual under the normal reference.
    return (1 + len(ordered) - np.searchsorted(ordered, scores, side="left")) / (len(ordered) + 1)


class UnifiedDetector:
    """PCA nearest-neighbor distance, retaining orthogonal novelty as a residual.

    All learned transformations use fit_normal only. The three predeclared
    scoring rules share preprocessing and receive their own normal thresholds.
    No function accepts labels during fitting or calibration.
    """

    def __init__(self, k=5, retained_variance=0.95, max_components=64, alpha=0.05):
        self.k = k
        self.retained_variance = retained_variance
        self.max_components = max_components
        self.alpha = alpha

    def fit(self, fit_normal):
        x = self._validate(fit_normal)
        if len(x) < self.k:
            raise ValueError("Not enough normal reference samples")
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        # Full deterministic SVD; select q from normal training variance only.
        pca = PCA(svd_solver="full").fit(z)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        wanted = int(np.searchsorted(cumulative, self.retained_variance) + 1)
        self.q = min(wanted, self.max_components, len(pca.components_))
        self.pca_mean = pca.mean_
        self.components = pca.components_[:self.q].copy()
        self.variance_retained = float(cumulative[self.q - 1])
        self.dimension = x.shape[1]
        self.reference = (z - self.pca_mean) @ self.components.T
        self.neighbors = NearestNeighbors(n_neighbors=self.k, algorithm="brute", n_jobs=1).fit(self.reference)
        return self

    @staticmethod
    def _validate(x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected a nonempty finite two-dimensional matrix")
        return x

    def scores(self, x):
        x = self._validate(x)
        if x.shape[1] != self.dimension:
            raise ValueError("Feature dimensions do not match")
        z = self.scaler.transform(x) - self.pca_mean
        u = z @ self.components.T
        residual_sq = np.square(z - u @ self.components).sum(axis=1)
        distances = self.neighbors.kneighbors(u, return_distance=True)[0]
        return {
            "knn_residual": np.sqrt(np.square(distances) + residual_sq[:, None]).mean(axis=1) / np.sqrt(self.dimension),
            "knn_only": distances.mean(axis=1) / np.sqrt(self.dimension),
            "center": np.linalg.norm(z, axis=1) / np.sqrt(self.dimension),
        }

    def calibrate(self, calibration_normal):
        self.calibration = self.scores(calibration_normal)
        self.thresholds = {name: conformal_threshold(values, self.alpha) for name, values in self.calibration.items()}
        return self

    def predict(self, x):
        scores = self.scores(x)
        return {name: {
            "score": values,
            "p_normal": normal_p_values(values, self.calibration[name]),
            "prediction": (values > self.thresholds[name]).astype(int),
        } for name, values in scores.items()}
