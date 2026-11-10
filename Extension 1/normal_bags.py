"""One normal-neighbor algorithm for images (many vectors) and rows (one vector)."""
from math import ceil

import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

from unified import conformal_threshold, normal_p_values


class NormalBagDetector:
    """Normal-only metric -> reference kNN -> upper-tail mean -> sample calibration.

    A bag is one original observation, never an independent calibration patch.
    MCD only estimates a metric: all fitted normal rows remain eligible for the
    reference bank. The bank is a deterministic uniform subsample, not a coreset.
    """

    def __init__(self, metric="standard", k=5, max_bank=8192, top_fraction=.02,
                 alpha=.05, seed=42, distance="euclidean", reference_mode="samples"):
        if metric not in ("standard", "mcd_whiten", "identity"):
            raise ValueError("Unknown normal metric")
        if distance not in ("euclidean", "manhattan") or reference_mode not in ("samples", "median"):
            raise ValueError("Unknown distance or normal reference mode")
        if reference_mode == "median" and k != 1:
            raise ValueError("A single median prototype requires k=1")
        if k < 1 or max_bank < k or not 0 < top_fraction <= 1 or not 0 < alpha < 1:
            raise ValueError("Invalid detector parameters")
        self.metric, self.k, self.max_bank = metric, k, max_bank
        self.top_fraction, self.alpha, self.seed = top_fraction, alpha, seed
        self.distance, self.reference_mode = distance, reference_mode

    @staticmethod
    def _bags(values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
            raise ValueError("Expected finite observations x descriptors x features")
        return x

    def transform(self, vectors):
        return (self.scaler.transform(vectors) - self.center) @ self.whitening

    def fit(self, normal_bags):
        x = self._bags(normal_bags)
        self.fit_shape = x.shape
        self.dimension = x.shape[-1]
        vectors = x.reshape(-1, self.dimension)
        if len(vectors) < self.k:
            raise ValueError("Not enough normal reference vectors")
        self.scaler = (RobustScaler() if self.metric == "mcd_whiten" else
                       StandardScaler(with_mean=False, with_std=False) if self.metric == "identity" else
                       StandardScaler())
        z = self.scaler.fit_transform(vectors)
        self.center = np.zeros(self.dimension)
        self.whitening = np.eye(self.dimension)
        self.covariance = None
        if self.metric == "mcd_whiten":
            covariance = MinCovDet(support_fraction=.75, random_state=self.seed).fit(z)
            self.center = covariance.location_
            self.covariance = covariance.covariance_
            values, vectors_eig = np.linalg.eigh(self.covariance)
            self.eigenvalue_floor = max(float(values.max()) * 1e-6, 1e-12)
            self.whitening = vectors_eig / np.sqrt(np.maximum(values, self.eigenvalue_floor))[None, :]
        full_reference = (z - self.center) @ self.whitening
        if self.reference_mode == "median":
            # This point summarizes every fitted descriptor; it has no single
            # source row or owner. Empty indices and owner -1 mark aggregation.
            self.memory_indices = np.empty(0, dtype=int)
            self.reference = np.median(full_reference, axis=0, keepdims=True)
            self.reference_owners = np.full(1, -1, dtype=int)
        elif len(full_reference) > self.max_bank:
            self.memory_indices = np.sort(np.random.default_rng(self.seed).choice(
                len(full_reference), self.max_bank, replace=False))
        else:
            self.memory_indices = np.arange(len(full_reference))
        if self.reference_mode == "samples":
            self.reference = full_reference[self.memory_indices]
            self.reference_owners = self.memory_indices // x.shape[1]
        self.neighbors = NearestNeighbors(n_neighbors=self.k, algorithm="brute", n_jobs=1,
                                          metric=self.distance).fit(self.reference)
        return self

    def descriptor_scores(self, bags):
        x = self._bags(bags)
        if x.shape[-1] != self.dimension:
            raise ValueError("Feature dimension differs from fitted reference")
        z = self.transform(x.reshape(-1, self.dimension))
        # Bound pairwise working memory even when many full images are scored.
        scores = np.empty(len(z))
        for start in range(0, len(z), 1024):
            distance = self.neighbors.kneighbors(z[start:start+1024], return_distance=True)[0]
            scores[start:start+len(distance)] = distance.mean(axis=1) / np.sqrt(self.dimension)
        return scores.reshape(x.shape[:2])

    def scores(self, bags):
        values = self.descriptor_scores(bags)
        count = max(1, ceil(self.top_fraction * values.shape[1]))
        return np.partition(values, values.shape[1]-count, axis=1)[:, -count:].mean(axis=1)

    def calibrate(self, normal_bags):
        self.calibration = self.scores(normal_bags)
        self.threshold = conformal_threshold(self.calibration, self.alpha)
        return self

    def predict(self, bags, alpha=None):
        values = self.scores(bags)
        threshold = self.threshold if alpha is None else conformal_threshold(self.calibration, alpha)
        return dict(score=values, p_normal=normal_p_values(values, self.calibration),
                    prediction=(values > threshold).astype(int), threshold=threshold)
