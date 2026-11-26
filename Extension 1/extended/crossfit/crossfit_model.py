"""Fixed five-fold normal-calibration ensemble for a six-column thyroid input."""
import math

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer, RobustScaler


def valid_rows(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 6 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite rows with all six original features")
    return x


class NormalFold:
    def fit(self, x, cal):
        selected = valid_rows(x)[:, [1, 2, 5]]
        self.power = PowerTransformer(standardize=False).fit(selected)
        self.scaler = RobustScaler(quantile_range=(10, 90)).fit(self.power.transform(selected))
        if float(np.median(self.raw(x))) != 0:
            raise ValueError("Expected the fitted normal descriptor median to be zero")
        calibration_scores = np.sort(self.raw(cal))
        rank = math.ceil((len(cal) + 1) * 0.995)
        if rank > len(cal):
            raise ValueError("Normal calibration sample too small at alpha 0.005")
        self.threshold = float(calibration_scores[rank - 1])
        if self.threshold <= 0:
            raise ValueError("Normal calibration threshold must be positive")
        return self

    def raw(self, x):
        selected = valid_rows(x)[:, [1, 2, 5]]
        z2, z3, z6 = self.scaler.transform(self.power.transform(selected)).T
        return np.maximum(z2, 0) * np.maximum(-z6, 0) / (1 + np.maximum(z3, 0))

    def scores(self, x):
        return self.raw(x) / self.threshold


class CrossCalibratedReference:
    def fit(self, unique_normals):
        x = valid_rows(unique_normals)
        self.models = []
        for fit, cal in KFold(5, shuffle=True, random_state=42).split(x):
            self.models.append(NormalFold().fit(x[fit], x[cal]))
        return self

    def scores(self, x):
        return np.median(np.stack([m.scores(x) for m in self.models]), axis=0)

    def predict(self, x):
        score = self.scores(x)
        return dict(score=score, threshold=1.0, prediction=(score > 1.0).astype(int))
