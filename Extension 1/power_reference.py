"""Normal-trained power adapter and the shared normal-reference scoring core.

The feature pair and geometry were selected during development on previously
inspected course tests. Actual fitting and calibration use normal training rows
only, but the reported course results are not independent blind estimates.
"""
import numpy as np
from sklearn.preprocessing import PowerTransformer, RobustScaler

from normal_bags import NormalBagDetector
from unified import conformal_threshold, normal_p_values


class PowerPrototypeReference:
    """Six attributes -> directional joint-deviation descriptor -> reference distance."""

    def __init__(self, alpha=.005):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = alpha
        self.features = (1, 5)
        self.directions = (1, -1)

    @staticmethod
    def _rows(values):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim == 3 and x.shape[1] == 1:
            x = x[:, 0, :]
        if x.ndim != 2 or not len(x) or x.shape[1] != 6 or not np.isfinite(x).all():
            raise ValueError("Expected nonempty finite rows with all six original features")
        return x

    def standardized_coordinates(self, rows):
        x = self._rows(rows)
        return self.scaler.transform(self.power.transform(x[:, self.features]))

    def transform(self, rows):
        z = self.standardized_coordinates(rows)
        # A large descriptor requires both selected coordinates to deviate in
        # the development-selected directions; neither direction is a label.
        return np.maximum(z * np.asarray(self.directions), 0.).prod(axis=1, keepdims=True)

    def fit(self, normal_rows):
        x = self._rows(normal_rows)
        self.fit_count, self.dimension = x.shape
        if len(x) < 3 or np.any(np.ptp(x[:, self.features], axis=0) == 0):
            raise ValueError("At least three normal rows and nonconstant selected features are required")
        self.power = PowerTransformer(method="yeo-johnson", standardize=False).fit(x[:, self.features])
        z = self.power.transform(x[:, self.features])
        self.scaler = RobustScaler(quantile_range=(10., 90.)).fit(z)
        self.core = NormalBagDetector(metric="identity", distance="manhattan",
            reference_mode="median", k=1, alpha=self.alpha).fit(self.transform(x))
        if self.core.reference[0, 0] != 0.:
            raise ValueError("The fitted normal joint-deviation descriptor must have median zero")
        return self

    def scores(self, rows):
        # The scalar descriptor has d=1, so the shared core's sqrt(d) factor is
        # one. Its distance to the fitted zero normal median equals the product.
        return self.core.scores(self.transform(rows))

    def calibrate(self, normal_rows):
        self.calibration = self.scores(normal_rows)
        self.threshold = conformal_threshold(self.calibration, self.alpha)
        return self

    def predict(self, rows, alpha=None):
        score = self.scores(rows)
        threshold = self.threshold if alpha is None else conformal_threshold(self.calibration, alpha)
        return dict(score=score, p_normal=normal_p_values(score, self.calibration),
                    prediction=(score > threshold).astype(int), threshold=threshold)

    @property
    def selection(self):
        return dict(method="normal_power_joint_deviation_reference", original_dimension=self.dimension,
            feature_indices_0based=list(self.features), feature_indices_1based=[i+1 for i in self.features],
            directions=list(self.directions), descriptor_dimension=1,
            fit_rows=self.fit_count, power_method=self.power.method,
            power_standardize=self.power.standardize, power_lambdas=self.power.lambdas_.tolist(),
            quantile_range=list(self.scaler.quantile_range), scaler_center=self.scaler.center_.tolist(),
            scaler_scale=self.scaler.scale_.tolist(), normal_reference=self.core.reference.tolist(),
            shared_core="NormalBagDetector", metric=self.core.metric, distance=self.core.distance,
            reference_mode=self.core.reference_mode, k=self.core.k,
            descriptor="max(z_feature_2, 0) * max(-z_feature_6, 0)",
            score="Manhattan distance of the scalar joint-deviation descriptor to its fitted zero normal median",
            alpha=self.alpha, threshold_rule="normal calibration order statistic; strict score > threshold",
            fitting_labels_used=False, calibration_labels_used=False,
            development_selection="Feature pair, transformation, directions, product geometry and alpha selected on previously inspected course tests; not a fresh blind benchmark")
