"""Continuous marginal tail extensions to the existing three-column score."""
import numpy as np
from power_reference import CorrectedPowerReference


class TailCorrectedReference:
    """Preserve trained coordinates below Q90; splice a positive-survival tail above it."""

    def __init__(self, kind):
        self.kind = kind
        if kind not in ("second_exponential", "second_gpd", "both_gpd"):
            raise ValueError(kind)

    def fit(self, normals):
        self.base = CorrectedPowerReference(alpha=.005).fit(normals)
        self.splices = []
        axes = [(1, 0, 1)]  # raw feature index, transformed column, orientation
        if self.kind == "both_gpd":
            axes.append((5, 2, -1))
        for raw, column, direction in axes:
            values = direction * normals[:, raw]
            lower, cutoff = np.quantile(values, [.80, .90])
            tail = values[values > cutoff] - cutoff
            if len(tail) < 15:
                raise ValueError("Not enough distinct upper-tail observations for a marginal tail")
            q50, q75 = np.quantile(tail, [.50, .75])
            shape = 0.
            if self.kind != "second_exponential":
                # For a generalized Pareto distribution, Q75/Q50 = 1 + 2**shape.
                raw_shape = np.log2(max(q75/q50 - 1., 1e-12))
                shape = float(np.clip(raw_shape * len(tail)/(len(tail)+50), 0., .75))
            median_factor = (np.expm1(np.log(2.)*shape)/shape) if shape > 1e-10 else np.log(2.)
            scale = float(q50 / median_factor)
            anchors = np.repeat(np.median(normals, axis=0)[None, :], 2, axis=0)
            anchors[:, raw] = direction * np.array([lower, cutoff])
            old, at_cut = direction * self.base.standardized_coordinates(anchors)[:, column]
            slope = float((at_cut-old)/np.log(2.))
            if not (scale > 0 and slope > 0 and at_cut > 0):
                raise ValueError("Tail extension needs an increasing, positive directional coordinate at Q90")
            self.splices.append(dict(raw=raw, column=column, direction=direction, cutoff=float(cutoff),
                                     anchor=float(at_cut), slope=slope, shape=shape, scale=scale, excess_n=len(tail)))
        return self

    def scores(self, x):
        z = self.base.standardized_coordinates(x)
        h2, h6 = z[:,0].copy(), -z[:,2].copy()
        for p in self.splices:
            h = h2 if p["raw"] == 1 else h6
            excess = p["direction"]*x[:,p["raw"]] - p["cutoff"]
            use = excess > 0
            t = excess[use]/p["scale"]
            shape = p["shape"]
            log_tail = np.log1p(shape*t)/shape if shape > 1e-10 else t
            h[use] = p["anchor"] + p["slope"]*log_tail
        desc = np.maximum(h2,0.)*np.maximum(h6,0.)/(1.+np.maximum(z[:,1],0.))
        return self.base.core.scores(desc[:,None])
