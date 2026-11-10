"""Reproduce training-only selection of the thyroid normal-reference metric.

Only the six feature columns from thyroid/train-set.csv are read. The final
normal calibration subset (seed 42, 20%) is reserved before model selection and
never used by this search. Four metrics share the same mean 5-neighbor-distance
head. The fixed inner protocol compares normal validation false-positive rates
and three synthetic perturbation families. These proxies are NOT disease labels
and their detection rates do not estimate clinical recall.

CLI outputs must use a new directory outside the containing optional-experiment
directory. Import select_metric(train) for an entirely in-memory experiment.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import platform
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
FEATURES = tuple(f"feature_{i}" for i in range(1, 7))
METRIC_NAMES = ("knn_standard", "knn_robust", "knn_lw_whiten", "knn_mcd_whiten")
PROTOCOL = {
    "version": 1,
    "purpose": "Choose a distance metric using normal training and synthetic diagnostics only",
    "input_member": "thyroid/train-set.csv",
    "feature_columns": list(FEATURES),
    "deduplication": "exact equality of all features; retain first original row",
    "outer_split": {"seed": 42, "calibration_fraction": 0.2,
                    "reserved_calibration_use": "excluded from all metric selection"},
    "inner_seeds": [42, 43, 44],
    "inner_split": {"fit": 0.6, "calibration": 0.2, "normal_validation": 0.2,
                    "rounding": "ceil for calibration and validation; remaining rows fit"},
    "head": {"neighbors": 5, "distance": "Euclidean", "aggregation": "mean distance / sqrt(d)"},
    "metrics": {
        "knn_standard": {"scaler": "standard", "whitening": None},
        "knn_robust": {"scaler": "robust", "whitening": None},
        "knn_lw_whiten": {"scaler": "standard", "whitening": "ledoit_wolf",
                          "relative_eigenvalue_floor": 1e-6, "absolute_eigenvalue_floor": 1e-12},
        "knn_mcd_whiten": {"scaler": "robust", "whitening": "mcd", "support_fraction": 0.75,
                           "random_state": 42, "relative_eigenvalue_floor": 1e-6,
                           "absolute_eigenvalue_floor": 1e-12},
    },
    "alphas": [0.01, 0.02, 0.05],
    "selection_alpha": 0.01,
    "synthetic_families": {
        "single_coordinate_shift": "one uniform random feature += uniform random sign * 3 * fit IQR",
        "independent_marginals": "each feature independently sampled from normal fit rows",
        "radial_scale": "fit median + 2 * (normal validation - fit median)",
    },
    "synthetic_random_state": "inner_seed + 1000",
    "zero_iqr_fallback": "max(normal fit population standard deviation, 1e-12)",
    "selection_rule": {
        "eligibility": "finite thresholds at alpha .01 in all inner splits and mean normal FPR <= .03",
        "objective": "highest unweighted mean synthetic AUROC across 3 seeds and 3 synthetic families",
        "max_mean_normal_fpr": 0.03,
        "tie_tolerance": 1e-6,
        "tie_preference": list(METRIC_NAMES),
    },
    "threshold": "rank=ceil((n_cal+1)*(1-alpha)); sorted score at rank; +inf if rank>n_cal",
    "prediction": "strictly score > threshold",
    "nonfinite_threshold_serialization": "null means +infinity; no sample can be flagged",
    "interpretation": "A normal-validation stability filter and synthetic diagnostic objective. "
                      "No official test data, disease labels, optimal test F1 or clinical recall are used.",
}


def _validate(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite two-dimensional normal training matrix")
    return x


def _outer_split(train: np.ndarray):
    """Match the main run: first-row deduplication, permutation seed 42, then sort."""
    _, keep = np.unique(train, axis=0, return_index=True)
    keep = np.sort(keep)
    order = np.random.default_rng(42).permutation(keep)
    n_cal = math.ceil(len(keep) * 0.2)
    return np.sort(order[n_cal:]), np.sort(order[:n_cal]), keep


def _threshold(scores: np.ndarray, alpha: float) -> float:
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float("inf")


class MetricKNN:
    """Fit a metric on normal references, retaining EVERY reference for 5-NN.

    MCD determines only a covariance-based linear transform. Its robust support
    subset is not used to discard rare normal observations from the memory bank.
    """

    def __init__(self, name: str):
        if name not in METRIC_NAMES:
            raise ValueError(f"Unknown frozen metric: {name}")
        self.name = name

    def fit(self, fit_normal: np.ndarray):
        x = _validate(fit_normal)
        if len(x) < 5:
            raise ValueError("At least five normal reference samples are required")
        self.dimension = x.shape[1]
        robust = self.name in ("knn_robust", "knn_mcd_whiten")
        self.scaler = RobustScaler() if robust else StandardScaler()
        z = self.scaler.fit_transform(x)
        self.center = np.zeros(self.dimension)
        self.whitening = np.eye(self.dimension)
        if self.name in ("knn_lw_whiten", "knn_mcd_whiten"):
            estimator = (LedoitWolf() if self.name == "knn_lw_whiten" else
                         MinCovDet(support_fraction=0.75, random_state=42)).fit(z)
            self.center = estimator.location_
            eigenvalues, vectors = np.linalg.eigh(estimator.covariance_)
            floor = max(float(eigenvalues.max()) * 1e-6, 1e-12)
            # Columns are eigenvectors; row-vector application is z @ whitening.
            self.whitening = vectors / np.sqrt(np.maximum(eigenvalues, floor))[None, :]
        references = (z - self.center) @ self.whitening
        self.neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", n_jobs=1).fit(references)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        x = _validate(x)
        if x.shape[1] != self.dimension:
            raise ValueError("Scoring dimensions differ from fitted dimensions")
        transformed = (self.scaler.transform(x) - self.center) @ self.whitening
        distances = self.neighbors.kneighbors(transformed, return_distance=True)[0]
        scores = distances.mean(axis=1) / np.sqrt(self.dimension)
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite scores from {self.name}")
        return scores


def _synthetics(fit: np.ndarray, normal_validation: np.ndarray, seed: int):
    rng = np.random.default_rng(seed + 1000)
    n, d = normal_validation.shape
    median = np.median(fit, axis=0)
    iqr = np.quantile(fit, 0.75, axis=0) - np.quantile(fit, 0.25, axis=0)
    scale = np.where(iqr > 0, iqr, np.maximum(fit.std(axis=0), 1e-12))
    shifted = normal_validation.copy()
    feature = rng.integers(0, d, n)
    shifted[np.arange(n), feature] += rng.choice([-1.0, 1.0], n) * 3 * scale[feature]
    independent = np.column_stack([fit[rng.integers(0, len(fit), n), j] for j in range(d)])
    radial = median + 2 * (normal_validation - median)
    return dict(single_coordinate_shift=shifted, independent_marginals=independent, radial_scale=radial)


def select_metric(train: np.ndarray) -> dict:
    """Return all diagnostic results and a frozen-rule selection; perform no I/O.

    ``train`` must contain normal training features only. The final calibration
    rows are returned as indices for verification but are never scored here.
    Small datasets may yield no eligible metric because finite calibration at
    alpha=.01 needs at least 99 inner calibration samples. In that case selected
    and selected_config are None rather than silently relaxing the protocol.
    """
    train = _validate(train)
    fit_idx, final_cal_idx, keep = _outer_split(train)
    proper = train[fit_idx]
    n_inner_cal = math.ceil(0.2 * len(proper))
    n_inner_val = math.ceil(0.2 * len(proper))
    if len(proper) - n_inner_cal - n_inner_val < max(5, train.shape[1] + 1):
        raise ValueError("Insufficient unique normal samples for inner fitting and covariance estimation")
    rows, messages = [], []
    with threadpool_limits(limits=4):
        for seed in PROTOCOL["inner_seeds"]:
            order = np.random.default_rng(seed).permutation(len(proper))
            calibration = proper[order[:n_inner_cal]]
            validation = proper[order[n_inner_cal:n_inner_cal+n_inner_val]]
            fit = proper[order[n_inner_cal+n_inner_val:]]
            synthetic = _synthetics(fit, validation, seed)
            for name in METRIC_NAMES:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    detector = MetricKNN(name).fit(fit)
                messages.extend({"seed": seed, "metric": name, "warning": str(item.message)} for item in caught)
                calibration_scores = detector.score(calibration)
                normal_scores = detector.score(validation)
                synthetic_scores = {family: detector.score(x) for family, x in synthetic.items()}
                for alpha in PROTOCOL["alphas"]:
                    threshold = _threshold(calibration_scores, alpha)
                    fp = int(np.count_nonzero(normal_scores > threshold))
                    for family, scores in synthetic_scores.items():
                        auc = roc_auc_score(np.r_[np.zeros(len(normal_scores)), np.ones(len(scores))],
                                            np.r_[normal_scores, scores])
                        rows.append({"seed": seed, "metric": name, "alpha": alpha,
                                     "synthetic_family": family, "n_fit": len(fit),
                                     "n_calibration": len(calibration), "n_normal_validation": len(validation),
                                     "threshold": threshold if np.isfinite(threshold) else None,
                                     "finite_threshold": bool(np.isfinite(threshold)),
                                     "normal_fp": fp, "normal_fpr": fp/len(normal_scores),
                                     "synthetic_tpr": float(np.mean(scores > threshold)),
                                     "synthetic_auroc": float(auc)})
    summary = []
    for name in METRIC_NAMES:
        for alpha in PROTOCOL["alphas"]:
            group = [r for r in rows if r["metric"] == name and r["alpha"] == alpha]
            summary.append({"metric": name, "alpha": alpha,
                            "finite_thresholds": all(r["finite_threshold"] for r in group),
                            "normal_fpr_mean": float(np.mean([r["normal_fpr"] for r in group])),
                            "normal_fpr_max": float(max(r["normal_fpr"] for r in group)),
                            "synthetic_tpr_mean": float(np.mean([r["synthetic_tpr"] for r in group])),
                            "synthetic_auroc_mean": float(np.mean([r["synthetic_auroc"] for r in group]))})
    candidates = [r for r in summary if r["alpha"] == PROTOCOL["selection_alpha"]
                  and r["finite_thresholds"]
                  and r["normal_fpr_mean"] <= PROTOCOL["selection_rule"]["max_mean_normal_fpr"]]
    selected = None
    if candidates:
        best = max(r["synthetic_auroc_mean"] for r in candidates)
        tolerance = PROTOCOL["selection_rule"]["tie_tolerance"]
        selected = next(name for name in METRIC_NAMES if any(
            r["metric"] == name and best-r["synthetic_auroc_mean"] <= tolerance for r in candidates))
    return {"protocol": copy.deepcopy(PROTOCOL), "selected": selected,
            "selected_config": copy.deepcopy(PROTOCOL["metrics"][selected]) if selected else None,
            "original_training_rows": len(train), "unique_training_rows": len(keep),
            "dimension": train.shape[1], "proper_fit_rows": len(fit_idx),
            "reserved_final_calibration_rows": len(final_cal_idx),
            "proper_fit_indices": fit_idx.tolist(), "reserved_final_calibration_indices": final_cal_idx.tolist(),
            "retained_original_indices": keep.tolist(), "diagnostics": rows, "summary": summary,
            "eligible_selection_candidates": candidates, "warnings": messages,
            "selection_status": "selected" if selected else "no_candidate_satisfies_frozen_rule",
            "final_calibration_used_for_selection": False, "official_test_data_used": False}


def _json_write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _csv_write(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_normal_training(archive: Path):
    with ZipFile(archive) as source:
        # No test member is opened, parsed, returned, or used for this search.
        raw = source.read(PROTOCOL["input_member"])
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if reader.fieldnames != list(FEATURES):
        raise ValueError("Expected six feature columns and no label column in normal training CSV")
    x = np.asarray([[float(row[k]) for k in FEATURES] for row in reader], dtype=np.float64)
    return _validate(x), hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=HERE.parent.parent / "thyroid.zip")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory outside the optional-experiment folder")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(HERE.parent):
        parser.error("--output must be outside the optional-experiment folder so it contains code only")
    if output.exists():
        parser.error("Choose a new --output directory; existing outputs are preserved")
    if not args.archive.is_file():
        parser.error(f"Archive not found: {args.archive}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "protocol": copy.deepcopy(PROTOCOL)}
    _json_write(output / "protocol_before_training_read.json", frozen)
    x, digest = _read_normal_training(args.archive)
    result = select_metric(x)
    result.update(training_csv_sha256=digest, elapsed_seconds=time.perf_counter()-started,
                  python=platform.python_version(),
                  packages={p: version(p) for p in ("numpy", "scipy", "scikit-learn", "threadpoolctl")})
    _json_write(output / "selection.json", result)
    _csv_write(output / "inner_diagnostics.csv", result["diagnostics"])
    _csv_write(output / "metric_comparison.csv", result["summary"])
    fit = set(result["proper_fit_indices"])
    final_cal = set(result["reserved_final_calibration_indices"])
    _csv_write(output / "outer_split_manifest.csv", [
        {"original_row_1based": i+1,
         "split": "proper_fit_research" if i in fit else "reserved_final_calibration" if i in final_cal else "duplicate_removed"}
        for i in range(len(x))])
    print(f"Selected metric: {result['selected']}")
    print(f"Proper normal pool: {len(fit)}; reserved final calibration: {len(final_cal)}")
    for row in result["summary"]:
        if row["alpha"] == PROTOCOL["selection_alpha"]:
            print(f"{row['metric']:18s} proxy AUROC={row['synthetic_auroc_mean']:.4f} "
                  f"normal FPR={row['normal_fpr_mean']:.5f} proxy TPR={row['synthetic_tpr_mean']:.4f}")
    print(f"Complete selection record: {output / 'selection.json'}")


if __name__ == "__main__":
    main()
