"""Compare frozen thyroid geometries on normal training data and artificial directional shifts.

Run in the implementation directory with --thyroid <archive> --output <result.json>.
The optional three-column correction is a diagnostic and is never a selection candidate.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from course_data import FEATURES, split_normal
from power_reference import CorrectedPowerReference, PowerPrototypeReference
from unified import conformal_threshold


ALPHA = 0.005
CANDIDATES = ("product", "minor_coordinate", "frontier_5", "frontier_15", "monotone_tree")
PROFILES = {
    "balanced": ((0.85, 0.995, 0.75), (0.005, 0.15, 0.75)),
    "second_dominant": ((0.95, 0.999, 2.0), (0.02, 0.25, 0.5)),
    "sixth_dominant": ((0.75, 0.98, 0.5), (0.001, 0.05, 2.0)),
}


def read_normal_rows(archive):
    with ZipFile(archive) as bundle:
        data = bundle.read("thyroid/train-set.csv")
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    if reader.fieldnames != FEATURES:
        raise ValueError(f"Unexpected training columns: {reader.fieldnames}")
    rows = np.asarray([[float(row[name]) for name in FEATURES] for row in reader])
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError("Expected finite normal training rows")
    return rows, hashlib.sha256(data).hexdigest()


def simulate(normals, profile, seed):
    """Retain each row's other four coordinates while replacing columns 2 and 6."""
    rng = np.random.default_rng(seed)
    simulated = normals.copy()
    for index, (low, high, extent), direction in zip((1, 5), profile, (1, -1), strict=True):
        values = normals[:, index]
        q10, q90 = np.quantile(values, [0.1, 0.9])
        draws = rng.uniform(low, high, len(normals))
        shift = rng.uniform(0, extent, len(normals)) * (q90 - q10)
        simulated[:, index] = np.quantile(values, draws) + direction * shift
    return simulated


class GeometryComparison:
    def __init__(self, normal_fit):
        self.product = PowerPrototypeReference(alpha=ALPHA).fit(normal_fit)
        self.corrected = CorrectedPowerReference(alpha=ALPHA).fit(normal_fit)
        self.reference = self.coordinates(normal_fit)
        proxies = np.vstack([
            simulate(normal_fit, profile, seed=200 + i)
            for i, profile in enumerate(PROFILES.values())
        ])
        features = np.vstack([self.reference, self.coordinates(proxies)])
        labels = np.r_[np.zeros(len(normal_fit)), np.ones(len(proxies))]
        weights = np.r_[np.ones(len(normal_fit)), np.full(len(proxies), 1.0 / len(PROFILES))]
        self.tree = HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=7, min_samples_leaf=15, l2_regularization=2,
            monotonic_cst=[1, 1], early_stopping=False, random_state=42,
        ).fit(features, labels, sample_weight=weights)

    def coordinates(self, rows):
        return self.product.standardized_coordinates(rows) * np.array([1.0, -1.0])

    def scores(self, rows, method):
        if method == "product":
            return self.product.scores(rows)
        if method == "corrected":
            return self.corrected.scores(rows)
        z = self.coordinates(rows)
        if method == "minor_coordinate":
            return z.min(axis=1)
        if method == "monotone_tree":
            return self.tree.decision_function(z)
        k = {"frontier_5": 5, "frontier_15": 15}[method]
        result = []
        for chunk in np.array_split(z, max(1, math.ceil(len(z) / 200))):
            deficits = np.maximum(chunk[:, None, :] - self.reference[None, :, :], 0).max(axis=2)
            result.append(np.partition(deficits, k - 1, axis=1)[:, k - 1])
        return np.concatenate(result)


def summarize(records, method):
    rows = [r for r in records if r["method"] == method]
    scenario_totals = {
        name: sum(r["simulated_detected_counts"][name] for r in rows) for name in PROFILES
    }
    n = sum(r["normal_validation_rows"] for r in rows)
    fp = sum(r["false_positives"] for r in rows)
    rates = {name: count / n for name, count in scenario_totals.items()}
    return dict(n=n, false_positives=fp, normal_fpr=fp / n,
                simulated_count_per_scenario=n, simulated_detected_counts=scenario_totals,
                simulated_detection_rates=rates, mean_simulated_detection_rate=float(np.mean(list(rates.values()))))


def run(archive):
    train, train_sha = read_normal_rows(archive)
    fit_idx, reserved, _ = split_normal(train, seed=42, fraction=0.2, deduplicate=True)
    development = train[fit_idx]
    rows = []
    folds = KFold(n_splits=3, shuffle=True, random_state=1126)
    methods = (*CANDIDATES, "corrected")
    for i, (available, held_out) in enumerate(folds.split(development)):
        indices = np.random.default_rng(800 + i).permutation(available)
        n_cal = math.ceil(0.25 * len(indices))
        inner_fit = development[indices[n_cal:]]
        inner_cal = development[indices[:n_cal]]
        val = development[held_out]
        model = GeometryComparison(inner_fit)
        simulations = {
            name: simulate(val, profile, seed=900 + 10 * i + j)
            for j, (name, profile) in enumerate(PROFILES.items())
        }
        for method in methods:
            cut = float(conformal_threshold(model.scores(inner_cal, method), ALPHA))
            fp = int(np.sum(model.scores(val, method) > cut))
            detected = {name: int(np.sum(model.scores(data, method) > cut))
                        for name, data in simulations.items()}
            rows.append(dict(fold=i + 1, method=method, fit_rows=len(inner_fit),
                             calibration_rows=len(inner_cal), normal_validation_rows=len(val),
                             threshold=cut, false_positives=fp, normal_fpr=fp / len(val),
                             simulated_detected_counts=detected,
                             simulated_detection_rates={name: count / len(val) for name, count in detected.items()}))
    aggregate = {method: summarize(rows, method) for method in methods}
    eligible = [m for m in CANDIDATES if aggregate[m]["normal_fpr"] <= 2 * ALPHA]
    winner = max(eligible or list(CANDIDATES), key=lambda m: aggregate[m]["mean_simulated_detection_rate"])
    return dict(
        scope="Training-only stress test. Synthetic detection rates do not estimate disease recall.",
        archive_member="thyroid/train-set.csv", member_sha256=train_sha,
        test_features_read=False, test_labels_read=False, alpha=ALPHA,
        historical_selection="The two-column directions and alpha had been selected on the course test. The three-column correction was historically selected from 96 configurations using that test.",
        counts=dict(raw_normal=len(train), unique_normal=len(fit_idx)+len(reserved),
                    development_normal=len(development), unused_outer_normal_calibration=len(reserved)),
        fold_protocol="Outer calibration set aside with original seed 42. Split development normals into 3 held-out folds (KFold, seed 1126). Within each remaining two-thirds, reserve 25% for quantile calibration (seeds 800 + zero-based fold). Use strict score > normal calibration threshold for every method.",
        simulation_protocol="For each held-out row, replace column 2 and column 6 by independent marginal quantile draws from the held-out normals, then add an independent one-sided uniform displacement measured in the column's Q90-Q10. Other four columns retain the corresponding row's values. Seeds: 900 + 10*zero-based fold + zero-based scenario. The tree learns from separately simulated inner-fit rows (seeds 200 + zero-based scenario).",
        profile_parameter_order=["quantile_lower","quantile_upper","maximum_shift_in_Q90_minus_Q10"],
        profiles=PROFILES, criteria="Highest pooled mean of the three simulated detection rates among five prespecified candidates with held-out normal FPR <= 1%. If none pass, highest rate overall. Ties use listed order.",
        candidate_specs={
            "product": "max(z2,0)*max(-z6,0); normal-fitted Yeo-Johnson and Q10-Q90 scaling",
            "minor_coordinate": "min(z2,-z6) in the same normal-fitted coordinates",
            "frontier_5": "5th-smallest one-sided Chebyshev distance to normal reference orthants",
            "frontier_15": "15th-smallest one-sided Chebyshev distance to normal reference orthants",
            "monotone_tree": "Monotonic histogram gradient boosting on (z2,-z6) with synthetic positives, 100 rounds, 7 leaves, min leaf=15, L2=2; total class weights balanced",
        }, candidate_order=CANDIDATES, selected_among_five=winner,
        extra_diagnostic=dict(method="corrected", score="product / (1 + max(z3, 0))",
            available_for_selection=False, change_to_original_selection=False),
        aggregate=aggregate, folds=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thyroid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path to result JSON")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        report = run(args.thyroid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Selected among original five: {report['selected_among_five']}")
    for name in ("product", "corrected"):
        stats = report["aggregate"][name]
        print(f"{name}: normal FPR={stats['normal_fpr']:.4%}, simulation detection={stats['mean_simulated_detection_rate']:.4%}")
