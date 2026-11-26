"""Audit attainable fixed-score operating points, label conflicts, and the missed case."""
import argparse
import csv
from collections import Counter, defaultdict
from fractions import Fraction
import io
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from course_data import FEATURES, split_normal


def analyze(archive_path, results):
    with ZipFile(archive_path) as archive:
        raw = [list(csv.DictReader(io.TextIOWrapper(archive.open(f"thyroid/{part}-set.csv"), "utf-8-sig")))
               for part in ("train", "test")]
    train, test = (np.array([[float(r[k]) for k in FEATURES] for r in rows]) for rows in raw)
    labels = np.array([int(float(r["label"])) for r in raw[1]])
    keys = [tuple(r) for r in test]
    by_vector = defaultdict(Counter)
    for vector, label in zip(keys, labels, strict=True):
        by_vector[vector][int(label)] += 1
    opposites = [counts for counts in by_vector.values() if len(counts) > 1]
    train_keys = set(map(tuple, train))
    overlap = np.array([key in train_keys for key in keys])

    with (results / "correction" / "predictions.csv").open(encoding="utf-8-sig") as f:
        rows = [row for row in csv.DictReader(f) if row["method"] == "corrected"]
    assert len(rows) == len(test)
    assert [r["sample"] for r in rows] == [f"test_row_{i+1:04}" for i in range(len(test))]
    np.testing.assert_array_equal(labels, [int(r["label"]) for r in rows])
    scores = np.array([float(r["score"]) for r in rows])
    cuts = np.array([float(r["threshold"]) for r in rows])
    assert len(np.unique(cuts)) == 1
    predictions = (scores > cuts).astype(int)
    np.testing.assert_array_equal(predictions, [int(r["prediction"]) for r in rows])

    # Enumerate the distinct decision sets: starting with no positives, move one entire tied-score block at a time.
    score_groups = defaultdict(Counter)
    for score, label in zip(scores, labels, strict=True):
        score_groups[float(score)][int(label)] += 1
    normal, disease = (int(np.sum(labels == label)) for label in (0, 1))
    fp = tp = 0
    def outcome(fp, tp):
        tn, fn = normal - fp, disease - tp
        return dict(tn=tn, fp=fp, fn=fn, tp=tp, correct=tn+tp, f1=Fraction(2*tp, 2*tp+fp+fn))
    outcomes = [outcome(0, 0)]
    for score in sorted(score_groups, reverse=True):
        fp += score_groups[score][0]
        tp += score_groups[score][1]
        outcomes.append(outcome(fp, tp))
    def report_optimum(key):
        best = max(outcome[key] for outcome in outcomes)
        result = []
        for row in outcomes:
            if row[key] == best:
                result.append(dict(row, accuracy=row["correct"]/len(test), f1=float(row["f1"])))
        return result

    missed = np.flatnonzero((labels == 1) & (predictions == 0))
    assert len(missed) == 1
    index = int(missed[0])
    fit, cal, _ = split_normal(train, 42, .2, deduplicate=True)
    with threadpool_limits(limits=1):
        six = joblib.load(results / "run2" / "thyroid.joblib")["normal_model"]
        assert six.metric == "mcd_whiten" and six.k == 5 and six.dimension == 6
        z = six.transform(test[index:index+1])
        distances, indices = six.neighbors.kneighbors(z)
        fit_ids = fit[six.reference_owners[indices[0]]]
        scored = float(six.scores(test[index:index+1])[0])
        np.testing.assert_allclose(distances[0].mean()/np.sqrt(6), scored)
        cal_scores = six.scores(train[cal])
        np.testing.assert_allclose(cal_scores, six.calibration, atol=1e-12, rtol=0)
        sorted_cal = (int(np.sum(cal_scores <= scored)), len(cal_scores))

    return dict(
        purpose="Post-hoc diagnostics of a previously frozen score. None of the evaluated cut positions is exported or used in model fitting.",
        six_feature_consistency=dict(test_rows=len(labels), unique_test_vectors=len(by_vector),
            repeated_test_rows=len(labels)-len(by_vector), test_groups_with_both_labels=len(opposites),
            test_rows_in_conflicting_groups=sum(c.total() for c in opposites),
            min_errors_forced_by_identical_test_inputs=sum(min(c.values()) for c in opposites),
            test_normals_overlapping_normal_train=int(np.sum(overlap & (labels==0))),
            test_diseased_overlapping_normal_train=int(np.sum(overlap & (labels==1)))),
        monotonic_threshold_bounds=dict(decision="higher score means anomalous", test_rows=len(labels),
            distinct_scores=len(score_groups), distinct_decision_sets=len(outcomes),
            max_correct=report_optimum("correct"), max_f1=report_optimum("f1")),
        missed_disease=dict(sample=rows[index]["sample"], corrected_score=float(scores[index]),
            frozen_threshold=float(cuts[index]), six_feature_empirical_percentiles=(np.mean(train[fit] <= test[index],axis=0)*100).tolist(),
            normal_reference="Six raw features, fit-only RobustScaler and robust MCD whitening, 5-nearest Euclidean distances divided by sqrt(6) and averaged",
            reference_count=len(fit), calibration_count=len(cal), nearest_fit_row_ids_1based=(fit_ids+1).tolist(),
            five_distances_divided_by_sqrt6=(distances[0]/np.sqrt(6)).tolist(), mean_distance=scored,
            six_feature_threshold=six.threshold, calibration_normal_count_at_or_below=sorted_cal[0],
            six_feature_normal_calibration_percentile=100*sorted_cal[0]/sorted_cal[1],
            six_feature_p_normal=(1+int(np.sum(cal_scores >= scored)))/(1+len(cal_scores))),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thyroid", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, help="Directory containing run2 and correction")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.thyroid, args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
