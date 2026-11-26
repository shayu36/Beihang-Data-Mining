"""Fit and verify the frozen normal-calibration ensemble; see README.md for commands."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from crossfit_model import CrossCalibratedReference, NormalFold


HERE = Path(__file__).resolve().parent
FEATURES = [f"feature_{j}" for j in range(1, 7)]
PROFILES = {
    "balanced": ((0.85, 0.995, 0.75), (0.005, 0.15, 0.75)),
    "second_dominant": ((0.95, 0.999, 2.0), (0.02, 0.25, 0.5)),
    "sixth_dominant": ((0.75, 0.98, 0.5), (0.001, 0.05, 2.0)),
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def source_hash(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_zip(archive, split, labels=False):
    with ZipFile(archive) as z:
        data = z.read(f"thyroid/{split}-set.csv")
    rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    if rows.fieldnames != FEATURES + (["label"] if split == "test" else []):
        raise ValueError("Unexpected archive schema")
    if labels:
        return np.asarray([int(float(row["label"])) for row in rows])
    return np.asarray([[float(row[k]) for k in FEATURES] for row in rows])


def simulate(val, profile, seed):
    rng = np.random.default_rng(seed)
    out = val.copy()
    for j, (low, high, extent), sign in zip((1, 5), profile, (1, -1), strict=True):
        values = val[:, j]
        q10, q90 = np.quantile(values, [0.1, 0.9])
        draws = rng.uniform(low, high, len(val))
        offset = rng.uniform(0, extent, len(val)) * (q90 - q10)
        out[:, j] = np.quantile(values, draws) + sign * offset
    return out


def outer_diagnostic(unique):
    records = []
    for i, (available, held) in enumerate(KFold(5, shuffle=True, random_state=1126).split(unique)):
        train, val = unique[available], unique[held]
        ensemble = CrossCalibratedReference().fit(train)
        order = np.random.default_rng(42).permutation(len(train))
        calibration_count = math.ceil(0.2 * len(train))
        control = NormalFold().fit(train[np.sort(order[calibration_count:])], train[np.sort(order[:calibration_count])])
        samples = {"normal": val}
        for j, (name, profile) in enumerate(PROFILES.items()):
            samples[name] = simulate(val, profile, 900 + 10 * i + j)
        for name, model in (("single", control), ("crossfit", ensemble)):
            counts = {k: int(np.sum(model.scores(v) > 1.0)) for k, v in samples.items()}
            thresholds = [control.threshold] if name == "single" else [m.threshold for m in ensemble.models]
            records.append(dict(fold=i + 1, method=name, n=len(val), fit_available=len(train),
                                raw_calibration_thresholds=thresholds, flagged=counts,
                                rates={k: v / len(val) for k, v in counts.items()}))
    summary = {}
    for name in ("single", "crossfit"):
        rows = [r for r in records if r["method"] == name]
        n = sum(r["n"] for r in rows)
        counts = {k: sum(r["flagged"][k] for r in rows) for k in ("normal", *PROFILES)}
        fpr = [r["rates"]["normal"] for r in rows]
        summary[name] = dict(n_per_scenario=n, flagged=counts, rates={k: v / n for k, v in counts.items()},
                             mean_synthetic_detection=float(np.mean([counts[k] / n for k in PROFILES])),
                             minimum_fold_fpr=min(fpr), maximum_fold_fpr=max(fpr),
                             fold_fpr_population_sd=float(np.std(fpr)))
    return records, summary


def freeze(args):
    args.output.mkdir(parents=True, exist_ok=True)
    train = read_zip(args.thyroid, "train")
    _, first = np.unique(train, axis=0, return_index=True)
    unique = train[np.sort(first)]
    records, summary = outer_diagnostic(unique)
    model = CrossCalibratedReference().fit(unique)
    path = args.output / "crossfit_model.joblib"
    joblib.dump(model, path)
    frozen = dict(method="median of five corrected fold score/threshold ratios",
                  alpha=0.005, decision="strict score > 1", raw_normal_rows=len(train),
                  unique_normal_rows=len(unique), outer_validation=summary, folds=records,
                  final_fold_thresholds=[m.threshold for m in model.models],
                  model_sha256=digest(path), source_sha256={p: source_hash(HERE / p) for p in
                      ("run_crossfit.py", "crossfit_model.py", "protocol.json")},
                  test_data_read=False)
    dump(args.output / "freeze.json", frozen)
    print(json.dumps(dict(outer_validation=summary, final_fold_thresholds=frozen["final_fold_thresholds"]), indent=2))


def evaluate(args):
    path = args.output / "crossfit_model.joblib"
    frozen_path = args.output / "freeze.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if any(source_hash(HERE / p) != value for p, value in frozen["source_sha256"].items()) or digest(path) != frozen["model_sha256"]:
        raise ValueError("Model, protocol or source changed after freeze")
    model = joblib.load(path)
    result = model.predict(read_zip(args.thyroid, "test"))
    scores, pred = result["score"], result["prediction"]
    csv_path = args.output / "scores_before_labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_1based", "score", "threshold", "prediction"])
        writer.writerows((i + 1, float(s), 1.0, int(p)) for i, (s, p) in enumerate(zip(scores, pred, strict=True)))
    y = read_zip(args.thyroid, "test", labels=True)
    if len(y) != len(scores) or not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Invalid course test labels")
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    stats = dict(n=len(y), tn=tn, fp=fp, fn=fn, tp=tp, accuracy=(tn + tp) / len(y),
                 precision=tp / (tp + fp), recall=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn),
                 auroc=float(roc_auc_score(y, scores)), average_precision=float(average_precision_score(y, scores)),
                 scores_before_labels_sha256=digest(csv_path), freeze_sha256=digest(frozen_path))
    dump(args.output / "course_evaluation.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "evaluate"])
    p.add_argument("--thyroid", type=Path, required=True, help="Path to the original course thyroid.zip")
    p.add_argument("--output", type=Path, required=True, help="Directory for frozen model and evaluation")
    args = p.parse_args()
    with threadpool_limits(limits=1):
        (freeze if args.stage == "freeze" else evaluate)(args)
