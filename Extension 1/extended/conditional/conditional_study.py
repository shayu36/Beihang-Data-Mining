"""Reproduce the prespecified six-column stress study and frozen course evaluation."""
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
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from conditional_reference import CANDIDATES, ConditionalReference, normal_threshold


HERE = Path(__file__).resolve().parent
PROFILES = {
    "balanced": ((0.85, 0.995, 0.75), (0.005, 0.15, 0.75)),
    "second_dominant": ((0.95, 0.999, 2.0), (0.02, 0.25, 0.5)),
    "sixth_dominant": ((0.75, 0.98, 0.5), (0.001, 0.05, 2.0)),
}
SCENARIOS = (*PROFILES, "pair_shuffle", "all_shuffle")
FEATURES = [f"feature_{i}" for i in range(1, 7)]


def checksum(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_archive(archive, split, labels=False):
    with ZipFile(archive) as bundle:
        data = bundle.read(f"thyroid/{split}-set.csv")
    rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    if rows.fieldnames != FEATURES + (["label"] if split == "test" else []):
        raise ValueError("Unexpected CSV schema")
    if labels:
        return np.array([int(float(r["label"])) for r in rows])
    return np.array([[float(r[k]) for k in FEATURES] for r in rows])


def split_normal(train):
    _, keep = np.unique(train, axis=0, return_index=True)
    ids = np.random.default_rng(42).permutation(np.sort(keep))
    n_cal = math.ceil(len(ids) * 0.2)
    return train[np.sort(ids[n_cal:])], train[np.sort(ids[:n_cal])]


def shift(normals, profile, seed):
    rng = np.random.default_rng(seed)
    out = normals.copy()
    for col, (lo, hi, strength), direction in zip((1, 5), profile, (1, -1), strict=True):
        vals = normals[:, col]
        q10, q90 = np.quantile(vals, [0.1, 0.9])
        draws = rng.uniform(lo, hi, len(normals))
        offset = rng.uniform(0, strength, len(normals)) * (q90 - q10)
        out[:, col] = np.quantile(vals, draws) + direction * offset
    return out


def shuffle(normals, cols, seed):
    """Derange row assignments within each selected feature, retaining its marginal."""
    rng = np.random.default_rng(seed)
    out = normals.copy()
    for col in cols:
        order = rng.permutation(len(normals))
        out[order, col] = normals[np.roll(order, 1), col]
    return out


def challenge(fit, val, fold):
    spearman = np.abs(spearmanr(fit).statistic)
    ij = np.triu_indices(6, 1)
    pair = np.argmax(spearman[ij])
    a, b = int(ij[0][pair]), int(ij[1][pair])
    sims = {name: shift(val, profile, 900 + 10 * fold + j)
            for j, (name, profile) in enumerate(PROFILES.items())}
    sims["pair_shuffle"] = shuffle(val, [a], seed=1526 + 10 * fold)
    sims["all_shuffle"] = shuffle(val, range(6), seed=1527 + 10 * fold)
    return sims, [a + 1, b + 1], float(spearman[a, b])


def freeze(args):
    args.output.mkdir(parents=True, exist_ok=True)
    train = read_archive(args.thyroid, "train")
    development, final_cal = split_normal(train)
    record = []
    for i, (base, heldout) in enumerate(KFold(3, shuffle=True, random_state=1126).split(development)):
        order = np.random.default_rng(800 + i).permutation(base)
        n_cal = math.ceil(0.25 * len(order))
        fit, cal, val = development[order[n_cal:]], development[order[:n_cal]], development[heldout]
        model = ConditionalReference().fit(fit)
        sims, pair, rho = challenge(fit, val, i)
        batches = {k: model.components(x) for k, x in {"cal": cal, "val": val, **sims}.items()}
        for method in CANDIDATES:
            cut = normal_threshold(model.combine(batches["cal"], method))
            counts = {key: int(np.sum(model.combine(comps, method) > cut))
                      for key, comps in batches.items() if key != "cal"}
            record.append(dict(fold=i + 1, method=method, threshold=cut, n_fit=len(fit),
                               n_calibration=len(cal), n_validation=len(val),
                               nearest_neighbors=model.k, strongest_pair=pair,
                               strongest_abs_spearman=rho, flagged=counts))
    summary = {}
    for method in CANDIDATES:
        rows = [r for r in record if r["method"] == method]
        n = sum(r["n_validation"] for r in rows)
        counts = {k: sum(r["flagged"][k] for r in rows) for k in ("val", *SCENARIOS)}
        rates = {k: v / n for k, v in counts.items()}
        directed = float(np.mean([rates[k] for k in PROFILES]))
        shuffled = (rates["pair_shuffle"] + rates["all_shuffle"]) / 2
        summary[method] = dict(n_per_scenario=n, counts=counts, rates=rates, directed=directed,
                               shuffled=shuffled, objective=(directed + shuffled) / 2)
    eligible = [
        method for method in CANDIDATES
        if summary[method]["rates"]["val"] <= 0.01
        and summary[method]["directed"] >= 0.9 * summary["product"]["directed"]
    ]
    selected = max(eligible, key=lambda k: summary[k]["objective"]) if eligible else "product"
    model = ConditionalReference(method=selected).fit(development).calibrate(final_cal)
    mp = args.output / "conditional_model.joblib"
    joblib.dump(model, mp)
    frozen = dict(selected=selected, eligible=eligible, aggregate=summary, folds=record,
                  alpha=0.005, threshold=model.threshold, normals_fit=len(development),
                  normals_calibration=len(final_cal), model_sha256=checksum(mp),
                  study_sha256=checksum(Path(__file__)), estimator_sha256=checksum(HERE / "conditional_reference.py"),
                  protocol_sha256=checksum(HERE / "protocol.json"),
                  calibration_scale_model="fit-only leave-one-out 99.5% conditional residual; fit-only 99.5% product",
                  course_opened_during_selection=False, k=model.k)
    save(args.output / "freeze.json", frozen)
    for name, stats in summary.items():
        print(f"{name:12} FPR={stats['rates']['val']:.4f} dir={stats['directed']:.4f} "
              f"shuffle={stats['shuffled']:.4f} objective={stats['objective']:.4f}")
    print(f"Selected {selected}, threshold {model.threshold:.9f}")


def evaluate(args):
    data = args.output / "freeze.json"
    frozen = json.loads(data.read_text(encoding="utf-8"))
    mp = args.output / "conditional_model.joblib"
    for file, field in ((mp, "model_sha256"), (Path(__file__), "study_sha256"),
                        (HERE / "conditional_reference.py", "estimator_sha256"), (HERE / "protocol.json", "protocol_sha256")):
        if checksum(file) != frozen[field]:
            raise ValueError(f"Changed after freeze: {file.name}")
    model = joblib.load(mp)
    result = model.predict(read_archive(args.thyroid, "test"))
    scores, pred = result["score"], result["prediction"]
    sp = args.output / "scores_before_labels.csv"
    with sp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_1based", "score", "threshold", "prediction"])
        writer.writerows((i + 1, float(s), model.threshold, int(p)) for i, (s, p) in enumerate(zip(scores, pred, strict=True)))
    y = read_archive(args.thyroid, "test", labels=True)
    if not np.array_equal(np.unique(y), [0, 1]) or len(y) != len(scores):
        raise ValueError("Invalid test labels")
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    stats = dict(method=frozen["selected"], n=len(y), tn=tn, fp=fp, fn=fn, tp=tp,
                 accuracy=(tn + tp) / len(y), precision=tp / (tp + fp) if tp + fp else 0,
                 recall=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn),
                 auroc=float(roc_auc_score(y, scores)), average_precision=float(average_precision_score(y, scores)),
                 frozen_sha256=checksum(data), scores_before_labels_sha256=checksum(sp))
    save(args.output / "course_evaluation.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "evaluate"])
    parser.add_argument("--thyroid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (freeze if args.stage == "freeze" else evaluate)(args)
