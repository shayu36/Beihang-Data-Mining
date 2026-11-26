"""Evaluate the shared image-model search on outer folds of course training images."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from supervised_refinement import SupervisedNormalRefinement


OUTER_SEED = 2026
INNER_SEED = 42
FOLDS = 5
SEARCH = {
    "C_values": [0.1, 1.0, 10.0, 100.0],
    "gamma_values": ["scale", 0.0001, 0.001, 0.01],
    "svm_weights": [0.5, 0.25, 0.75],
}


def operating_point(y, pred):
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tn + tp) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
    }


def arr_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def evaluate_domain(cache, domain):
    gx = cache[f"{domain}_global_train"]
    lx = cache[f"{domain}_bag_train"]
    y = cache[f"{domain}_train_labels"]
    if gx.ndim != 2 or lx.ndim != 3 or len(gx) != len(lx) or y.shape != (len(gx),):
        raise ValueError(f"{domain}: misaligned training feature arrays or labels")
    if set(np.unique(y)) != {0, 1} or np.bincount(y.astype(int)).min() < FOLDS:
        raise ValueError(f"{domain}: stratification needs at least {FOLDS} images per class")
    if len(np.unique(gx, axis=0)) != len(gx):
        raise ValueError(f"{domain}: repeated global features require grouped outer folds")
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=OUTER_SEED)
    folds = []
    outer_predictions = np.full(len(y), -1, dtype=int)
    for fold_id, (train, held_out) in enumerate(splitter.split(gx, y), 1):
        model = SupervisedNormalRefinement(seed=INNER_SEED, n_splits=FOLDS, **SEARCH)
        model.fit(gx[train], lx[train], y[train])
        scores = model.score(gx[held_out], lx[held_out])
        predictions = (scores >= model.threshold).astype(int)
        if np.any(outer_predictions[held_out] != -1):
            raise RuntimeError("Repeated outer-fold prediction")
        outer_predictions[held_out] = predictions
        chosen = model.selection["svm_selected"]
        folds.append({
            "fold": fold_id, "training_count": len(train), "validation_count": len(held_out),
            "selected_C": chosen["C"], "selected_gamma": chosen["gamma"],
            "svm_weight": model.svm_weight, "threshold": model.threshold,
            "auroc": float(roc_auc_score(y[held_out], scores)),
            "ap": float(average_precision_score(y[held_out], scores)),
            **operating_point(y[held_out], predictions),
        })
        print(f"{domain} outer fold {fold_id}/{FOLDS} complete", flush=True)
    if np.any(outer_predictions < 0):
        raise RuntimeError("Missing outer-fold prediction")
    return {
        "n": len(y), "normal": int(np.sum(y == 0)), "anomaly": int(np.sum(y == 1)),
        "training_sha256": {"global": arr_hash(gx), "bag": arr_hash(lx), "labels": arr_hash(y)},
        "pooled_decisions": operating_point(y, outer_predictions),
        "mean_fold_auroc": float(np.mean([f["auroc"] for f in folds])),
        "mean_fold_ap": float(np.mean([f["ap"] for f in folds])),
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="features.npz from run.py")
    parser.add_argument("--output", type=Path, required=True, help="JSON result path")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    result = {
        "scope": "Training-image nested CV of the shared parameter grid for both domains. "
                 "Each image is evaluated once by an outer model that omitted it. "
                 "This assesses the common search procedure, not the zipper-specific deployed setting.",
        "outer_seed": OUTER_SEED, "inner_seed": INNER_SEED, "outer_folds": FOLDS, "inner_folds": FOLDS,
        "search": SEARCH,
        "threshold": "Maximize F1 on inner out-of-fold predictions; classify outer score >= threshold",
        "aggregation": "Pool fold decisions for confusion counts; average ranking metrics across outer folds.",
    }
    with np.load(args.features, allow_pickle=False) as cache, threadpool_limits(limits=args.threads):
        result["domains"] = {domain: evaluate_domain(cache, domain) for domain in ("hazelnut", "zipper")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
