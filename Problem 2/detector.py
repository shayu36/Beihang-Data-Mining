"""Fit global SVM and normal-patch detectors and calibrate their fusion on training folds."""
from itertools import product

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from image_features import GLOBAL_DIM, GRID

SVM_GRID = tuple(product((1., 10.), (.1, 1.)))
FRACTIONS = (.005, .01, .02)
LOCAL_WEIGHTS = (0., .125, .25, .5, 1.)
BANK_SIZE = 8192
SEED = 42


def evaluate(y, scores, threshold):
    y = np.asarray(y)
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), normal=int((y == 0).sum()), abnormal=int(y.sum()),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
                auroc=float(roc_auc_score(y, scores)), ap=float(average_precision_score(y, scores)))


def threshold_for(y, scores):
    """Sweep all distinct decision regions. Equal scores always receive the same label."""
    scores = np.asarray(scores, dtype="float64")
    unique = np.unique(scores)
    if not np.isfinite(scores).all() or set(y) != {0, 1}:
        raise ValueError("Calibration requires finite scores and both labels")
    cuts = np.r_[np.nextafter(unique[0], -np.inf), unique[:-1] + np.diff(unique) / 2,
                 np.nextafter(unique[-1], np.inf)]
    order = np.argsort(scores, kind="stable")
    below = np.searchsorted(scores[order], cuts, side="left")
    fn = np.r_[0, np.cumsum(y[order] == 1)][below]
    tn = below - fn
    tp = (y == 1).sum() - fn
    fp = (y == 0).sum() - tn
    recall = tp / (tp + fn)
    balanced = (recall + tn / (tn + fp)) / 2
    f1 = 2 * tp / (2 * tp + fp + fn)
    best = np.lexsort((-cuts, recall, balanced, f1))[-1]
    return float(cuts[best])


def choose_column(y, matrix, settings):
    trials = []
    for column, setting in enumerate(settings):
        score = matrix[:, column]
        threshold = threshold_for(y, score)
        trials.append(dict(column=column, setting=setting, threshold=threshold, **evaluate(y, score, threshold)))
    winner = max(trials, key=lambda r: (r["f1"], r["balanced_accuracy"], r["auroc"]))
    return winner, trials


def fit_svm(x, labels, setting):
    c, multiplier = setting
    svm = make_pipeline(StandardScaler(), SVC(C=c, gamma=multiplier / GLOBAL_DIM, class_weight="balanced"))
    return svm.fit(x, labels)


def normal_bank(patches, normal_ids, seed):
    count = patches.shape[1]
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(normal_ids) * count, min(BANK_SIZE, len(normal_ids) * count), replace=False)
    return np.ascontiguousarray(patches[normal_ids[positions // count], positions % count])


@torch.inference_mode()
def distance_maps(patches, ids, bank):
    """Squared-norm identity for exact Euclidean 1NN, with bounded working memory."""
    memory = torch.from_numpy(bank)
    bank_norms = memory.square().sum(dim=1)
    maps = np.empty((len(ids), GRID * GRID), dtype=np.float32)
    for row, image_id in enumerate(ids):
        query = torch.from_numpy(patches[image_id])
        for start in range(0, len(query), 256):
            part = query[start:start + 256]
            sq = part.square().sum(dim=1)[:, None] + bank_norms[None, :] - 2 * part @ memory.T
            maps[row, start:start + len(part)] = sq.clamp_min_(0).min(dim=1).values.sqrt_().numpy()
    return maps


def top_average(maps, fraction):
    k = max(1, int(np.ceil(maps.shape[1] * fraction)))
    return np.partition(maps, -k, axis=1)[:, -k:].mean(axis=1)


def scale_from_normal(scores, y):
    normal = scores[y == 0]
    q10, median, q90 = np.percentile(normal, (10, 50, 90))
    return float(median), max(float(q90 - q10), 1e-8)


def normalized(scores, scale):
    center, spread = scale
    return (scores - center) / spread


def fuse(global_score, local_score, config):
    weight = config["local_weight"]
    return ((1 - weight) * normalized(global_score, config["global_scale"])
            + weight * normalized(local_score, config["local_scale"]))


def fit_detector(global_x, patches, labels, ids, folds=5):
    """Calibrate branch scores with held-out training folds before fitting the final model."""
    y = labels[ids]
    global_oof = np.empty((len(ids), len(SVM_GRID)))
    local_oof = np.empty((len(ids), len(FRACTIONS)))
    fold_ids = np.full(len(ids), -1)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (fit, val) in enumerate(cv.split(ids, y)):
        for col, setting in enumerate(SVM_GRID):
            svm = fit_svm(global_x[ids[fit]], y[fit], setting)
            global_oof[val, col] = svm.decision_function(global_x[ids[val]])
        bank = normal_bank(patches, ids[fit[y[fit] == 0]], SEED + fold)
        maps = distance_maps(patches, ids[val], bank)
        for col, fraction in enumerate(FRACTIONS):
            local_oof[val, col] = top_average(maps, fraction)
        fold_ids[val] = fold
    if (fold_ids < 0).any():
        raise RuntimeError("An image did not receive an out-of-fold prediction")
    global_best, global_trials = choose_column(y, global_oof, SVM_GRID)
    local_best, local_trials = choose_column(y, local_oof, FRACTIONS)
    gs = global_oof[:, global_best["column"]]
    ps = local_oof[:, local_best["column"]]
    config = dict(svm=global_best["setting"], fraction=local_best["setting"],
                  global_scale=scale_from_normal(gs, y), local_scale=scale_from_normal(ps, y))
    scores = np.column_stack([fuse(gs, ps, dict(config, local_weight=w)) for w in LOCAL_WEIGHTS])
    best, fusion_trials = choose_column(y, scores, LOCAL_WEIGHTS)
    config.update(local_weight=best["setting"], thresholds={"global": global_best["threshold"],
                  "local": local_best["threshold"], "fusion": best["threshold"]})
    model = dict(config=config, svm=fit_svm(global_x[ids], y, config["svm"]),
                 bank=normal_bank(patches, ids[y == 0], SEED))
    selection = dict(global_trials=global_trials, local_trials=local_trials, fusion_trials=fusion_trials,
                     oof=dict(image_ids=ids.tolist(), folds=fold_ids.tolist(), labels=y.tolist(),
                              global_scores=global_oof.tolist(), local_scores=local_oof.tolist()))
    return model, selection


def score_detector(model, global_x, patches, ids):
    config = model["config"]
    global_score = model["svm"].decision_function(global_x[ids])
    maps = distance_maps(patches, ids, model["bank"])
    local_score = top_average(maps, config["fraction"]).astype("float64")
    return {"global": global_score, "local": local_score, "fusion": fuse(global_score, local_score, config)}, maps
