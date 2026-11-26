"""Frozen experimental protocol: extract -> cluster -> seal -> evaluate.

Labels are opened only by evaluate(), after all assignments have been written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from itertools import combinations
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score)
from threadpoolctl import threadpool_limits

from features import extract, file_sha256, unit_rows
from verify import recompute_external_metrics

ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = ROOT.parent / "DM_2026_Dataset" / "Cluster.zip"
SEEDS = [11, 23, 37, 53, 71]
PROTOCOL = {
    "version": 1,
    "task": "600-image transductive clustering; K=6 from assignment",
    "primary_method": "deep", "primary_seed": 11,
    "selection": "Predeclared primary method and seed; no external-label selection",
    "seeds": SEEDS, "n_clusters": 6,
    "kmeans": {"init": "k-means++", "n_init": 20, "max_iter": 300, "tol": 0.0001},
    "handcrafted": {"resize": [224, 224], "hsv_bins": [8, 4, 4],
                    "lbp_P": 8, "lbp_R": 1, "lbp_method": "uniform", "dimension": 138,
                    "normalization": "sqrt(L1 histogram) per block, concatenate / sqrt(2)"},
    "deep": {"architecture": "resnet18", "weights": "IMAGENET1K_V1", "layer": "avgpool",
             "dimension": 512, "transform": "official weights.transforms(); resize256 center224",
             "normalization": "L2 per sample", "finetuning": False},
    "pca": {"n_components": 64, "svd_solver": "full", "whiten": False,
            "normalization": "L2 after PCA", "fit_scope": "all 600 unlabelled images"},
    "ward": {"linkage": "ward", "metric": "euclidean", "representation": "deep"},
    "external_metrics": ["optimal-matching ACC", "ARI", "NMI arithmetic"],
    "timestamp_note": "UTC timestamps; client reporting date uses Asia/Shanghai",
}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def cluster(features, output):
    """All model configurations and output assignments are independent of labels."""
    pca = PCA(**{k: PROTOCOL["pca"][k] for k in ("n_components", "svd_solver", "whiten")})
    features["deep_pca"] = unit_rows(pca.fit_transform(features["deep"]))
    np.savez_compressed(output / "features.npz", **features)
    np.savez_compressed(output / "pca_model.npz", components=pca.components_, mean=pca.mean_,
                        explained_variance_ratio=pca.explained_variance_ratio_)
    assignments, rows, centers = {}, [], {}
    for method in ("handcrafted", "deep", "deep_pca", "ward"):
        x = features["deep" if method == "ward" else method]
        for seed in ([0] if method == "ward" else SEEDS):
            start = time.perf_counter()
            if method == "ward":
                estimator = AgglomerativeClustering(n_clusters=6, linkage="ward", metric="euclidean")
            else:
                estimator = KMeans(n_clusters=6, random_state=seed, **PROTOCOL["kmeans"])
            labels = estimator.fit_predict(x)
            elapsed = time.perf_counter() - start
            key = f"{method}_s{seed}"
            assignments[key] = labels
            if method != "ward":
                centers[key] = estimator.cluster_centers_
            rows.append(dict(method=method, seed=seed,
                             silhouette=float(silhouette_score(x, labels)),
                             calinski_harabasz=float(calinski_harabasz_score(x, labels)),
                             davies_bouldin=float(davies_bouldin_score(x, labels)),
                             seconds=elapsed, inertia=float(getattr(estimator, "inertia_", np.nan))))
            print(f"Clustered {key}: sizes={np.bincount(labels, minlength=6).tolist()}", flush=True)
    np.savez_compressed(output / "clusters.npz", **assignments)
    np.savez_compressed(output / "centers.npz", **centers)
    write_csv(output / "internal_metrics.csv", rows)
    manifest = dict(stage="all_assignments_saved_before_label_read", sealed_at_utc=utc_now(),
                    artifacts={name: file_sha256(output / name) for name in
                               ("protocol.json", "features.npz", "clusters.npz", "centers.npz", "internal_metrics.csv")},
                    pca_explained_variance=float(pca.explained_variance_ratio_.sum()))
    save_json(output / "sealed_before_evaluation.json", manifest)
    return assignments, rows, manifest


def sklearn_external(y, c):
    """Production metrics; separate verify.py independently reconstructs formulas."""
    classes, yi = np.unique(y, return_inverse=True)
    ids, ci = np.unique(c, return_inverse=True)
    matrix = np.zeros((len(ids), len(classes)), dtype=np.int64)
    np.add.at(matrix, (ci, yi), 1)
    r, col = linear_sum_assignment(matrix, maximize=True)
    return dict(acc=float(matrix[r, col].sum() / len(y)),
                ari=float(adjusted_rand_score(y, c)),
                nmi=float(normalized_mutual_info_score(y, c, average_method="arithmetic")))


def evaluate(archive, output, features, assignments, rows, manifest):
    # This is the first label read in the complete experiment.
    with zipfile.ZipFile(archive) as z:
        labels = json.loads(z.read("Cluster/cluster_labels.json"))
    names = features["names"].tolist()
    if set(labels) != set(names):
        raise ValueError("Official label filenames and image inventory differ")
    truth = [labels[name] for name in names]
    for row in rows:
        c = assignments[f"{row['method']}_s{row['seed']}"]
        row.update(sklearn_external(truth, c))
        check = recompute_external_metrics(truth, c)
        for key in ("acc", "ari", "nmi"):
            if not np.isclose(row[key], check[key], atol=1e-12, rtol=0):
                raise AssertionError(f"Independent metric disagreement: {key}")
    write_csv(output / "all_metrics.csv", rows)
    primary = assignments["deep_s11"]
    external = recompute_external_metrics(truth, primary)
    predictions = [dict(filename=name, cluster=int(c), true_label=t)
                   for name, c, t in zip(names, primary, truth)]
    write_csv(output / "predictions.csv", predictions)
    summary = {}
    for method in ("handcrafted", "deep", "deep_pca", "ward"):
        records = [r for r in rows if r["method"] == method]
        out = {metric: {"mean": float(np.mean([r[metric] for r in records])),
                        "std": float(np.std([r[metric] for r in records], ddof=1)) if len(records) > 1 else 0.0}
               for metric in ("acc", "ari", "nmi", "silhouette", "calinski_harabasz", "davies_bouldin", "seconds")}
        runs = [assignments[f"{method}_s{r['seed']}"] for r in records]
        stability = [float(adjusted_rand_score(a, b)) for a, b in combinations(runs, 2)]
        out.update(n_runs=len(records), pairwise_stability_ari=stability,
                   stability_mean=float(np.mean(stability)) if stability else None,
                   representative_cluster_sizes=np.bincount(runs[0], minlength=6).tolist())
        summary[method] = out
    save_json(output / "summary.json", summary)
    save_json(output / "metrics.json", dict(external=external, primary_method="deep", primary_seed=11,
                                           evaluation_started_after=manifest["sealed_at_utc"],
                                           evaluation_completed_at_utc=utc_now()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "run")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory to preserve evidence: {args.output}")
    args.output.mkdir(parents=True)
    save_json(args.output / "protocol.json", dict(PROTOCOL, frozen_at_utc=utc_now()))
    with threadpool_limits(limits=args.threads):
        features, inventory = extract(args.archive, args.output, args.weights, args.batch_size, args.threads)
        assignments, rows, sealed = cluster(features, args.output)
        summary = evaluate(args.archive, args.output, features, assignments, rows, sealed)
    with (args.output / "predictions.csv").open(encoding="utf-8") as stream:
        class_counts = dict(Counter(row["true_label"] for row in csv.DictReader(stream)))
    manifest = dict(inventory, environment=dict(python=platform.python_version(), platform=platform.platform(),
                    packages={p: importlib.metadata.version(p) for p in
                              ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "scikit-image", "Pillow")}),
                    source_hashes={p.name: file_sha256(p) for p in ROOT.glob("*.py")},
                    class_counts=class_counts,
                    completed_at_utc=utc_now())
    save_json(args.output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
