"""Recompute external clustering metrics from saved assignments, without a model.

The only label-file readers in this module are explicit evaluation functions.
Importing this module never reads ground-truth labels.  ACC uses maximum-weight
one-to-one assignment; ARI and arithmetic NMI are recomputed from their formulas.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import ZipFile

import numpy as np
from scipy.optimize import linear_sum_assignment


def _label_vector(values: Sequence, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    if result.dtype.kind in "fc" and not np.isfinite(result).all():
        raise ValueError(f"{name} contains a nonfinite label")
    if result.dtype.kind == "O" and any(value is None for value in result):
        raise ValueError(f"{name} contains a missing label")
    return result


def recompute_external_metrics(y_true: Sequence, y_pred: Sequence) -> dict:
    """Return JSON-safe external metrics and explicitly oriented count matrices.

    ``contingency_matrix`` has cluster rows and true-class columns.  Matched
    ``confusion_matrix`` has true-class rows and predicted-class columns.
    When there are more predicted clusters than true classes, an extra column
    (named None in ``confusion_columns``) records unmatched predictions.  ACC is
    always a one-to-one matching score, never a many-to-one majority-vote score.
    Class/cluster identifiers are sorted by np.unique; their numerical values
    have no meaning in the metrics.  NMI uses arithmetic entropy normalization.
    """
    truth = _label_vector(y_true, "y_true")
    predicted = _label_vector(y_pred, "y_pred")
    if len(truth) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length")
    classes, true_index = np.unique(truth, return_inverse=True)
    clusters, cluster_index = np.unique(predicted, return_inverse=True)
    n_samples = len(truth)
    table = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    np.add.at(table, (cluster_index, true_index), 1)

    match_rows, match_columns = linear_sum_assignment(table, maximize=True)
    matched = int(table[match_rows, match_columns].sum())
    mapping_index = dict(zip(match_rows.tolist(), match_columns.tolist()))
    has_unmatched = len(match_rows) < len(clusters)
    confusion = np.zeros((len(classes), len(classes) + int(has_unmatched)), dtype=np.int64)
    for row in range(len(clusters)):
        column = mapping_index.get(row, len(classes))
        confusion[:, column] += table[row, :]

    # Pair-count definition of adjusted Rand index (no sklearn metric calls).
    row_counts = table.sum(axis=1)
    column_counts = table.sum(axis=0)
    choose_two = lambda counts: float(np.sum(counts * (counts - 1) // 2))
    same_both = choose_two(table)
    same_pred = choose_two(row_counts)
    same_true = choose_two(column_counts)
    total_pairs = n_samples * (n_samples - 1) / 2
    if total_pairs == 0:
        ari = 1.0
    else:
        expected_pairs = same_pred * same_true / total_pairs
        denominator = (same_pred + same_true) / 2 - expected_pairs
        ari = 1.0 if denominator == 0 else (same_both - expected_pairs) / denominator

    # Mutual information and the arithmetic average of the two entropies.
    nonzero_row, nonzero_col = np.nonzero(table)
    counts = table[nonzero_row, nonzero_col].astype(float)
    mutual_information = float(np.sum(
        counts / n_samples * np.log(
            counts * n_samples / (row_counts[nonzero_row] * column_counts[nonzero_col])
        )
    ))
    pred_probability = row_counts / n_samples
    true_probability = column_counts / n_samples
    pred_entropy = -float(np.sum(pred_probability * np.log(pred_probability)))
    true_entropy = -float(np.sum(true_probability * np.log(true_probability)))
    entropy_mean = (pred_entropy + true_entropy) / 2
    nmi = 1.0 if entropy_mean == 0 else float(np.clip(mutual_information / entropy_mean, 0, 1))

    class_names = classes.tolist()
    cluster_ids = clusters.tolist()
    return {
        "n_samples": n_samples,
        "n_classes": len(classes),
        "n_clusters": len(clusters),
        "acc": matched / n_samples,
        "ari": float(ari),
        "nmi": nmi,
        "class_names": class_names,
        "cluster_ids": cluster_ids,
        "cluster_to_class": {
            str(cluster_ids[row]): class_names[mapping_index[row]] if row in mapping_index else None
            for row in range(len(clusters))
        },
        "contingency_matrix": table.tolist(),
        "confusion_matrix": confusion.tolist(),
        "confusion_columns": class_names + ([None] if has_unmatched else []),
    }


def _unique_json_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_official_labels(*, data_zip: Path | None = None, labels_json: Path | None = None) -> dict:
    """Read the answer labels only when explicitly called for final evaluation."""
    if (data_zip is None) == (labels_json is None):
        raise ValueError("Provide exactly one of data_zip and labels_json")
    if labels_json is not None:
        raw = Path(labels_json).read_text(encoding="utf-8-sig")
    else:
        with ZipFile(data_zip) as archive:
            raw = archive.read("Cluster/cluster_labels.json").decode("utf-8-sig")
    labels = json.loads(raw, object_pairs_hook=_unique_json_object)
    if not isinstance(labels, dict) or not labels:
        raise ValueError("Official labels must be a nonempty filename-to-class object")
    if not all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in labels.items()):
        raise ValueError("Official label filenames and classes must be nonempty strings")
    return labels


def _csv_column(fieldnames: Sequence[str], aliases: Sequence[str], *, required: bool) -> str | None:
    found = [name for name in aliases if name in fieldnames]
    if len(found) > 1 or (required and not found):
        raise ValueError(f"Expected {'one' if required else 'at most one'} column from {tuple(aliases)}")
    return found[0] if found else None


def _compare_saved_metrics(actual: dict, expected_path: Path, tolerance: float = 1e-10) -> None:
    saved = json.loads(expected_path.read_text(encoding="utf-8-sig"))
    saved = saved.get("external", saved)
    for key in ("acc", "ari", "nmi", "n_samples"):
        if key not in saved:
            raise ValueError(f"Saved metrics are missing {key}")
    for key, value in actual.items():
        if key not in saved:
            continue
        expected = saved[key]
        if key in ("acc", "ari", "nmi"):
            if not np.isfinite(expected) or not np.isclose(value, expected, rtol=0, atol=tolerance):
                raise ValueError(f"Metric mismatch for {key}: recomputed={value}, saved={expected}")
        elif value != expected:
            raise ValueError(f"Metric mismatch for {key}: recomputed={value}, saved={expected}")


def verify_predictions(
    predictions_path: str | Path,
    labels: Mapping[str, str] | None = None,
    expected_metrics_path: str | Path | None = None,
) -> dict:
    """Audit a saved CSV and recompute metrics, without loading/training a model.

    If authoritative ``labels`` are supplied, coverage must be exact and any
    true_label column is checked against them.  Without labels the function can
    only check internal consistency of the saved CSV, and requires true_label.
    """
    predictions_path = Path(predictions_path)
    with predictions_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Duplicate CSV column names")
        filename_column = _csv_column(fields, ("filename",), required=True)
        cluster_column = _csv_column(fields, ("cluster", "pred_cluster", "cluster_id"), required=True)
        truth_column = _csv_column(fields, ("true_label", "label"), required=labels is None)
        rows = list(reader)
    if not rows:
        raise ValueError("The prediction CSV is empty")
    filenames, truth, predicted = [], [], []
    for line, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row {line}")
        filename = row[filename_column]
        if not filename:
            raise ValueError(f"Missing filename at CSV row {line}")
        filenames.append(filename)
        try:
            predicted.append(int(row[cluster_column]))
        except ValueError as error:
            raise ValueError(f"Cluster ID at CSV row {line} must be an integer") from error
        if labels is not None:
            if filename not in labels:
                raise ValueError(f"Unknown filename in predictions: {filename}")
            expected_truth = labels[filename]
            if truth_column is not None and row[truth_column] != expected_truth:
                raise ValueError(f"Saved true label disagrees with official label for {filename}")
            truth.append(expected_truth)
        else:
            if not row[truth_column]:
                raise ValueError(f"Missing true label at CSV row {line}")
            truth.append(row[truth_column])
    if len(filenames) != len(set(filenames)):
        raise ValueError("Duplicate filenames in predictions")
    if labels is not None and set(filenames) != set(labels):
        missing = sorted(set(labels) - set(filenames))
        raise ValueError(f"Predictions do not cover all official labels; missing {len(missing)}: {missing[:5]}")
    actual = recompute_external_metrics(truth, predicted)
    if expected_metrics_path is not None:
        _compare_saved_metrics(actual, Path(expected_metrics_path))
    return actual


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_metric_rows(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {}
    for row in rows:
        key = f"{row['method']}_s{int(row['seed'])}"
        if key in keyed:
            raise ValueError(f"Duplicate experiment {key} in {path.name}")
        keyed[key] = row
    return keyed


def verify_run_artifacts(run_dir: str | Path, labels: Mapping[str, str], predictions_path: Path) -> dict:
    """Check the sealed artifacts and all saved experiments without rerunning models.

    A matching seal establishes consistency with the stored digest record.  It
    does not independently prove the historical order in which files were made.
    """
    run_dir = Path(run_dir).resolve()
    seal_path = run_dir / "sealed_before_evaluation.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8-sig"))
    required = {"protocol.json", "features.npz", "clusters.npz", "centers.npz", "internal_metrics.csv"}
    artifacts = seal.get("artifacts", {})
    if not required.issubset(artifacts):
        raise ValueError(f"Seal omits required artifacts: {sorted(required - set(artifacts))}")
    verified_hashes = {}
    for name, expected in artifacts.items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir):
            raise ValueError(f"Sealed artifact escapes run directory: {name}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Sealed SHA256 mismatch: {name}")
        verified_hashes[name] = actual

    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8-sig"))
    seeds = protocol["seeds"]
    primary_key = f"{protocol['primary_method']}_s{protocol['primary_seed']}"
    if primary_key != "deep_s11" or protocol["n_clusters"] != 6:
        raise ValueError("The declared primary experiment must be deep_s11 with six clusters")
    expected_keys = {f"{method}_s{seed}" for method in ("handcrafted", "deep", "deep_pca") for seed in seeds} | {"ward_s0"}
    if len(expected_keys) != 16:
        raise ValueError("Expected five seeds for each of three KMeans methods plus Ward (16 experiments)")
    dimensions = {"handcrafted": protocol["handcrafted"]["dimension"],
                  "deep": protocol["deep"]["dimension"],
                  "deep_pca": protocol["pca"]["n_components"]}
    with np.load(run_dir / "features.npz", allow_pickle=False) as feature_file:
        names_array = feature_file["names"]
        if names_array.ndim != 1:
            raise ValueError("Feature names must be one-dimensional")
        names = names_array.tolist()
        if len(names) != 600 or len(set(names)) != 600 or set(names) != set(labels):
            raise ValueError("Feature inventory must match all 600 official filenames exactly")
        feature_shapes, representations = {}, {}
        for method, dimension in dimensions.items():
            vectors = feature_file[method]
            if vectors.shape != (len(names), dimension) or not np.isfinite(vectors).all():
                raise ValueError(f"Invalid feature shape or nonfinite values: {method}")
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=1e-6):
                raise ValueError(f"Feature rows are not unit length: {method}")
            feature_shapes[method] = list(vectors.shape)
            representations[method] = vectors

    with np.load(run_dir / "pca_model.npz", allow_pickle=False) as pca_file:
        pca_mean, pca_components = pca_file["mean"], pca_file["components"]
        if pca_mean.shape != (dimensions["deep"],) or pca_components.shape != (dimensions["deep_pca"], dimensions["deep"]):
            raise ValueError("Saved PCA mean/components have invalid shapes")
        if not np.isfinite(pca_mean).all() or not np.isfinite(pca_components).all():
            raise ValueError("Saved PCA mean/components contain nonfinite values")
        reconstructed_pca = (representations["deep"] - pca_mean) @ pca_components.T
        norms = np.linalg.norm(reconstructed_pca, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Reconstructed PCA features contain zero vectors")
        reconstructed_pca /= norms
        if not np.allclose(reconstructed_pca, representations["deep_pca"], rtol=0, atol=1e-10):
            raise ValueError("Saved deep_pca does not match saved deep, PCA mean/components, and L2 normalization")

    with np.load(run_dir / "clusters.npz", allow_pickle=False) as cluster_file:
        if set(cluster_file.files) != expected_keys:
            raise ValueError("Saved cluster experiments differ from the frozen protocol")
        assignments = {key: cluster_file[key] for key in cluster_file.files}
    for key, assignment in assignments.items():
        if assignment.shape != (len(names),) or assignment.dtype.kind not in "iu":
            raise ValueError(f"Invalid assignment shape or dtype: {key}")
        if set(assignment.tolist()) != set(range(6)):
            raise ValueError(f"Expected six nonempty clusters with IDs 0..5: {key}")
    with np.load(run_dir / "centers.npz", allow_pickle=False) as center_file:
        if set(center_file.files) != expected_keys - {"ward_s0"}:
            raise ValueError("Saved centers do not match the KMeans experiments")
        for key in center_file.files:
            method = key.rsplit("_s", 1)[0]
            centers = center_file[key]
            if centers.shape != (6, dimensions[method]) or not np.isfinite(centers).all():
                raise ValueError(f"Invalid center shape or nonfinite values: {key}")
            distances = np.sum((representations[method][:, None, :] - centers[None, :, :]) ** 2, axis=2)
            nearest = np.argmin(distances, axis=1)
            if not np.array_equal(nearest, assignments[key]):
                raise ValueError(f"Saved assignments are not nearest-center predictions: {key}")

    with predictions_path.open(encoding="utf-8-sig", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    if [row["filename"] for row in predictions] != names:
        raise ValueError("Prediction row order does not match features.npz names")
    if not np.array_equal(np.asarray([int(row["cluster"]) for row in predictions]), assignments[primary_key]):
        raise ValueError("Prediction clusters do not match sealed clusters.npz deep_s11")

    internal_rows = _read_metric_rows(run_dir / "internal_metrics.csv")
    all_rows = _read_metric_rows(run_dir / "all_metrics.csv")
    if set(internal_rows) != expected_keys or set(all_rows) != expected_keys:
        raise ValueError("Metric tables must each contain the 16 protocol experiments exactly once")
    truth = [labels[name] for name in names]
    recomputed = {}
    for key in sorted(expected_keys):
        for field, original in internal_rows[key].items():
            if field not in all_rows[key]:
                raise ValueError(f"all_metrics.csv lacks sealed internal field {field}: {key}")
            copied = all_rows[key][field]
            if field == "method":
                equal = original == copied
            else:
                equal = np.isclose(float(original), float(copied), rtol=0, atol=1e-10, equal_nan=True)
            if not equal:
                raise ValueError(f"Internal metric was changed after sealing: {key}/{field}")
        values = recompute_external_metrics(truth, assignments[key])
        recomputed[key] = {metric: values[metric] for metric in ("acc", "ari", "nmi")}
        for metric, value in recomputed[key].items():
            saved = float(all_rows[key][metric])
            if not np.isfinite(saved) or not np.isclose(value, saved, rtol=0, atol=1e-10):
                raise ValueError(f"External metric mismatch in all_metrics.csv: {key}/{metric}")
    return {
        "passed_checks": [
            "SHA256 of every artifact named in sealed_before_evaluation.json matches the saved seal",
            "Feature inventory exactly covers 600 official filenames; representations have declared shapes and finite unit rows",
            "Saved PCA features reproduce from deep features, saved PCA mean/components, and L2 normalization",
            "Cluster and center inventories match all 16 declared experiments; assignments have six nonempty clusters",
            "All 15 KMeans assignments equal the nearest saved center for their saved feature representation",
            "Prediction filenames preserve feature order and every prediction equals sealed deep_s11 assignment",
            "All 16 rows preserve sealed internal metrics and independently recomputed ACC, ARI, arithmetic NMI match all_metrics.csv",
        ],
        "seal_sha256": _sha256(seal_path),
        "pca_model_sha256": _sha256(run_dir / "pca_model.npz"),
        "verified_artifact_sha256": verified_hashes,
        "feature_shapes": feature_shapes,
        "verified_experiment_count": len(recomputed),
        "all_external_recomputed": recomputed,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=root / "results" / "run")
    parser.add_argument("--predictions", type=Path, help="Override run_dir/predictions.csv")
    parser.add_argument("--metrics", type=Path, help="Override run_dir/metrics.json")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--labels-json", type=Path)
    source.add_argument("--data-zip", type=Path, default=root.parent / "DM_2026_Dataset" / "Cluster.zip")
    parser.add_argument("--output", type=Path, help="Audit output; default run_dir/verification.json")
    args = parser.parse_args()
    prediction_path = args.predictions or args.run_dir / "predictions.csv"
    metric_path = args.metrics or args.run_dir / "metrics.json"
    labels = load_official_labels(labels_json=args.labels_json) if args.labels_json else load_official_labels(data_zip=args.data_zip)
    metrics = verify_predictions(prediction_path, labels, metric_path)
    artifact_evidence = verify_run_artifacts(args.run_dir, labels, prediction_path)
    evidence = {
        "status": "passed",
        "scope": "Saved-artifact consistency and independent metric reconstruction; no image encoding or model fitting rerun, no independent proof of historical execution order",
        **artifact_evidence,
        "predictions_sha256": _sha256(prediction_path),
        "saved_metrics_sha256": _sha256(metric_path),
        "external": metrics,
    }
    evidence["passed_checks"].append("Saved true_label values agree with official labels by filename and metrics.json.external matches independently recomputed metrics")
    output_path = args.output or args.run_dir / "verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "acc": metrics["acc"], "ari": metrics["ari"], "nmi": metrics["nmi"], "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
