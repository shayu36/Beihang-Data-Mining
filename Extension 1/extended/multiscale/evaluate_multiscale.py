"""One-shot scoring audit and evaluation of an already frozen cross-modal experiment."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def run(output, original_scores, images, thyroid):
    from course_data import DOMAINS, evaluation_labels, sha256
    from experiment import metrics, save_csv, save_json

    destination = output / "official_metrics.csv"
    if destination.exists() or (output / "official_comparison.json").exists():
        raise ValueError("This frozen directory has already been evaluated")
    frozen = json.loads((output / "frozen.json").read_text(encoding="utf-8"))
    scored = json.loads((output / "scored.json").read_text(encoding="utf-8"))
    checks = {
        "frozen.json": scored["frozen_sha256"],
        "scores_before_labels.csv": scored["scores_sha256"],
        "protocol.json": frozen["protocol_sha256"],
        "training_selection.json": frozen["selection_sha256"],
    }
    for member, expected in checks.items():
        if sha256(output / member) != expected:
            raise ValueError(f"{member} changed after freezing")
    if frozen["archives_sha256"] != {"images": sha256(images), "thyroid": sha256(thyroid)}:
        raise ValueError("External archives differ from the frozen experiment")
    scores = read_csv(output / "scores_before_labels.csv")
    if len(scores) != scored["rows"]:
        raise ValueError("Unlabeled record count differs from score manifest")
    original = {(r["domain"], r["sample"]): r for r in read_csv(original_scores)}
    winner = frozen["winner"]
    groups, names = {}, {}
    for domain in DOMAINS:
        base = [r for r in scores if r["domain"] == domain and r["method"] == "single_5"]
        new = [r for r in scores if r["domain"] == domain and r["method"] == winner]
        names[domain] = [r["sample"] for r in base]
        if (not base or [r["sample"] for r in new] != names[domain]
                or len(set(names[domain])) != len(names[domain])):
            raise ValueError(f"{domain}: mismatched paired samples")
        for row in base:
            old = original[domain, row["sample"]]
            for field in ("score", "threshold", "p_normal"):
                np.testing.assert_allclose(float(row[field]), float(old[field]), rtol=2e-10, atol=2e-12)
            if row["prediction"] != old["prediction"]:
                raise ValueError("Recomputed baseline has different predictions")
        groups[domain] = (base, new)
    if sum(len(x) for x in names.values()) != len(original):
        raise ValueError("Original baseline includes unexpected samples")
    # Only the independent evaluation step joins official labels to frozen predictions.
    labels = evaluation_labels(images, thyroid, names)
    results, paired = [], {}
    for domain in DOMAINS:
        y = labels[domain]
        predictions = []
        for method, rows in zip(("single_5", winner), groups[domain], strict=True):
            pred = np.asarray([int(r["prediction"]) for r in rows])
            score = np.asarray([float(r["score"]) for r in rows])
            results.append(dict(domain=domain, method=method, **metrics(y, score, pred)))
            predictions.append(pred)
        old, new = predictions
        paired[domain] = {
            "wrong_to_correct": int(np.sum((old != y) & (new == y))),
            "correct_to_wrong": int(np.sum((old == y) & (new != y))),
            "unchanged": int(np.sum(old == new)),
        }
    save_csv(destination, results)
    save_json(output / "official_comparison.json", dict(
        winner=winner, rows=len(scores), baseline_matches_original=True,
        original_scores_sha256=sha256(original_scores), frozen_sha256=checks["frozen.json"],
        scores_sha256=checks["scores_before_labels.csv"], paired_changes=paired,
        metrics_sha256=sha256(destination)))
    for row in results:
        print(f"{row['domain']:8} {row['method']:16} F1={row['f1']:.4f} "
              f"AP={row['average_precision']:.4f} FP={row['fp']} FN={row['fn']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True, help="Completed label-free output directory")
    parser.add_argument("--baseline", type=Path, required=True, help="Original shared scores_before_labels.csv")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--thyroid", type=Path, required=True)
    parser.add_argument("--implementation", type=Path)
    args = parser.parse_args()
    if args.implementation:
        sys.path.insert(0, str(args.implementation.resolve()))
    run(args.frozen, args.baseline, args.images, args.thyroid)
