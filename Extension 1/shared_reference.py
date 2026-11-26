"""Run the same normal-only detector and decision rule on all three domains."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from course_data import DOMAINS, evaluation_labels, sha256, split_normal
from experiment import metrics, save_csv, save_json
from normal_bags import NormalBagDetector


SETTINGS = dict(metric="standard", distance="euclidean", reference_mode="samples",
                k=5, max_bank=8192, top_fraction=0.02, alpha=0.05, seed=42)


def run(source, image_archive, thyroid_archive, output):
    frozen = json.loads((source / "frozen.json").read_text(encoding="utf-8"))
    archives = {"images": sha256(image_archive), "thyroid": sha256(thyroid_archive)}
    if archives != frozen["archives_sha256"]:
        raise ValueError("Course archives differ from the feature-extraction run")
    info = json.loads((source / "run_info.json").read_text(encoding="utf-8"))
    if sha256(source / "features.npz") != info["features_sha256"]:
        raise ValueError("Feature archive differs from the recorded extraction")
    names = json.loads((source / "sample_names.json").read_text(encoding="utf-8"))["test"]
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", dict(settings=SETTINGS, calibration_fraction=0.2,
              archives_sha256=archives, features_sha256=info["features_sha256"],
              rule="p_normal <= 0.05, ties included in the upper-tail count"))
    predictions = []
    with np.load(source / "features.npz", allow_pickle=False) as arrays:
        for domain in DOMAINS:
            labels = arrays[f"{domain}_train_labels"]
            normals = np.flatnonzero(labels == 0)
            fit, cal, _ = split_normal(arrays[f"{domain}_global_train"][normals], seed=42,
                                       fraction=0.2, deduplicate=(domain == "thyroid"))
            bags = arrays[f"{domain}_bag_train"]
            model = NormalBagDetector(**SETTINGS).fit(bags[normals[fit]]).calibrate(bags[normals[cal]])
            scored = model.predict(arrays[f"{domain}_bag_test"])
            decisions = scored["p_normal"] <= SETTINGS["alpha"]
            if not np.array_equal(decisions, scored["prediction"]):
                raise ValueError("Rank and strict threshold decisions disagree")
            for sample, score, p_normal, pred in zip(names[domain], scored["score"],
                                                     scored["p_normal"], decisions, strict=True):
                predictions.append(dict(domain=domain, sample=sample, score=float(score),
                                        p_normal=float(p_normal), threshold=model.threshold,
                                        prediction=int(pred)))
    save_csv(output / "scores_before_labels.csv", predictions)
    test_labels = evaluation_labels(image_archive, thyroid_archive, names)
    result = []
    for domain in DOMAINS:
        rows = [row for row in predictions if row["domain"] == domain]
        y = test_labels[domain]
        for row, label in zip(rows, y, strict=True):
            row["label"] = int(label)
        result.append(dict(domain=domain, **metrics(y, [r["score"] for r in rows],
                                                   [r["prediction"] for r in rows])))
    save_csv(output / "predictions.csv", predictions)
    save_csv(output / "metrics.csv", result)
    for row in result:
        print(f"{row['domain']}: F1={row['f1']:.4f}, FP={row['fp']}, FN={row['fn']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "images", "thyroid", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    with threadpool_limits(limits=args.threads):
        run(args.run, args.images, args.thyroid, args.output)
