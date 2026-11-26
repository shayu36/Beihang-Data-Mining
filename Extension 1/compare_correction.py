"""Reproduce the frozen, exploratory thyroid correction and its two-feature comparator.

The three-feature correction was chosen from 96 configurations using the
course test. Fitting and calibration below use the normal training rows.
"""
import argparse
import csv
import io
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from course_data import overlap_with_training, sha256, split_normal, thyroid_features
from experiment import metrics, save_csv, save_json
from power_reference import CorrectedPowerReference, PowerPrototypeReference

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thyroid", type=Path, default=HERE.parent.parent / "thyroid.zip")
    parser.add_argument("--output", type=Path, required=True, help="New comparison output directory")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory to preserve existing results")

    train, test = thyroid_features(args.thyroid)
    fit, cal, _ = split_normal(train, seed=42, fraction=.2, deduplicate=True)
    models = {"original": PowerPrototypeReference(alpha=.005), "corrected": CorrectedPowerReference(alpha=.005)}
    with threadpool_limits(limits=args.threads):
        outputs = {}
        for name, model in models.items():
            model.fit(train[fit]).calibrate(train[cal])
            outputs[name] = model.predict(test)

    # Keep evaluation downstream of the frozen transformations, thresholds and test scores.
    with ZipFile(args.thyroid) as archive:
        reader = csv.DictReader(io.StringIO(archive.read("thyroid/test-set.csv").decode("utf-8-sig")))
        labels = np.asarray([float(row["label"]) for row in reader])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Thyroid test labels must be 0 or 1")
    labels = labels.astype(int)
    overlap = overlap_with_training(train, test)
    rows, results = [], []
    for name, result in outputs.items():
        scores, predictions, threshold = result["score"], result["prediction"], result["threshold"]
        for scope, mask in (("official", np.ones(len(test), dtype=bool)), ("no_train_overlap", ~overlap)):
            summary = metrics(labels[mask], scores[mask], predictions[mask])
            results.append(dict(method=name, scope=scope, threshold=threshold, **summary))
        for i, (score, prediction) in enumerate(zip(scores, predictions, strict=True)):
            rows.append(dict(method=name, sample=f"test_row_{i+1:04}", score=float(score),
                             threshold=threshold, prediction=int(prediction), label=int(labels[i]),
                             overlaps_train=int(overlap[i])))

    output.mkdir(parents=True)
    model = models["corrected"]
    model_path = output / "thyroid_corrected.joblib"
    joblib.dump(dict(version=4, domain="thyroid", kind="corrected_power_reference",
                     method="corrected", model=model), model_path)
    save_csv(output / "predictions.csv", rows)
    save_json(output / "comparison.json", dict(
        seed=42, alpha=.005, fit_rows=len(fit), calibration_rows=len(cal), archive_sha256=sha256(args.thyroid),
        development_selection=model.selection["development_selection"], test_labels_used_to_fit=False,
        test_is_independent=False, selection=model.selection, model_file=model_path.name,
        model_sha256=sha256(model_path), predictions_sha256=sha256(output / "predictions.csv"), results=results,
    ))
    for row in results:
        print(f"{row['method']:10} {row['scope']:16} F1={row['f1']:.6f} FP={row['fp']} FN={row['fn']}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
