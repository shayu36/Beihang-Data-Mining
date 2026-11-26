"""Regression check for the published course comparison and the corrected model's single-row CLI."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from course_data import sha256, thyroid_features

# Digest of both methods' CSV produced by the original comparison script, before model export.
HISTORICAL_CSV_SHA256 = "864e92ce053d4c065ea0571971c82ab165758abd706609b827d38c2a7d3a827f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thyroid", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference", type=Path, help="Historical predictions.csv (defaults to the published run CSV)")
    parser.add_argument("--output", type=Path, help="Optional verification JSON")
    args = parser.parse_args()
    evidence = json.loads((args.run / "comparison.json").read_text(encoding="utf-8"))
    csv_file = args.run / "predictions.csv"
    reference = args.reference or csv_file
    if sha256(reference) != HISTORICAL_CSV_SHA256:
        raise ValueError("The reference CSV does not match the published, pre-export course predictions")
    model_file = args.run / "thyroid_corrected.joblib"
    if sha256(model_file) != evidence["model_sha256"] or sha256(csv_file) != evidence["predictions_sha256"]:
        raise ValueError("Saved artifacts differ from comparison.json")
    if sha256(args.thyroid) != evidence["archive_sha256"]:
        raise ValueError("The thyroid archive differs from the run input")

    with reference.open(encoding="utf-8-sig", newline="") as f:
        reference_rows = list(csv.DictReader(f))
    with csv_file.open(encoding="utf-8-sig", newline="") as f:
        for current, previous in zip(csv.DictReader(f), reference_rows, strict=True):
            for key in ("method", "sample", "prediction", "label", "overlaps_train"):
                if current[key] != previous[key]:
                    raise ValueError(f"Changed {key} in {previous['method']} {previous['sample']}")
            if not all(np.isclose(float(current[key]), float(previous[key]), rtol=1e-12, atol=1e-12)
                       for key in ("score", "threshold")):
                raise ValueError(f"Changed score or threshold in {previous['method']} {previous['sample']}")
    historical = [row for row in reference_rows if row["method"] == "corrected"]
    _, test = thyroid_features(args.thyroid)
    expected_ids = [f"test_row_{i+1:04}" for i in range(len(test))]
    if [row["sample"] for row in historical] != expected_ids:
        raise ValueError("Historical sample IDs differ from the original archive order")
    scores = np.array([float(row["score"]) for row in historical])
    original_scores = np.array([float(row["score"]) for row in reference_rows if row["method"] == "original"])
    thresholds = np.array([float(row["threshold"]) for row in historical])
    predictions = np.array([int(row["prediction"]) for row in historical])

    payload = joblib.load(model_file)
    with threadpool_limits(limits=1):
        model = payload["model"]
        result = model.predict(test)
        np.testing.assert_allclose(result["score"], scores, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(result["prediction"], predictions)
        np.testing.assert_allclose(result["threshold"], thresholds, rtol=1e-12, atol=1e-12)

        # In each predicted class, exercise the CLI on the row most affected by correction.
        cli_samples = []
        for prediction in (0, 1):
            eligible = np.flatnonzero(predictions == prediction)
            index = eligible[np.argmax(np.abs(scores[eligible] - original_scores[eligible]))]
            command = [sys.executable, str(Path(__file__).with_name("predict.py")), "--model", str(model_file),
                       "--values", *map(str, test[index]), "--threads", "1"]
            answer = json.loads(subprocess.check_output(command, text=True))
            if (answer["domain"], answer["method"], answer["model_kind"], answer["prediction"]) != (
                    "thyroid", "corrected", "corrected_power_reference", prediction):
                raise ValueError("Incorrect CLI model identity or decision")
            np.testing.assert_allclose(
                [answer["score"], answer["threshold"], answer["p_normal"]],
                [scores[index], thresholds[index], result["p_normal"][index]], rtol=1e-12, atol=1e-12,
            )
            cli_samples.append(dict(sample=expected_ids[index], **answer))
    report = dict(status="passed", historical_csv_sha256=HISTORICAL_CSV_SHA256,
                  archived_predictions_byte_identical=sha256(csv_file) == HISTORICAL_CSV_SHA256,
                  model_rows_checked=len(test), max_score_difference=float(np.max(np.abs(scores-result["score"]))),
                  cli_samples=cli_samples)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
