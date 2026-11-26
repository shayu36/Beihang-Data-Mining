"""Evaluate one frozen learned scorer, saving all scores before associating labels."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from threadpoolctl import threadpool_limits


def evaluate(archive, output):
    evaluation_path = output / "course_evaluation.json"
    if evaluation_path.exists():
        raise FileExistsError("The frozen model already has a course evaluation")
    frozen = json.loads((output / "frozen.json").read_text(encoding="utf-8"))
    model_path = output / "frozen_model.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != frozen["model_sha256"]:
        raise ValueError("Frozen model hash changed")
    model = joblib.load(model_path)
    if model["method"] != frozen["selected"] or model["threshold"] != frozen["threshold"]:
        raise ValueError("Selected method or threshold differs")
    with ZipFile(archive) as bundle:
        if hashlib.sha256(bundle.read("thyroid/train-set.csv")).hexdigest() != frozen["normal_training_member_sha256"]:
            raise ValueError("The input archive differs from the training run")
        member = bundle.read("thyroid/test-set.csv")
    data = list(csv.DictReader(io.StringIO(member.decode("utf-8-sig"))))
    x = np.array([[float(row[f"feature_{i}"]) for i in range(1, 7)] for row in data])
    with threadpool_limits(limits=1):
        scores = model["model"].scores(x)
    predictions = scores > model["threshold"]
    with (output / "scores_before_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_1based", "score", "threshold", "prediction"])
        writer.writerows((i, float(s), model["threshold"], int(p))
                         for i, (s, p) in enumerate(zip(scores, predictions, strict=True), 1))
    labels = np.array([int(float(row["label"])) for row in data])
    tn, fp, fn, tp = map(int, confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    audit = dict(
        selected=frozen["selected"], threshold=model["threshold"],
        test_member_sha256=hashlib.sha256(member).hexdigest(),
        model_sha256=frozen["model_sha256"], n=len(x), tn=tn, fp=fp, fn=fn, tp=tp,
        accuracy=(tn + tp) / len(x), precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn),
        auroc=float(roc_auc_score(labels, scores)), ap=float(average_precision_score(labels, scores)),
    )
    evaluation_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thyroid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Completed training output directory")
    args = parser.parse_args()
    evaluate(args.thyroid, args.output)
