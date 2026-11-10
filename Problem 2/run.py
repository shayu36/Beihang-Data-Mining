"""Task 2: train and evaluate from the course ZIP, or predict an individual image."""
import argparse
import csv
import hashlib
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits

from course_data import CATEGORIES, ROOT, image_lists, official_labels
from detector import (BANK_SIZE, FRACTIONS, LOCAL_WEIGHTS, SEED, SVM_GRID, evaluate, fit_detector,
                      score_detector)
from image_features import Encoder, GRID, encode_members

HERE = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def experiment(args, folder):
    started = time.perf_counter()
    encoder = Encoder(args.weights)
    seen_pixels, models, config_log, test_rows = {}, {}, {}, []
    with ZipFile(args.archive) as archive:
        train, test = image_lists(archive)
        for category in CATEGORIES:
            print(f"[train] {category}: extracting features from original training images", flush=True)
            names = [name for name, _ in train[category]]
            y = np.array([label for _, label in train[category]])
            gx, patches = encode_members(archive, names, encoder, args.batch_size, seen_pixels)
            model, search = fit_detector(gx, patches, y, np.arange(len(y)))
            models[category] = model
            payload = dict(version=1, category=category, weights_sha256=encoder.weights_sha256, detector=model)
            model_file = folder / f"{category}.joblib"
            joblib.dump(payload, model_file)
            config_log[category] = dict(config=model["config"], model_sha256=file_hash(model_file),
                                        train_normal=int((y == 0).sum()), train_abnormal=int(y.sum()))
            save_json(folder / f"{category}_selection.json", search)
            del gx, patches
        save_json(folder / "frozen.json", dict(categories=config_log, seed=SEED, bank_size=BANK_SIZE,
                  svm_grid=SVM_GRID, fractions=FRACTIONS, local_weights=LOCAL_WEIGHTS,
                  weights_sha256=encoder.weights_sha256, frozen_at=datetime.now(timezone.utc).isoformat()))
        for category in CATEGORIES:
            print(f"[test] {category}: scoring raw test images with frozen models", flush=True)
            names = test[category]
            gx, patches = encode_members(archive, names, encoder, args.batch_size, seen_pixels)
            scores, _ = score_detector(models[category], gx, patches, np.arange(len(names)))
            for i, name in enumerate(names):
                row = dict(category=category, path=name)
                for branch, values in scores.items():
                    threshold = config_log[category]["config"]["thresholds"][branch]
                    row.update({branch + "_score": float(values[i]), branch + "_threshold": threshold,
                                branch + "_prediction": int(values[i] >= threshold)})
                test_rows.append(row)
            del gx, patches
        save_csv(folder / "scores_before_labels.csv", test_rows)
        labels = official_labels(archive, [row["path"] for row in test_rows])
    for row in test_rows:
        row["label"] = labels[row["path"]]
    save_csv(folder / "test_predictions.csv", test_rows)
    measurements = []
    for category in CATEGORIES:
        group = [r for r in test_rows if r["category"] == category]
        y = np.array([r["label"] for r in group])
        for branch in ("all_normal", "global", "local", "fusion"):
            if branch == "all_normal":
                scores, threshold = np.zeros(len(y)), 1.
            else:
                scores = np.array([r[branch + "_score"] for r in group])
                threshold = group[0][branch + "_threshold"]
            value = evaluate(y, scores, threshold)
            measurements.append(dict(category=category, method=branch, threshold=threshold, **value))
            print(f"{category:8} {branch:10} {value['tn']+value['tp']:2}/{len(y)}"
                  f"  AUROC={value['auroc']:.4f}  F1={value['f1']:.4f}  FP={value['fp']}  FN={value['fn']}")
    save_csv(folder / "test_metrics.csv", measurements)
    save_json(folder / "run_info.json", dict(archive_sha256=file_hash(args.archive), unique_decoded_images=len(seen_pixels),
              seconds=time.perf_counter() - started, cpu_threads=args.threads, batch_size=args.batch_size,
              python=platform.python_version(), platform=platform.platform(),
              packages={p: version(p) for p in ("torch", "torchvision", "numpy", "scikit-learn", "pillow", "joblib", "threadpoolctl")}))
    correct = sum(row["fusion_prediction"] == row["label"] for row in test_rows)
    print(f"Course test accuracy: {correct}/{len(test_rows)} ({correct/len(test_rows):.2%})", flush=True)


def predict(args):
    payload = joblib.load(args.model)
    encoder = Encoder(args.weights)
    if payload["version"] != 1 or encoder.weights_sha256 != payload["weights_sha256"]:
        raise ValueError("Saved model and feature extractor do not match")
    if args.member is not None:
        with ZipFile(args.archive) as archive:
            train, test = image_lists(archive)
            category = payload["category"]
            known = set(test[category]) | {name for name, _ in train[category]}
            if args.member not in known:
                raise ValueError(f"Unknown member for {category}: {args.member}")
            image_input = BytesIO(archive.read(ROOT + args.member))
    else:
        if args.heatmap and args.heatmap.resolve() == args.image.resolve():
            raise ValueError("Use a distinct path for the heatmap")
        image_input = args.image
    with Image.open(image_input) as source:
        rgb = source.convert("RGB")
    gx, patches = encoder.encode([rgb])
    model = payload["detector"]
    scores, maps = score_detector(model, gx, patches, [0])
    threshold = model["config"]["thresholds"]["fusion"]
    result = dict(category=payload["category"], score=float(scores["fusion"][0]), threshold=threshold,
                  prediction="bad" if scores["fusion"][0] >= threshold else "good")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.heatmap:
        interpolated = Image.fromarray(maps[0].reshape(GRID, GRID)).resize(rgb.size, Image.Resampling.BILINEAR)
        field = np.asarray(interpolated)
        strength = np.clip(field / max(float(np.percentile(field, 99)), 1e-8), 0, 1)
        mask = Image.fromarray((strength ** 3 * 190).astype("uint8"))
        args.heatmap.parent.mkdir(parents=True, exist_ok=True)
        Image.composite(Image.new("RGB", rgb.size, (255, 72, 0)), rgb, mask).save(args.heatmap)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("experiment", help="Train from scratch and evaluate on the supplied test split")
    train.add_argument("--archive", type=Path, default=HERE.parent / "Image_Anomaly_Detection.zip")
    train.add_argument("--output", type=Path, help="New directory for run artifacts (default: timestamped runs directory)")
    train.add_argument("--batch-size", type=positive_int, default=8)
    single = commands.add_parser("predict", help="Use one fitted category model on one image")
    single.add_argument("--model", type=Path, required=True)
    source = single.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="An image on disk")
    source.add_argument("--member", help="A relative image path in the course ZIP, e.g. zipper/test/001.png")
    single.add_argument("--archive", type=Path, default=HERE.parent / "Image_Anomaly_Detection.zip")
    single.add_argument("--heatmap", type=Path, help="Optional local distance overlay, e.g. sample.png")
    for command in (train, single):
        command.add_argument("--weights", type=Path, help="Optional offline official Wide ResNet50-2 ImageNet V1 weights")
        command.add_argument("--threads", type=positive_int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=args.threads):
        if args.command == "predict":
            predict(args)
            return
        output = (args.output or HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
        if output.exists():
            parser.error("Output directory already exists. Choose a new --output directory.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".task2-", dir=output.parent) as temporary:
            staging = Path(temporary) / "experiment"
            staging.mkdir()
            experiment(args, staging)
            os.rename(staging, output)
        print(f"Run artifacts: {output}")


if __name__ == "__main__":
    main()
