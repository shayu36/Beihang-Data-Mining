"""Read original course archives; labels have a separate evaluation entry point."""
import csv
import hashlib
import io
import json
import re
from zipfile import ZipFile

import numpy as np

IMAGE_ROOT = "Image_Anomaly_Detection/"
DOMAINS = ("hazelnut", "zipper", "thyroid")
FEATURES = [f"feature_{i}" for i in range(1, 7)]


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_members(path):
    with ZipFile(path) as z:
        all_names = z.namelist()
        result = {}
        for domain in DOMAINS[:2]:
            train = sorted(n for n in all_names if re.fullmatch(IMAGE_ROOT + domain + r"/train/good/[^/]+\.png", n))
            test = sorted(n for n in all_names if re.fullmatch(IMAGE_ROOT + domain + r"/test/[^/]+\.png", n))
            if len(train) != 200 or len(train) != len(set(train)) or not test or len(test) != len(set(test)):
                raise ValueError(f"Unexpected or duplicate paths for {domain}")
            result[domain] = (train, test)
        return result


def labelled_image_members(path):
    """Include only the explicitly labelled course training directories."""
    normal_and_test = image_members(path)
    with ZipFile(path) as z:
        entries = z.namelist()
        result = {}
        for domain, (normal, test) in normal_and_test.items():
            abnormal = sorted(n for n in entries if re.fullmatch(
                IMAGE_ROOT + domain + r"/train/bad/[^/]+\.png", n))
            if len(abnormal) != 50 or len(set(abnormal)) != len(abnormal):
                raise ValueError(f"Expected 50 unique bad training images for {domain}")
            result[domain] = (normal + abnormal, np.array([0]*len(normal)+[1]*len(abnormal)), test)
        return result


def thyroid_features(path):
    # DictReader necessarily parses the CSV records, but label is never selected,
    # returned or passed to training/scoring. Evaluation reopens the file later.
    with ZipFile(path) as z:
        matrices = []
        for split in ("train", "test"):
            raw = io.StringIO(z.read(f"thyroid/{split}-set.csv").decode("utf-8-sig"))
            reader = csv.DictReader(raw)
            expected = FEATURES + (["label"] if split == "test" else [])
            if reader.fieldnames != expected:
                raise ValueError(f"Unexpected thyroid columns: {reader.fieldnames}")
            x = np.array([[float(row[f]) for f in FEATURES] for row in reader], dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError("Missing or non-finite thyroid features")
            matrices.append(x)
    return matrices


def split_normal(x, seed=42, fraction=0.2, deduplicate=False):
    # np.unique supplies a stable lexicographic order. Duplicates cannot leak
    # from reference to calibration, and repeat counts cannot weight the model.
    if deduplicate:
        _, keep = np.unique(x, axis=0, return_index=True)
        keep = np.sort(keep)
    else:
        keep = np.arange(len(x))
    order = np.random.default_rng(seed).permutation(keep)
    n_cal = int(np.ceil(len(keep) * fraction))
    return np.sort(order[n_cal:]), np.sort(order[:n_cal]), keep


def overlap_with_training(train, test):
    rows = {tuple(x) for x in train}
    return np.array([tuple(x) in rows for x in test])


def evaluation_labels(image_archive, thyroid_archive, names):
    labels = {}
    with ZipFile(image_archive) as z:
        raw = json.loads(z.read(IMAGE_ROOT + "image_anomaly_labels.json"))
        expected = {name.removeprefix(IMAGE_ROOT) for domain in DOMAINS[:2] for name in names[domain]}
        if set(raw) != expected:
            raise ValueError("Image test label paths do not align exactly")
        for domain in DOMAINS[:2]:
            labels[domain] = []
            for name in names[domain]:
                entry = raw[name.removeprefix(IMAGE_ROOT)]
                if entry["original_category"] != domain or entry["label"] not in ("good", "bad"):
                    raise ValueError("Invalid image label")
                labels[domain].append(int(entry["label"] == "bad"))
    with ZipFile(thyroid_archive) as z:
        reader = csv.DictReader(io.StringIO(z.read("thyroid/test-set.csv").decode("utf-8-sig")))
        values = [float(row["label"]) for row in reader]
        if not set(values) <= {0., 1.}:
            raise ValueError("Invalid thyroid label")
        labels["thyroid"] = [int(v) for v in values]
    return {domain: np.asarray(y, dtype=int) for domain, y in labels.items()}
