"""Read the course ZIP in place. Only the scoring step needs the official labels."""
import json
import re
from pathlib import PurePosixPath
from zipfile import ZipFile

CATEGORIES = ("hazelnut", "zipper")
ROOT = "Image_Anomaly_Detection/"
LABEL_FILE = ROOT + "image_anomaly_labels.json"
TRAIN = re.compile(r"(hazelnut|zipper)/train/(good|bad)/[^/]+\.png\Z")
TEST = re.compile(r"(hazelnut|zipper)/test/[^/]+\.png\Z")


def image_lists(archive: ZipFile):
    """Order by category, good/bad and filename; the filename is never a feature."""
    train = {c: [] for c in CATEGORIES}
    test = {c: [] for c in CATEGORIES}
    seen = set()
    for info in archive.infolist():
        if info.is_dir() or not info.filename.startswith(ROOT):
            continue
        name = info.filename[len(ROOT):]
        if PurePosixPath(name).name.startswith("."):
            continue
        train_match = TRAIN.fullmatch(name)
        test_match = TEST.fullmatch(name)
        if not (train_match or test_match):
            continue
        if name in seen:
            raise ValueError(f"Duplicate ZIP member: {name}")
        seen.add(name)
        if train_match:
            category, label = train_match.groups()
            train[category].append((name, int(label == "bad")))
        else:
            test[test_match.group(1)].append(name)
    for category in CATEGORIES:
        train[category].sort(key=lambda item: (item[1], item[0]))
        test[category].sort()
        labels = [label for _, label in train[category]]
        if [labels.count(0), labels.count(1)] != [200, 50] or not test[category]:
            raise ValueError(f"Expected 200 good and 50 bad training images, and test images for {category}")
    return train, test


def official_labels(archive, paths):
    """Exact path join after all test predictions have been saved."""
    if archive.namelist().count(LABEL_FILE) != 1:
        raise ValueError("Expected exactly one official label file")
    raw = json.loads(archive.read(LABEL_FILE))
    if set(raw) != set(paths):
        raise ValueError("Test filenames and label JSON keys differ")
    result = {}
    for path, entry in raw.items():
        if entry["original_category"] != path.split("/")[0] or entry["label"] not in ("good", "bad"):
            raise ValueError(f"Invalid label record: {path}")
        result[path] = int(entry["label"] == "bad")
    return result
