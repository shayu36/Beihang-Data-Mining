"""Re-encode 12 fixed original images without reading any class labels.

The ResNet is constructed independently, with local official weights only.
The handcrafted descriptor is shared with features.py; its input identity and
saved values are checked, but this is not a second implementation of HSV/LBP.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
from torchvision.models import ResNet18_Weights, resnet18

from features import traditional_descriptor


ROOT = Path(__file__).resolve().parent
OFFICIAL_WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_float64(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms == 0):
        raise ValueError("Cannot normalize a nonfinite or zero feature vector")
    return values / norms


def verify_features(run_dir: Path, archive_path: Path, weights_path: Path, threads: int = 4) -> dict:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Local official weights required; this verifier never downloads: {weights_path}")
    checkpoint_sha256 = file_sha256(weights_path)
    if checkpoint_sha256 != OFFICIAL_WEIGHT_SHA256:
        raise ValueError("Checkpoint SHA256 does not match the official ResNet18 IMAGENET1K_V1 file")

    indices = np.linspace(0, 599, 12, dtype=int)
    with np.load(run_dir / "features.npz", allow_pickle=False) as saved:
        names = saved["names"]
        if names.shape != (600,) or len(set(names.tolist())) != 600:
            raise ValueError("Expected exactly 600 distinct saved image filenames")
        selected_names = names[indices].tolist()
        expected_hashes = saved["image_sha256"][indices].tolist()
        saved_deep = saved["deep"][indices]
        saved_handcrafted = saved["handcrafted"][indices]
    if saved_deep.shape != (12, 512) or saved_handcrafted.shape != (12, 138):
        raise ValueError("Unexpected saved feature dimensions")

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    # weights=None prevents a network request.  We load and hash the local file.
    network = resnet18(weights=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    network.load_state_dict(state, strict=True)
    network.fc = torch.nn.Identity()
    network.eval().requires_grad_(False)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    samples, tensors, handcrafted = [], [], []
    with ZipFile(archive_path) as archive:
        for index, name, expected_hash in zip(indices.tolist(), selected_names, expected_hashes):
            raw = archive.read(f"Cluster/dataset/{name}")
            actual_hash = sha256(raw).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Original-image SHA256 disagrees with the saved feature inventory: {name}")
            with Image.open(BytesIO(raw)) as source:
                source.load()
                image = source.convert("RGB")
            tensors.append(transform(image))
            handcrafted.append(traditional_descriptor(image))
            samples.append({"index": index, "filename": name, "raw_sha256": actual_hash,
                            "raw_sha256_matches_saved": True})
    with torch.inference_mode():
        raw_deep = network(torch.stack(tensors)).cpu().numpy()
    actual_deep = normalize_float64(raw_deep)
    actual_handcrafted = normalize_float64(np.stack(handcrafted))
    tolerance = 1e-6
    checks = {}
    for name, actual, expected in (("deep", actual_deep, saved_deep),
                                   ("handcrafted", actual_handcrafted, saved_handcrafted)):
        errors = np.abs(actual - expected)
        passed = bool(np.allclose(actual, expected, rtol=0, atol=tolerance))
        if not passed:
            raise ValueError(f"Re-encoded {name} features disagree: maximum absolute error {errors.max():.12g}")
        checks[name] = {"shape": list(actual.shape), "maximum_absolute_error": float(errors.max()),
                        "allclose": passed, "atol": tolerance, "rtol": 0.0,
                        "per_image_maximum_absolute_error": errors.max(axis=1).tolist()}
    return {
        "status": "passed",
        "scope": "12 original images re-encoded without reading class labels; independent ResNet construction, shared HSV/LBP descriptor",
        "sample_selection": "np.linspace(0, 599, 12, dtype=int), in saved filename order",
        "n_samples": 12,
        "samples": samples,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_name": weights_path.name,
        "saved_features_sha256": file_sha256(run_dir / "features.npz"),
        "source_sha256": {path.name: file_sha256(path) for path in (Path(__file__), ROOT / "features.py")},
        "environment": {"device": "cpu", "threads": threads, "torch": str(torch.__version__)},
        "normalization": "float64 L2 per image; comparison uses rtol=0, atol=1e-6",
        "feature_checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=ROOT / "results" / "run")
    parser.add_argument("--archive", type=Path, default=ROOT.parent / "DM_2026_Dataset" / "Cluster.zip")
    parser.add_argument("--weights", type=Path, default=ROOT / ".cache" / "torch" / "checkpoints" / "resnet18-f37072fd.pth")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_features(args.run_dir, args.archive, args.weights, args.threads)
    output = args.output or args.run_dir / "feature_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "n_samples": result["n_samples"],
                      "deep_max_abs_error": result["feature_checks"]["deep"]["maximum_absolute_error"],
                      "handcrafted_max_abs_error": result["feature_checks"]["handcrafted"]["maximum_absolute_error"],
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
