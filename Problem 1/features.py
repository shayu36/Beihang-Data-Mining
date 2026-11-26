"""Label-free image inventory and reproducible frozen representations."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
import torch
from torchvision.models import ResNet18_Weights, resnet18


def file_sha256(path):
    digest = sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit_rows(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(x).all():
        raise ValueError("Non-finite or zero image descriptor")
    return np.asarray(x / norms, dtype=np.float64)


def traditional_descriptor(image):
    """Hellinger HSV(128) and uniform LBP(10), equally weighted blocks."""
    small = image.resize((224, 224), Image.Resampling.BILINEAR)
    hsv = np.asarray(small.convert("HSV"), dtype=np.float64) / 256.0
    hist = np.histogramdd(hsv.reshape(-1, 3), bins=(8, 4, 4),
                          range=((0, 1), (0, 1), (0, 1)))[0].ravel()
    gray = np.asarray(small.convert("L"))
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    texture = np.bincount(lbp.astype(np.int64).ravel(), minlength=10)
    return np.concatenate([np.sqrt(hist / hist.sum()),
                           np.sqrt(texture / texture.sum())]) / np.sqrt(2)


def extract(archive, output, weights_path=None, batch_size=16, threads=4):
    """No label JSON is accessed in this function."""
    start = time.perf_counter()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(Path(__file__).resolve().parent / ".cache" / "torch"))
    weights = ResNet18_Weights.IMAGENET1K_V1
    if weights_path:
        if file_sha256(weights_path) != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec":
            raise ValueError("--weights must be the official ResNet18 IMAGENET1K_V1 checkpoint")
        net = resnet18(weights=None)
        net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        checkpoint = Path(weights_path)
    else:
        net = resnet18(weights=weights, progress=True)
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    net.fc = torch.nn.Identity()
    net.eval().requires_grad_(False)
    transform = weights.transforms()
    descriptors, deep, hashes, pixels = [], [], [], {}
    sizes, modes, duplicate_pixels = Counter(), Counter(), []
    with zipfile.ZipFile(archive) as z:
        members = sorted(n for n in z.namelist()
                         if n.startswith("Cluster/dataset/") and n.lower().endswith(".png"))
        names = [Path(n).name for n in members]
        if len(names) != 600 or len(set(names)) != 600:
            raise ValueError("Expected 600 uniquely named course images")
        batch = []
        for index, member in enumerate(members):
            content = z.read(member)
            with Image.open(BytesIO(content)) as original:
                original.load()
                sizes[f"{original.width}x{original.height}"] += 1
                modes[original.mode] += 1
                rgb = original.convert("RGB")
            pixel_hash = sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if pixel_hash in pixels:
                duplicate_pixels.append([pixels[pixel_hash], names[index]])
            pixels[pixel_hash] = names[index]
            hashes.append(sha256(content).hexdigest())
            descriptors.append(traditional_descriptor(rgb))
            batch.append(transform(rgb))
            if len(batch) == batch_size or index == len(members) - 1:
                with torch.inference_mode():
                    deep.append(net(torch.stack(batch)).numpy())
                batch.clear()
                print(f"Encoded {index + 1}/{len(members)}", flush=True)
    result = dict(names=np.array(names), image_sha256=np.array(hashes),
                  handcrafted=unit_rows(np.stack(descriptors)),
                  deep=unit_rows(np.concatenate(deep)))
    np.savez_compressed(Path(output) / "features.npz", **result)
    return result, dict(n_images=len(names), image_sizes=dict(sizes), image_modes=dict(modes),
                        duplicate_pixel_pairs=duplicate_pixels,
                        unique_file_hashes=len(set(hashes)), archive_sha256=file_sha256(archive),
                        checkpoint_sha256=file_sha256(checkpoint), checkpoint_name=checkpoint.name,
                        feature_seconds=time.perf_counter() - start,
                        torch_version=torch.__version__, device="cpu", threads=threads)
