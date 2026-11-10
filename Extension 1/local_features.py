"""Frozen global and local image representations from one ResNet-18 pass.

The global output is identical to ``image_adapter.ImageEncoder``.  The local
output retains a 16 x 16 spatial grid instead of collapsing spatial structure.
It is intended as input to a separately fitted normal-patch memory bank.

This is a compact local-feature adapter inspired by normal feature memories;
it is not a complete PatchCore implementation.  It reads no labels, chooses
no model hyperparameters and does not change the caller's CPU thread count.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from image_adapter import (
    DEFAULT_BATCH_SIZE,
    FEATURE_DIM as GLOBAL_DIM,
    IMAGE_SIZE,
    ImageEncoder,
)


GRID = 16
PATCH_COUNT = GRID * GRID
RAW_PATCH_DIM = 128 + 256
PATCH_DIM = 64
PROJECTION_SEED = 2026


class ImageLocalEncoder(ImageEncoder):
    """Return compatible global vectors and compact local patch descriptors.

    ``weights`` has exactly the same meaning as for ``ImageEncoder``: None
    loads torchvision's official ResNet18 IMAGENET1K_V1 checkpoint; a local
    path loads a compatible state dictionary for offline execution.

    Both branches use a single frozen forward pass on the entire RGB image
    resized to 256 x 256. Local features use the following fixed operations:

    1. Smooth layer2 (128 x 32 x 32) and layer3 (256 x 16 x 16) separately
       with 3 x 3, stride-1 average pooling. At boundaries, padded zeros are
       excluded from the average (``count_include_pad=False``).
    2. Bilinearly upsample layer3 to the layer2 grid with align_corners=False.
    3. Concatenate channels, then average-pool to a 16 x 16 descriptor grid.
    4. Multiply each 384-D descriptor by a fixed Gaussian 384 x 64 matrix,
       drawn with NumPy PCG64 seed 2026 and scaled by 1/sqrt(64).

    Patch order is row-major: index ``r * GRID + c`` identifies grid cell
    (r, c). Projection parameters are data-independent and are never fitted.
    ``weights_sha256`` is inherited and exactly matches the global encoder;
    ``projection_sha256`` fingerprints the float32 projection matrix bytes.
    """

    def __init__(self, weights: str | Path | None = None):
        super().__init__(weights)
        rng = np.random.default_rng(PROJECTION_SEED)
        projection = (rng.standard_normal((RAW_PATCH_DIM, PATCH_DIM))
                      / np.sqrt(PATCH_DIM)).astype(np.float32)
        self.projection = torch.from_numpy(projection)
        self.projection_sha256 = hashlib.sha256(projection.tobytes()).hexdigest()

    @torch.inference_mode()
    def encode(self, images: Sequence[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(global[N,1792], patches[N,256,64])`` as finite float32.

        Caller-controlled batching bounds memory. An empty sequence returns
        matching empty arrays. No image or patch identifiers are features.
        """
        if len(images) == 0:
            return (np.empty((0, GLOBAL_DIM), dtype=np.float32),
                    np.empty((0, PATCH_COUNT, PATCH_DIM), dtype=np.float32))
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("encode expects a sequence of PIL.Image.Image objects")

        batch = torch.stack([self.transform(image.convert("RGB")) for image in images])
        net = self.net
        x = net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(batch)))))
        a = net.layer2(x)
        b = net.layer3(a)
        c = net.layer4(b)

        # Preserve the original global representation exactly, including its
        # layer/mean/max concatenation order. Local smoothing is out-of-place.
        statistics = []
        for layer in (a, b, c):
            statistics.extend((layer.mean(dim=(2, 3)), layer.amax(dim=(2, 3))))
        global_features = torch.cat(statistics, dim=1).cpu().numpy().astype(np.float32, copy=False)

        local_a = F.avg_pool2d(a, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        local_b = F.avg_pool2d(b, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        local_b = F.interpolate(local_b, size=local_a.shape[-2:], mode="bilinear", align_corners=False)
        local = F.adaptive_avg_pool2d(torch.cat((local_a, local_b), dim=1), (GRID, GRID))
        patches = (local.flatten(2).transpose(1, 2) @ self.projection).cpu().numpy()
        patches = np.ascontiguousarray(patches, dtype=np.float32)

        if global_features.shape != (len(images), GLOBAL_DIM) or not np.isfinite(global_features).all():
            raise ValueError("Invalid or non-finite global feature matrix")
        if patches.shape != (len(images), PATCH_COUNT, PATCH_DIM) or not np.isfinite(patches).all():
            raise ValueError("Invalid or non-finite local feature matrix")
        return global_features, patches


def encode_members(
    zip_path: str | Path,
    members: Sequence[str],
    encoder: ImageLocalEncoder,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode named ZIP images without extraction, preserving input order.

    The function reads only the selected images and never opens label files.
    ``members`` must contain full ZIP member paths, as in the global adapter.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    members = list(members)
    if len(set(members)) != len(members):
        raise ValueError("Duplicate requested ZIP members")
    global_features = np.empty((len(members), GLOBAL_DIM), dtype=np.float32)
    local_features = np.empty((len(members), PATCH_COUNT, PATCH_DIM), dtype=np.float32)
    with ZipFile(zip_path) as archive:
        available = Counter(info.filename for info in archive.infolist())
        for member in members:
            if available[member] != 1:
                raise ValueError(f"Expected exactly one ZIP member: {member}")
        for start in range(0, len(members), batch_size):
            batch = []
            for member in members[start:start + batch_size]:
                with Image.open(BytesIO(archive.read(member))) as source:
                    batch.append(source.convert("RGB"))
            end = start + len(batch)
            global_features[start:end], local_features[start:end] = encoder.encode(batch)
            print(f"  global/local features: {end}/{len(members)}", flush=True)
    return global_features, local_features
