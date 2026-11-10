"""Frozen ImageNet ResNet-18 adapter for the cross-modal experiment.

The whole image is resized to 256 x 256, rather than center-cropped.  This
deliberate change from the classifier's default transform retains image edges.
Layer 2/3/4 channel-wise spatial means and maxima form a 1,792-D vector.
No course labels are read here and no encoder weights are trained.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor


IMAGE_SIZE = 256
FEATURE_DIM = 1792
DEFAULT_BATCH_SIZE = 8


class ImageEncoder:
    """CPU-only, frozen ResNet-18 with optional local official state dictionary.

    ``weights=None`` obtains torchvision's official IMAGENET1K_V1 weights,
    downloading them only if absent from the local torch cache.  A supplied
    path must contain a compatible raw state_dict, not a serialized model.
    ``weights_sha256`` fingerprints the loaded parameter names and tensor bytes
    in state_dict iteration order, rather than the serialized checkpoint file.
    """

    def __init__(self, weights: str | Path | None = None):
        torch.use_deterministic_algorithms(True)
        if weights is None:
            self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=False)
        else:
            weight_path = Path(weights)
            if not weight_path.is_file():
                raise FileNotFoundError(weight_path)
            self.net = resnet18(weights=None)
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            self.net.load_state_dict(state, strict=True)
        self.net = self.net.cpu().eval().requires_grad_(False)
        self.transform = Compose([
            Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR,
                   antialias=True),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        digest = hashlib.sha256()
        for name, tensor in self.net.state_dict().items():
            array = tensor.detach().cpu().contiguous().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite encoder parameter: {name}")
            digest.update(name.encode("utf-8"))
            digest.update(array.tobytes())
        self.weights_sha256 = digest.hexdigest()

    @torch.inference_mode()
    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return a finite float32 (N, 1792) array in input image order."""
        if len(images) == 0:
            return np.empty((0, FEATURE_DIM), dtype=np.float32)
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("encode expects a sequence of PIL.Image.Image objects")
        batch = torch.stack([self.transform(image.convert("RGB")) for image in images])
        net = self.net
        x = net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(batch)))))
        a = net.layer2(x)
        b = net.layer3(a)
        c = net.layer4(b)
        statistics = []
        for layer in (a, b, c):
            statistics.extend((layer.mean(dim=(2, 3)), layer.amax(dim=(2, 3))))
        features = torch.cat(statistics, dim=1).cpu().numpy().astype(np.float32, copy=False)
        if features.shape != (len(images), FEATURE_DIM) or not np.isfinite(features).all():
            raise ValueError("Invalid or non-finite image feature matrix")
        return features


def encode_members(
    zip_path: str | Path,
    members: Sequence[str],
    encoder: ImageEncoder,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Read full ZIP member paths without extraction and encode in given order.

    Only the named images are read. ZIP metadata and filenames are used for
    retrieval and progress messages, never as learned input features.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    members = list(members)
    if len(set(members)) != len(members):
        raise ValueError("Duplicate requested ZIP members")
    output = np.empty((len(members), FEATURE_DIM), dtype=np.float32)
    with ZipFile(zip_path) as archive:
        available = {}
        for info in archive.infolist():
            available[info.filename] = available.get(info.filename, 0) + 1
        for member in members:
            if available.get(member, 0) != 1:
                raise ValueError(f"Expected exactly one ZIP member: {member}")
        for start in range(0, len(members), batch_size):
            batch = []
            for member in members[start:start + batch_size]:
                with Image.open(BytesIO(archive.read(member))) as source:
                    batch.append(source.convert("RGB"))
            end = start + len(batch)
            output[start:end] = encoder.encode(batch)
            print(f"  image features: {end}/{len(members)}", flush=True)
    return output
