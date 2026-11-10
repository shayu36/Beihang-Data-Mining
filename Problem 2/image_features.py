"""Whole-image preprocessing and frozen, multiscale CNN representations."""
import hashlib
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

from course_data import ROOT

GRID = 48
PATCH_COUNT = GRID * GRID
PATCH_DIM = 256
GLOBAL_DIM = 7168


class Encoder:
    def __init__(self, weights=None):
        if weights is None:
            self.net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1, progress=False)
        else:
            self.net = wide_resnet50_2(weights=None)
            self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.net.eval().requires_grad_(False)
        self.transform = Compose([
            Resize((384, 384), interpolation=InterpolationMode.BILINEAR, antialias=True),
            ToTensor(), Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        rng = np.random.default_rng(2026)
        self.projection = torch.from_numpy((rng.standard_normal((1536, PATCH_DIM)) / np.sqrt(PATCH_DIM)).astype("float32"))
        digest = hashlib.sha256()
        for name, tensor in self.net.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.numpy().tobytes())
        self.weights_sha256 = digest.hexdigest()

    @torch.inference_mode()
    def encode(self, images):
        net = self.net
        batch = torch.stack([self.transform(image) for image in images])
        x = net.layer1(net.maxpool(net.relu(net.bn1(net.conv1(batch)))))
        a = net.layer2(x)
        b = net.layer3(a)
        c = net.layer4(b)
        statistics = []
        for layer in (a, b, c):
            statistics.extend((layer.mean(dim=(2, 3)), layer.amax(dim=(2, 3))))
        global_features = torch.cat(statistics, dim=1)
        a = F.avg_pool2d(a, 3, stride=1, padding=1)
        b = F.avg_pool2d(b, 3, stride=1, padding=1)
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        patches = torch.cat((a, b), dim=1).flatten(2).transpose(1, 2) @ self.projection
        return global_features.numpy(), patches.numpy()


def encode_members(archive, names, encoder, batch_size, seen_pixels):
    """Retain one category and split in RAM. No extraction or feature cache is needed."""
    global_x = np.empty((len(names), GLOBAL_DIM), dtype=np.float32)
    local_x = np.empty((len(names), PATCH_COUNT, PATCH_DIM), dtype=np.float32)
    for start in range(0, len(names), batch_size):
        batch = []
        for name in names[start:start + batch_size]:
            with Image.open(BytesIO(archive.read(ROOT + name))) as image:
                rgb = image.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if digest in seen_pixels:
                raise ValueError(f"Pixel-identical samples: {seen_pixels[digest]} and {name}")
            seen_pixels[digest] = name
            batch.append(rgb)
        end = start + len(batch)
        global_x[start:end], local_x[start:end] = encoder.encode(batch)
        if end == len(names) or start % (batch_size * 10) == 0:
            print(f"  images {end}/{len(names)}", flush=True)
    return global_x, local_x
