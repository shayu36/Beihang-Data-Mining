"""Score one image or one six-feature thyroid record with a saved domain model."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--values", type=float, nargs=6, metavar="FEATURE")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    payload = joblib.load(args.model)
    if payload.get("version") != 4:
        parser.error("Use a version-4 model produced by the current run.py")
    domain, model = payload["domain"], payload["model"]
    with threadpool_limits(limits=args.threads):
        if domain == "thyroid":
            if args.values is None:
                parser.error("A thyroid model requires --values in feature_1 to feature_6 order")
            x = np.asarray([args.values], dtype=np.float64)
        else:
            if args.image is None:
                parser.error("An image model requires --image")
            import torch
            from PIL import Image
            from local_features import ImageLocalEncoder
            torch.set_num_threads(args.threads)
            encoder = ImageLocalEncoder(args.weights)
            if (encoder.weights_sha256 != payload["weights_sha256"] or
                    encoder.projection_sha256 != payload["projection_sha256"]):
                raise ValueError("Feature extractor does not match the trained model")
            with Image.open(args.image) as image:
                global_x, x = encoder.encode([image.convert("RGB")])
        result = model.predict(x) if domain == "thyroid" else model.predict(global_x, x)
    answer = dict(domain=domain, method="improved", model_kind=payload["kind"],
        score=float(result["score"][0]), threshold=model.threshold,
        prediction=int(result["prediction"][0]))
    if "p_normal" in result:
        answer["p_normal"] = float(result["p_normal"][0])
    print(json.dumps(answer, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
