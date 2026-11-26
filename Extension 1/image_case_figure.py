"""Plot the closest correctly detected anomaly to each frozen image threshold."""

import argparse
import csv
import io
import json
from math import ceil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import joblib
import matplotlib
import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits

from course_data import sha256
from local_features import GRID, ImageLocalEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DOMAINS = ("hazelnut", "zipper")


def select_cases(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    selected = {}
    for domain in DOMAINS:
        candidates = [r for r in rows if r["method"] == "improved" and r["domain"] == domain
                      and r["label"] == "1" and r["prediction"] == "1"]
        if not candidates:
            raise ValueError(f"No true-positive image in {domain}")
        if any(r["decision_rule"] != ">=" or float(r["score"]) < float(r["threshold"])
               for r in candidates):
            raise ValueError(f"Prediction rule inconsistent for {domain}")
        row = min(candidates, key=lambda r: (float(r["score"]) - float(r["threshold"]), r["sample"]))
        selected[domain] = (row, len(candidates))
    return selected


def build_case(domain, selected, run, archive, names, cache, encoder, frozen):
    row, n_candidates = selected
    sample = row["sample"]
    index = names["test"][domain].index(sample)
    model_path = run / f"{domain}.joblib"
    if sha256(model_path) != frozen["domains"][domain]["model_sha256"]:
        raise ValueError(f"{domain}: model differs from frozen manifest")
    payload = joblib.load(model_path)
    if (payload["domain"] != domain or payload["weights_sha256"] != encoder.weights_sha256
            or payload["projection_sha256"] != encoder.projection_sha256):
        raise ValueError(f"{domain}: model and image encoder mismatch")
    model = payload["model"]
    with Image.open(io.BytesIO(archive.read(sample))) as source:
        rgb = source.convert("RGB")
    extracted_global, extracted_local = encoder.encode([rgb])
    gx = cache[f"{domain}_global_test"][index:index + 1]
    lx = cache[f"{domain}_bag_test"][index:index + 1]
    # A single-image convolution can have a different floating-point accumulation
    # order from extraction in batches. Retain frozen values for the reported scores.
    np.testing.assert_allclose(extracted_global, gx, atol=2e-5, rtol=1e-5)
    np.testing.assert_allclose(extracted_local, lx, atol=2e-5, rtol=1e-5)
    branches = model.branch_scores(gx, lx)
    fused = float(branches["fusion"][0])
    np.testing.assert_allclose([fused, model.threshold], [float(row["score"]), float(row["threshold"])],
                               rtol=1e-10, atol=1e-10)
    local = model.local_model
    values = local.descriptor_scores(lx)[0]
    grid = values.reshape(GRID, GRID, order="C")
    k = ceil(local.top_fraction * grid.size)
    np.testing.assert_allclose(np.sort(grid, axis=None)[-k:].mean(), branches["local"][0],
                               rtol=1e-10, atol=1e-10)
    peak = tuple(map(int, np.unravel_index(np.argmax(grid), grid.shape)))
    # Match the peak's row-major descriptor to its standardized nearest-reference
    # distance independently. ImageLocalEncoder indexes the bag by row * GRID + column.
    flat_index = peak[0] * GRID + peak[1]
    z = local.transform(lx[0, flat_index:flat_index + 1].astype(float))
    distances = np.linalg.norm(local.reference - z, axis=1) / np.sqrt(local.dimension)
    np.testing.assert_allclose(np.sort(distances)[:local.k].mean(), grid[peak], rtol=1e-10, atol=1e-10)
    record = {
        "domain": domain, "sample": sample, "true_positive_candidates": n_candidates,
        "score": fused, "threshold": model.threshold, "margin": fused - model.threshold,
        "svm_weight": model.svm_weight, "image_width_height": list(rgb.size),
        "patch_grid_rows_columns": list(grid.shape), "grid_peak_row_col_zero_based": list(peak),
        "patch_distance_min": float(grid.min()), "patch_distance_max": float(grid.max()),
        "top_patch_count": k, "local_branch_score": float(branches["local"][0]),
        "max_raw_feature_difference": {
            "global": float(np.max(np.abs(extracted_global - gx))),
            "local": float(np.max(np.abs(extracted_local - lx))),
        },
    }
    return np.asarray(rgb), grid, record


def render(cases, prefix):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    width, height = 7.2, 2.30
    fig = plt.figure(figsize=(width, height), facecolor="white")
    vmax = ceil(max(grid.max() for _, grid, _ in cases) * 10) / 10
    for column, (image, grid, case) in enumerate(cases):
        left = 0.014 + 0.463 * column
        panel_width = 0.212
        panel_height = panel_width * width / height
        title = case["domain"].title() + "  ·  " + PurePosixPath(case["sample"]).name
        fig.text(left, 0.965, title, fontsize=10.5, fontweight="semibold", va="top", color="#152334")
        for j, label in enumerate(["Original image", "Patch distance"]):
            x = left + j * 0.219
            ax = fig.add_axes([x, 0.14, panel_width, panel_height])
            ax.set_axis_off()
            fig.text(x, 0.820, label, fontsize=7.8, color="#384452", va="bottom")
            if j == 0:
                ax.imshow(image, interpolation="antialiased", origin="upper", aspect="auto")
            else:
                mapped = ax.imshow(grid, interpolation="nearest", origin="upper", aspect="equal",
                                   cmap="inferno", vmin=0, vmax=vmax)
        fig.text(left, 0.042, f"Fusion score  {case['score']:.3f}      Threshold  {case['threshold']:.3f}",
                 fontsize=8, color="#263545", va="bottom")
    cax = fig.add_axes([0.927, 0.14, 0.012, panel_height])
    colorbar = fig.colorbar(mapped, cax=cax, ticks=np.linspace(0, vmax, 4), format="%.2f")
    colorbar.ax.tick_params(labelsize=6.7, length=2.3, width=0.45, pad=2)
    colorbar.outline.set_linewidth(0.4)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), dpi=240, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(prefix.with_suffix(".png"), dpi=240)
    plt.close(fig)
    return vmax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Original course image ZIP")
    parser.add_argument("--run", type=Path, required=True, help="Frozen experiment output directory")
    parser.add_argument("--output", type=Path, required=True, help="Output prefix for PDF, PNG and JSON")
    parser.add_argument("--weights", type=Path, help="Optional offline ResNet18 state_dict")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    run = args.run
    frozen = json.loads((run / "frozen.json").read_text(encoding="utf-8"))
    if sha256(args.images) != frozen["archives_sha256"]["images"]:
        raise ValueError("Image archive differs from frozen manifest")
    selected = select_cases(run / "test_predictions.csv")
    names = json.loads((run / "sample_names.json").read_text(encoding="utf-8"))
    encoder = ImageLocalEncoder(args.weights)
    with ZipFile(args.images) as archive, np.load(run / "features.npz", allow_pickle=False) as cache:
        with threadpool_limits(limits=args.threads):
            cases = [build_case(d, selected[d], run, archive, names, cache, encoder, frozen) for d in DOMAINS]
    vmax = render(cases, args.output)
    record = {
        "selection": "Per domain, among correctly detected anomalies choose the smallest fused score "
                     "minus threshold; break ties by sample path. Illustration after evaluation.",
        "interpretation": "Each cell is a normal-reference distance from the fitted local branch. "
                          "The fusion also uses a global SVM. No pixel-level masks are used.",
        "display": {"origin": "top left", "grid_order": "row-major", "interpolation": "nearest",
                    "shared_color_limits": [0, vmax], "figure_inches": [7.2, 2.30], "dpi": 240},
        "cases": [record for _, _, record in cases],
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
