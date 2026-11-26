"""Select one normal-reference distance for all domains, freeze it, then score without labels."""

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits


SCALES = (5, 20)
POWERS = (0.0, 0.5, 1.0)
METHODS = ("single_5", "multiscale_0", "multiscale_0.5", "multiscale_1")
ALPHA = 0.05
PROFILES = ("coordinate_swap", "sparse_shift")


def array_digest(data):
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


class MultiScaleReference:
    def __init__(self, normal_bags):
        from normal_bags import NormalBagDetector

        self.core = NormalBagDetector(metric="standard", distance="euclidean", k=5, seed=42,
                                      top_fraction=0.02, max_bank=8192).fit(normal_bags)
        self.radii = {}
        self.centers = {}
        self.floors = {}
        # With X omitted, sklearn excludes the query reference point itself.
        radii = self.core.neighbors.kneighbors(n_neighbors=max(SCALES), return_distance=True)[0]
        for k in SCALES:
            local_radii = np.mean(radii[:, :k], axis=1) / np.sqrt(self.core.dimension)
            self.radii[k] = local_radii
            self.centers[k] = float(np.median(local_radii))
            self.floors[k] = max(float(np.quantile(local_radii, 0.1)), np.finfo(float).eps)
            if self.centers[k] <= 0:
                raise ValueError("Normal reference has no positive median distance")

    def score_candidates(self, bags):
        n, patches, dim = bags.shape
        transformed = self.core.transform(bags.reshape(-1, dim).astype(np.float64))
        all_patch_scores = np.empty((len(METHODS), len(transformed)))
        for start in range(0, len(transformed), 1024):
            zs = transformed[start:start + 1024]
            dist, neigh = self.core.neighbors.kneighbors(zs, n_neighbors=max(SCALES))
            terms = []
            for k in SCALES:
                dk = dist[:, :k].mean(axis=1) / np.sqrt(dim)
                if k == 5:
                    all_patch_scores[0, start:start + len(zs)] = dk
                nearby_radius = np.maximum(self.radii[k][neigh[:, :k]].mean(axis=1), self.floors[k])
                terms.append([dk / (self.centers[k] ** (1 - b) * nearby_radius ** b) for b in POWERS])
            # Two scales are equally weighted in log space; a zero distance stays zero.
            all_patch_scores[1:, start:start + len(zs)] = np.sqrt(np.asarray(terms[0]) * terms[1])
        scores = all_patch_scores.reshape(len(METHODS), n, patches)
        count = max(1, ceil(0.02 * patches))
        top = np.partition(scores, patches - count, axis=-1)[:, :, -count:]
        return dict(zip(METHODS, top.mean(axis=-1), strict=True))


def synthetic_bags(fit, held, seed):
    # The same bag- and coordinate-fraction rules apply to image patches and records.
    rng = np.random.default_rng(seed)
    _, patches, dim = held.shape
    n_patches, n_coordinates = ceil(0.05 * patches), ceil(np.sqrt(dim))
    q10, q90 = np.quantile(fit.reshape(-1, dim), [0.1, 0.9], axis=0)
    swapped, shifted = held.copy(), held.copy()
    for i in range(len(held)):
        p = rng.choice(patches, n_patches, replace=False)
        c = rng.choice(dim, n_coordinates, replace=False)
        ix = np.ix_(p, c)
        # The donor is sampled from the inner-fit normals and has the same patch coordinates.
        donor = fit[rng.integers(len(fit))]
        swapped[i][ix] = donor[ix]
        jump = rng.uniform(1, 2, n_coordinates) * rng.choice([-1, 1], n_coordinates) * (q90[c] - q10[c])
        shifted[i][ix] += jump
    return {"coordinate_swap": swapped, "sparse_shift": shifted}


def train_selection(cache, domains, splits):
    records = []
    for domain in domains:
        development = cache[f"{domain}_bag_train"][splits[domain]["fit"]]
        for fold, (fit, valid) in enumerate(KFold(3, shuffle=True, random_state=2026).split(development)):
            training, held = development[fit], development[valid]
            detector = MultiScaleReference(training)
            normal = detector.score_candidates(held)
            for profile, proxy in synthetic_bags(training, held, seed=32026 + fold).items():
                positive = detector.score_candidates(proxy)
                y = np.r_[np.zeros(len(held)), np.ones(len(held))]
                aps = {method: float(average_precision_score(y, np.r_[normal[method], positive[method]]))
                       for method in METHODS}
                records.append(dict(domain=domain, fold=fold + 1, inner_fit_count=len(fit),
                                    held_count=len(held), profile=profile, ap=aps))
            print(f"Training-only selection {domain} {fold + 1}/3", flush=True)
    mean_ap = {m: float(np.mean([r["ap"][m] for r in records])) for m in METHODS}
    # The original baseline is a reference, rather than a member of the alternative family.
    selected = max(METHODS[1:], key=lambda m: mean_ap[m])
    return dict(winner=selected, macro_proxy_ap=mean_ap, fold_profile_ap=records)


def run(source, output):
    from course_data import DOMAINS, sha256, split_normal
    from experiment import save_csv, save_json
    from unified import conformal_threshold, normal_p_values

    old_run_info = json.loads((source / "run_info.json").read_text(encoding="utf-8"))
    old_frozen = json.loads((source / "frozen.json").read_text(encoding="utf-8"))
    cache_path = source / "features.npz"
    cache_hash = sha256(cache_path)
    if cache_hash != old_run_info["features_sha256"]:
        raise ValueError("Feature cache hash differs from the original run")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", dict(
        candidates=list(METHODS), local_powers=list(POWERS), scales=list(SCALES), radius_floor_quantile=0.1,
        alpha=ALPHA, shared_top_fraction=0.02, calibration_rule="upper-tail normal rank <= 0.05",
        selection="Highest arithmetic mean synthetic-vs-normal AP over 3 domains, 3 inner folds "
                  "and 2 corruption profiles among multiscale candidates. Ties use smaller power.",
        inner_split="3-fold with shuffle seed 2026, on original proper-fit normals only",
        profiles="For each held-out bag select ceil(0.05 * bag length) random descriptors and "
                 "ceil(sqrt(dimension)) random coordinates, shared across these descriptors. "
                 "Swap coordinates for the same positions of a randomly sampled fit bag, or "
                 "add a shared signed Uniform(1,2) * (Q90-Q10) shift computed from inner-fit data.",
        score="For k in {5,20}, divide mean standardized neighbor distance by median training "
              "radius^(1-b) times local radius^b. Local radius is the mean leave-self-out radius "
              "of the k neighbors, floored at the tenth percentile of training radii. Use the "
              "geometric mean across k; average the top ceil(0.02*m) patch values.",
        script_sha256=sha256(Path(__file__)), features_sha256=cache_hash))
    with np.load(cache_path, allow_pickle=False) as cache:
        splits = {}
        for domain in DOMAINS:
            normals = np.flatnonzero(cache[f"{domain}_train_labels"] == 0)
            fit, cal, _ = split_normal(cache[f"{domain}_global_train"][normals], seed=42,
                                       fraction=0.2, deduplicate=domain == "thyroid")
            splits[domain] = {"fit": normals[fit], "cal": normals[cal]}
        selection = train_selection(cache, DOMAINS, splits)
        save_json(output / "training_selection.json", selection)
        winner = selection["winner"]
        state, models = {}, {}
        for domain in DOMAINS:
            tr = cache[f"{domain}_bag_train"]
            fit, cal = splits[domain]["fit"], splits[domain]["cal"]
            detector = MultiScaleReference(tr[fit])
            calibrated = detector.score_candidates(tr[cal])
            models[domain] = (detector, calibrated)
            state[domain] = dict(fit_indices=fit.tolist(), calibration_indices=cal.tolist(),
                                 reference_sha256=array_digest(detector.core.reference),
                                 radii_sha256={str(k): array_digest(detector.radii[k]) for k in SCALES},
                                 selected_calibration_sha256=array_digest(calibrated[winner]),
                                 selected_threshold=conformal_threshold(calibrated[winner], ALPHA),
                                 original_threshold=conformal_threshold(calibrated["single_5"], ALPHA))
        save_json(output / "frozen.json", dict(winner=winner, alpha=ALPHA, domains=state,
                  features_sha256=cache_hash, archives_sha256=old_frozen["archives_sha256"],
                  protocol_sha256=sha256(output / "protocol.json"),
                  selection_sha256=sha256(output / "training_selection.json")))
        # The winner, all reference banks and calibration thresholds are already frozen.
        names = json.loads((source / "sample_names.json").read_text(encoding="utf-8"))["test"]
        rows = []
        for domain in DOMAINS:
            detector, calibrated = models[domain]
            scores = detector.score_candidates(cache[f"{domain}_bag_test"])
            for method in ("single_5", winner):
                rank = normal_p_values(scores[method], calibrated[method])
                cut = conformal_threshold(calibrated[method], ALPHA)
                if not np.array_equal(rank <= ALPHA, scores[method] > cut):
                    raise ValueError("Normal-rank and threshold disagree")
                for name, s, p in zip(names[domain], scores[method], rank, strict=True):
                    rows.append(dict(domain=domain, sample=name, method=method,
                                     score=float(s), threshold=cut, p_normal=float(p),
                                     prediction=int(p <= ALPHA)))
    save_csv(output / "scores_before_labels.csv", rows)
    save_json(output / "scored.json", dict(frozen_sha256=sha256(output / "frozen.json"),
              scores_sha256=sha256(output / "scores_before_labels.csv"), rows=len(rows)))
    print(f"Frozen alternative {winner}; {len(rows)} unlabeled scores including baseline", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Original features and split provenance")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--implementation", type=Path, help="Directory with normal_bags.py")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.implementation:
        sys.path.insert(0, str(args.implementation.resolve()))
    with threadpool_limits(limits=args.threads):
        run(args.run, args.output)
