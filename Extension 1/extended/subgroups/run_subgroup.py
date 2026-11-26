"""Frozen subgroup study. Run freeze before evaluate with --thyroid and --output."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from subgroup_model import CANDIDATES, CONTROL, SubgroupReference


HERE = Path(__file__).resolve().parent
FEATURES = [f"feature_{j}" for j in range(1, 7)]


def dump(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def hash_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def source_hash(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def read(archive, split, labels=False):
    with ZipFile(archive) as z:
        data = z.read(f"thyroid/{split}-set.csv")
    rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    if rows.fieldnames != FEATURES + (["label"] if split == "test" else []):
        raise ValueError("Unexpected data columns")
    if labels:
        return np.asarray([int(float(r["label"])) for r in rows])
    return np.asarray([[float(r[k]) for k in FEATURES] for r in rows])


def key(config):
    return "control" if config == CONTROL else f"k{config[0]}_prior{config[1]}"


def simulate(control, fit, val, seed):
    rng = np.random.default_rng(seed)
    _, neighbors = control.nearest(val, 128)
    mild, stronger = [], []
    for recipient, indices in zip(val, neighbors, strict=True):
        local = fit[indices]
        joint = np.minimum(rankdata(local[:, 1]), rankdata(-local[:, 5]))
        ranked = np.argsort(joint, kind="stable")
        for i in rng.permutation(ranked[-math.ceil(0.1 * len(local)):]):
            donor = local[i]
            lower = ranked[:len(local) // 2]
            mask = (local[lower, 1] < donor[1]) & (local[lower, 5] > donor[5])
            if not mask.any():
                continue
            ref = local[rng.choice(lower[mask])]
            delta = donor - ref
            delta[[0, 4]] = 0
            mild.append(recipient + rng.uniform(0.45, 0.85) * delta)
            stronger.append(recipient + rng.uniform(0.85, 1.25) * delta)
            break
    if not mild:
        raise ValueError("No background-matched donor pairs")
    return {"mild": np.asarray(mild), "stronger": np.asarray(stronger)}


def validate_models(dev):
    evidence = []
    for i, (available, heldout) in enumerate(KFold(3, shuffle=True, random_state=1126).split(dev)):
        order = np.random.default_rng(800 + i).permutation(available)
        n_cal = math.ceil(0.25 * len(order))
        fit, cal, val = dev[order[n_cal:]], dev[order[:n_cal]], dev[heldout]
        control = SubgroupReference(*CONTROL).fit(fit).calibrate(cal)
        simulations = simulate(control, fit, val, seed=2126 + i)
        cohorts = {"normal": val, **simulations}
        near = {}
        for name, x in simulations.items():
            baseline_ratio = control.scores(x) / control.threshold
            near[name] = (baseline_ratio >= 0.5) & (baseline_ratio <= 1.5)
        for config in (CONTROL, *CANDIDATES):
            model = control if config == CONTROL else SubgroupReference(*config).fit(fit).calibrate(cal)
            pred = {name: model.predict(x)["prediction"].astype(bool) for name, x in cohorts.items()}
            evidence.append(dict(fold=i + 1, config=key(config), threshold=model.threshold,
                fit_n=len(fit), calibration_n=len(cal), n={name: len(x) for name,x in cohorts.items()},
                detected={name: int(p.sum()) for name,p in pred.items()},
                near_n={name: int(mask.sum()) for name,mask in near.items()},
                near_detected={name: int(np.sum(pred[name] & mask)) for name,mask in near.items()},
                normal_multiplier_quantiles=np.quantile(model.multipliers(val), [0,0.25,0.5,0.75,1]).tolist()))
    summary = {}
    for config in (CONTROL, *CANDIDATES):
        rows = [r for r in evidence if r["config"] == key(config)]
        sums = {field: {name: sum(r[field][name] for r in rows) for name in rows[0][field]}
                for field in ("n","detected","near_n","near_detected")}
        rates = {name: v / sums["n"][name] for name,v in sums["detected"].items()}
        near_rates = {name: v / sums["near_n"][name] for name,v in sums["near_detected"].items()}
        summary[key(config)] = dict(**sums, rates=rates, near_rates=near_rates,
            mean_simulated=(rates["mild"] + rates["stronger"]) / 2)
    return evidence, summary


def freeze(args):
    train = read(args.thyroid, "train")
    _, keep = np.unique(train, axis=0, return_index=True)
    order = np.random.default_rng(42).permutation(np.sort(keep))
    n_cal = math.ceil(0.2 * len(order))
    dev, cal = train[np.sort(order[n_cal:])], train[np.sort(order[:n_cal])]
    evidence, summary = validate_models(dev)
    eligible = [c for c in CANDIDATES if summary[key(c)]["rates"]["normal"] <= 0.005]
    if eligible:
        winner = max(eligible, key=lambda c: (summary[key(c)]["mean_simulated"], -summary[key(c)]["rates"]["normal"]))
    else:
        winner = min(CANDIDATES, key=lambda c: (summary[key(c)]["rates"]["normal"], -summary[key(c)]["mean_simulated"]))
    model = SubgroupReference(*winner).fit(dev).calibrate(cal)
    args.output.mkdir(parents=True, exist_ok=True)
    mp = args.output / "subgroup_model.joblib"
    joblib.dump(model, mp)
    frozen = dict(selected=key(winner), eligible=[key(c) for c in eligible], summary=summary, folds=evidence,
        threshold=model.threshold, global_q99=model.global_quantile, fit_n=len(dev), calibration_n=len(cal),
        k=winner[0], prior=winner[1], background_columns_1based=[1,5], model_sha256=hash_file(mp),
        calibration_multiplier_5number=np.quantile(model.multipliers(cal), [0,.25,.5,.75,1]).tolist(),
        source_sha256={p: source_hash(HERE/p) for p in ("protocol.json","subgroup_model.py","run_subgroup.py")},
        test_data_accessed=False)
    dump(args.output / "freeze.json", frozen)
    for name, s in summary.items():
        print(name, s["rates"], "near", s["near_rates"], "combined", s["mean_simulated"])
    print("Selected", frozen["selected"], "threshold", model.threshold, "global fit 99th percentile", model.global_quantile)


def evaluate(args):
    fp = args.output / "freeze.json"
    frozen = json.loads(fp.read_text(encoding="utf-8"))
    mp = args.output / "subgroup_model.joblib"
    if hash_file(mp) != frozen["model_sha256"] or any(source_hash(HERE/p) != h for p,h in frozen["source_sha256"].items()):
        raise ValueError("Changed since freeze")
    model = joblib.load(mp)
    test = read(args.thyroid, "test")
    result = model.predict(test)
    scores, pred = result["score"], result["prediction"]
    pre = args.output / "scores_before_labels.csv"
    with pre.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_1based","raw_score","multiplier","score","threshold","prediction"])
        raw, mult = model.raw_score(test), model.multipliers(test)
        writer.writerows((i + 1, float(r), float(m), float(s), model.threshold, int(p))
                         for i,(r,m,s,p) in enumerate(zip(raw,mult,scores,pred,strict=True)))
    y = read(args.thyroid, "test", labels=True)
    if len(y) != len(scores) or not np.array_equal(np.unique(y), [0,1]):
        raise ValueError("Invalid evaluation labels")
    tn,fp_count,fn,tp = map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    out = dict(selected=frozen["selected"], n=len(y), tn=tn, fp=fp_count, fn=fn, tp=tp,
        accuracy=(tn+tp)/len(y), precision=tp/(tp+fp_count), recall=tp/(tp+fn),
        f1=2*tp/(2*tp+fp_count+fn), auroc=float(roc_auc_score(y,scores)),
        average_precision=float(average_precision_score(y,scores)),
        freeze_sha256=hash_file(fp), scores_before_labels_sha256=hash_file(pre))
    dump(args.output / "course_evaluation.json", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze","evaluate"))
    p.add_argument("--thyroid", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        (freeze if args.stage == "freeze" else evaluate)(args)
