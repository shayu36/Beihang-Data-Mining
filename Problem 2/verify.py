"""Audit an experiment against the course ZIP; write evidence as Markdown in the run folder."""
import argparse
import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from course_data import CATEGORIES, ROOT, image_lists, official_labels
from detector import score_detector
from image_features import Encoder, encode_members


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def close(actual, expected, description, tolerance=1e-10):
    if not np.allclose(np.asarray(actual, dtype=float), expected, rtol=0, atol=tolerance):
        raise ValueError(f"{description}: {actual} != {expected}")


def independent_metrics(y, scores, threshold):
    """AUROC from positive/negative pairs, AP by tied score groups; no sklearn metrics."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = y == 1
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(positive.sum()) - tp
    tn = len(y) - tp - fp - fn
    pairs = scores[positive, None] - scores[~positive]
    auc = float((np.count_nonzero(pairs > 0) + .5 * np.count_nonzero(pairs == 0)) / pairs.size)
    order = np.argsort(-scores, kind="stable")
    stops = np.r_[scores[order][:-1] != scores[order][1:], True]
    cum_tp = np.cumsum(y[order])[stops]
    ranks = np.arange(1, len(y) + 1)[stops]
    ap = float(np.sum(np.diff(np.r_[0, cum_tp]) * cum_tp / ranks) / positive.sum())
    return dict(n=len(y), normal=tn+fp, abnormal=tp+fn, tn=tn, fp=fp, fn=fn, tp=tp,
                accuracy=(tp+tn)/len(y), precision=tp/max(tp+fp, 1), recall=tp/(tp+fn),
                f1=2*tp/(2*tp+fp+fn), balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)), auroc=auc, ap=ap)


def verify_metrics(record, y, scores, threshold):
    for name, value in independent_metrics(y, scores, threshold).items():
        close(float(record[name]), value, "metric " + name)


def exhaustive_threshold(y, scores):
    """Reference implementation: evaluate the predictions at every admissible cut at once."""
    distinct = np.unique(scores)
    cuts = np.r_[np.nextafter(distinct[0], -np.inf), distinct[:-1] + np.diff(distinct)/2,
                 np.nextafter(distinct[-1], np.inf)]
    predicted = scores[None, :] >= cuts[:, None]
    tp = predicted[:, y == 1].sum(axis=1)
    fp = predicted[:, y == 0].sum(axis=1)
    recall = tp/(y == 1).sum()
    negatives = (y == 0).sum()
    specificity = (negatives - fp)/negatives
    f1 = 2*tp/((y == 1).sum() + tp + fp)
    winner = max(range(len(cuts)), key=lambda i: (f1[i], .5*(recall[i]+specificity[i]), recall[i], -cuts[i]))
    return cuts[winner]


def verify_candidates(y, matrix, trials, settings):
    require(len(trials) == len(settings) == matrix.shape[1], "Candidate count mismatch")
    for i, trial in enumerate(trials):
        require(trial["column"] == i and trial["setting"] == settings[i], "Candidate order mismatch")
        values = matrix[:, i]
        cut = exhaustive_threshold(y, values)
        close(trial["threshold"], cut, "out-of-fold threshold")
        verify_metrics(trial, y, values, cut)
    return max(trials, key=lambda r: (r["f1"], r["balanced_accuracy"], r["auroc"]))


def verify_search(selection, config, frozen, expected_ids, all_y, n_folds):
    oof = selection["oof"]
    ids = np.asarray(oof["image_ids"])
    np.testing.assert_array_equal(ids, expected_ids)
    y = all_y[ids]
    np.testing.assert_array_equal(oof["labels"], y)
    folds = np.array(oof["folds"])
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=frozen["seed"])
    for fold, (_, validation) in enumerate(cv.split(ids, y)):
        np.testing.assert_array_equal(np.flatnonzero(folds == fold), validation)
    columns = []
    for name, grid_name, selected_name in (("global", "svm_grid", "svm"), ("local", "fractions", "fraction")):
        matrix = np.asarray(oof[name + "_scores"])
        best = verify_candidates(y, matrix, selection[name + "_trials"], frozen[grid_name])
        require(best["setting"] == config[selected_name], f"Selected {name} setting mismatch")
        close(config["thresholds"][name], best["threshold"], name + " cutoff")
        scores = matrix[:, best["column"]]
        q10, median, q90 = np.percentile(scores[y == 0], [10, 50, 90])
        spread = max(q90-q10, 1e-8)
        close(config[name + "_scale"], [median, spread], name + " calibration")
        columns.append((scores - median)/spread)
    weights = frozen["local_weights"]
    fusion = np.column_stack([(1-a)*columns[0] + a*columns[1] for a in weights])
    best = verify_candidates(y, fusion, selection["fusion_trials"], weights)
    require(best["setting"] == config["local_weight"], "Fusion weight mismatch")
    close(config["thresholds"]["fusion"], best["threshold"], "fusion cutoff")


def compare_runs(first, second):
    require(first.resolve() != second.resolve(), "Provide two different experiment directories")
    first_info, second_info = (read_json(folder/"run_info.json") for folder in (first, second))
    for key in ("archive_sha256", "batch_size", "cpu_threads"):
        require(first_info[key] == second_info[key], f"Repeated runs differ in {key}")
    left_frozen, right_frozen = (read_json(folder/"frozen.json") for folder in (first, second))
    require(left_frozen["frozen_at"] != right_frozen["frozen_at"], "Both directories contain the same training run")
    require(left_frozen["weights_sha256"] == right_frozen["weights_sha256"], "Repeated runs used different backbones")
    largest = 0.
    left, right = (read_csv(folder/"scores_before_labels.csv") for folder in (first, second))
    require(len(left) == len(right), "Repeated run sample count mismatch")
    for a, b in zip(left, right):
        require(a["path"] == b["path"] and a.keys() == b.keys(), "Repeated run path/schema mismatch")
        for key, value in a.items():
            if key in ("path", "category"):
                require(value == b[key], "Repeated run identity mismatch")
            else:
                diff = abs(float(value) - float(b[key]))
                require(diff <= 1e-5, f"Repeated run mismatch: {a['path']} {key}")
                largest = max(largest, diff)
    return largest


def audit(args):
    folder = args.run
    info = read_json(folder / "run_info.json")
    frozen = read_json(folder / "frozen.json")
    require(file_hash(args.archive) == info["archive_sha256"], "Course archive SHA256 mismatch")
    encoder = Encoder(args.weights)
    require(encoder.weights_sha256 == frozen["weights_sha256"], "Backbone weights mismatch")
    require(not encoder.net.training and all(not p.requires_grad for p in encoder.net.parameters()), "Backbone must be frozen")
    raw = read_csv(folder / "scores_before_labels.csv")
    scored = read_csv(folder / "test_predictions.csv")
    require(len(raw) == len(scored), "Before/after label join count mismatch")
    require("label" not in raw[0], "Label present before scoring finished")
    for before, after in zip(raw, scored):
        require(before == {k: v for k, v in after.items() if k != "label"}, "Prediction changed after label join")
    metrics_rows = read_csv(folder / "test_metrics.csv")
    measures = {(r["category"], r["method"]): r for r in metrics_rows}
    require(len(metrics_rows) == len(measures) == 8, "Expected eight test comparisons")
    n_train = n_test = right_test = 0
    max_diff = 0.
    diagnostic_rows = []
    boundary_rows = []
    branch_errors = []
    source_sizes = set()
    with ZipFile(args.archive) as archive:
        train, test = image_lists(archive)
        source_labels = official_labels(archive, [p for paths in test.values() for p in paths])
        seen = {}
        for category in CATEGORIES:
            for name, _ in train[category]:
                with Image.open(BytesIO(archive.read(ROOT + name))) as source:
                    source_sizes.add(source.size)
                    image = source.convert("RGB")
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                require(digest not in seen, "Duplicated training image")
                seen[digest] = name
            labels = np.array([y for _, y in train[category]])
            n_train += len(labels)
            config = frozen["categories"][category]["config"]
            verify_search(read_json(folder / f"{category}_selection.json"), config, frozen, np.arange(len(labels)), labels, 5)
        for category in CATEGORIES:
            model_file = folder / f"{category}.joblib"
            require(file_hash(model_file) == frozen["categories"][category]["model_sha256"], "Model file SHA256 mismatch")
            payload = joblib.load(model_file)
            require(payload["category"] == category and payload["weights_sha256"] == encoder.weights_sha256, "Wrong model")
            require(json.loads(json.dumps(payload["detector"]["config"])) == frozen["categories"][category]["config"],
                    "Frozen config and serialized model differ")
            names = test[category]
            for name in names:
                with archive.open(ROOT + name) as stream, Image.open(stream) as image:
                    source_sizes.add(image.size)
            rows = [r for r in scored if r["category"] == category]
            require([r["path"] for r in rows] == names, "Test paths do not match archive")
            labels = np.array([source_labels[n] for n in names])
            np.testing.assert_array_equal([int(r["label"]) for r in rows], labels)
            print(f"[verify] re-encoding {len(names)} {category} test images", flush=True)
            gx, patches = encode_members(archive, names, encoder, info["batch_size"], seen)
            require(gx.shape == (len(names), 7168) and patches.shape == (len(names), 2304, 256), "Feature shape mismatch")
            require(payload["detector"]["bank"].shape == (frozen["bank_size"], 256), "Memory bank dimensions mismatch")
            values, _ = score_detector(payload["detector"], gx, patches, np.arange(len(names)))
            config = payload["detector"]["config"]
            global_center, global_spread = config["global_scale"]
            local_center, local_spread = config["local_scale"]
            global_z = (np.array([float(r["global_score"]) for r in rows]) - global_center)/global_spread
            local_z = (np.array([float(r["local_score"]) for r in rows]) - local_center)/local_spread
            alpha = config["local_weight"]
            close([float(r["fusion_score"]) for r in rows], (1-alpha)*global_z + alpha*local_z, "Test fusion formula")
            baseline = measures[(category, "all_normal")]
            close(float(baseline["threshold"]), 1., "All-normal baseline cutoff")
            verify_metrics(baseline, labels, np.zeros(len(names)), 1.)
            for branch, scores in values.items():
                cut = payload["detector"]["config"]["thresholds"][branch]
                close([float(r[branch + "_threshold"]) for r in rows], cut, "Test threshold")
                close(float(measures[(category, branch)]["threshold"]), cut, "Threshold in metrics table")
                saved = np.array([float(r[branch + "_score"]) for r in rows])
                max_diff = max(max_diff, float(np.max(abs(saved-scores))))
                close(saved, scores, "Raw image score replay", 1e-5)
                np.testing.assert_array_equal([int(r[branch + "_prediction"]) for r in rows], scores >= cut)
                verify_metrics(measures[(category, branch)], labels, saved, cut)
                mistakes = np.flatnonzero((scores >= cut) != labels)
                branch_errors.append(f"| {category} | {branch} | " +
                                     (", ".join(f"`{names[i]}`" for i in mistakes) or "无") + " |")
                if branch == "fusion":
                    right = int(np.count_nonzero((scores >= cut) == labels))
                    diagnostic_rows.append(f"| {category} | {len(names)} | {right} | {measures[(category, branch)]['fp']} | {measures[(category, branch)]['fn']} |")
                    normals, anomalies = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
                    for idx in (normals[np.argmax(saved[normals])], anomalies[np.argmin(saved[anomalies])]):
                        boundary_rows.append(f"| `{names[idx]}` | {labels[idx]} | {saved[idx]:.6f} | {cut:.6f} | {saved[idx]-cut:+.6f} |")
                    right_test += right
            n_test += len(names)
        require(n_test == len(scored) and len(seen) == info["unique_decoded_images"] == n_train+n_test, "Dataset size mismatch")
    evidence = ["# 实验核验记录", "", f"- 原始 ZIP 校验值一致，解码核对 {n_train+n_test} 张图像。",
                f"- 原始尺寸集合为 {sorted(source_sizes)}，核对冻结编码器、7168 维全局特征、2304 × 256 局部特征和 {frozen['bank_size']} × 256 正常记忆库。",
                f"- 从 PNG 重新推理 {n_test} 张，三个分支共 {3*n_test} 个分数的最大绝对差为 {max_diff:.8g}。",
                f"- 对 {n_test} 张测试图从两支原始分数独立代入校准与融合公式，核对最终分数。",
                f"- 核对训练内折外分数、候选参数、阈值和融合权重，并用两两排序及分组累计重新计算 8 组测试指标。",
                f"- 课程测试集的融合预测共 {right_test}/{n_test} 张正确。"]
    searches = len(CATEGORIES)
    per_search = len(frozen["svm_grid"]) + len(frozen["fractions"]) + len(frozen["local_weights"])
    evidence.append(f"- {searches} 次训练内搜索的 {searches*per_search} 组候选均已按留存的折外分数复算。")
    if args.compare:
        difference = compare_runs(folder, args.compare)
        evidence.append(f"- 两次运行的 {n_test} 条课程测试记录一致，数值列最大绝对差为 {difference:.8g}。")
    evidence += ["", "| 类别 | 测试图数 | 正确数 | 误报 | 漏检 |", "|---|---:|---:|---:|---:|", *diagnostic_rows]
    evidence += ["", "## 各分支的误判样本", "", "| 类别 | 方法 | 误判路径 |", "|---|---|---|", *branch_errors]
    evidence += ["", "## 每类分数最高的正常图与最低的异常图", "",
                 "| 图像 | 真值 | 分数 | 阈值 | 分数减阈值 |", "|---|---:|---:|---:|---:|", *boundary_rows]
    (folder / "verification.md").write_text("\n".join(evidence) + "\n", encoding="utf-8")
    print("\n".join(evidence), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent.parent / "Image_Anomaly_Detection.zip")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--compare", type=Path, help="Also compare an earlier full run with this one")
    args = parser.parse_args()
    threads = read_json(args.run / "run_info.json")["cpu_threads"]
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=threads):
        audit(args)
