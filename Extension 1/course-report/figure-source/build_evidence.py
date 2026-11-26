"""Build the benchmark table and thyroid case figure from sample-level predictions."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


DOMAINS = {"hazelnut": "榛子", "zipper": "拉链", "thyroid": "甲状腺"}
METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "average_precision")
COUNTS = ("tn", "fp", "fn", "tp")
COLORS = {"baseline": "#727a85", "original": "#17668c", "corrected": "#be5724"}
SCHEME_LABELS = {"baseline": "基线", "original": "两列乘积", "corrected": "三列修正"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fields(rows):
    return (np.asarray([int(row["label"]) for row in rows]),
            np.asarray([int(row["prediction"]) for row in rows]),
            np.asarray([float(row["score"]) for row in rows]))


def recalculate(rows):
    y, p, scores = fields(rows)
    counts = confusion_matrix(y, p, labels=[0, 1]).ravel()
    values = (accuracy_score(y, p), precision_score(y, p, zero_division=0),
              recall_score(y, p, zero_division=0), f1_score(y, p, zero_division=0),
              roc_auc_score(y, scores), average_precision_score(y, scores))
    return {**dict(zip(COUNTS, map(int, counts), strict=True)),
            **dict(zip(METRICS, map(float, values), strict=True))}


def read_verified(results):
    entries = defaultdict(list)
    annotations = {}
    for r in read_csv(results / "run2" / "test_predictions.csv"):
        entries[r["domain"], r["method"]].append(r)
    for r in read_csv(results / "run2" / "metrics.csv"):
        if r["scope"] == "official":
            annotations[r["domain"], r["method"]] = r
    for r in read_csv(results / "shared" / "predictions.csv"):
        entries[r["domain"], "shared"].append(r)
    for r in read_csv(results / "shared" / "metrics.csv"):
        annotations[r["domain"], "shared"] = r
    for r in read_csv(results / "correction" / "predictions.csv"):
        entries["thyroid", r["method"]].append(r)
    comparison = json.loads((results / "correction" / "comparison.json").read_text(encoding="utf-8"))
    for r in comparison["results"]:
        if r["scope"] == "official":
            annotations["thyroid", r["method"]] = r

    if set(entries) != set(annotations):
        raise ValueError("Metric rows and prediction groups differ")
    expected_labels = {}
    metrics = {}
    for (domain, method), rows in entries.items():
        key = domain, method
        given = annotations[key]
        labels = {r["sample"]: int(r["label"]) for r in rows}
        if len(labels) != len(rows) or len(rows) != int(given["n"]):
            raise ValueError(f"{key}: duplicated or missing sample")
        if labels != expected_labels.setdefault(domain, labels):
            raise ValueError(f"{key}: sample IDs or labels differ across methods")
        computed = recalculate(rows)
        for name, value in computed.items():
            if not np.isclose(value, float(given[name]), rtol=0, atol=1e-12):
                raise ValueError(f"{key}: reported and reconstructed {name} differ")
        metrics[key] = computed
    reference = {r["sample"]: r for r in entries["thyroid", "improved"]}
    for r in entries["thyroid", "original"]:
        other = reference[r["sample"]]
        if any(float(r[name]) != float(other[name]) for name in ("score", "threshold", "prediction")):
            raise ValueError("Two-column reference differs across the original and correction runs")
    return entries, metrics


def build_table(root, metrics):
    common = [("baseline", "PCA 基线"), ("shared", "同参近邻")]
    image = [("supervised_only", "仅 SVM"), ("improved", "监督融合")]
    thyroid = [("normal_reference", "白化近邻"), ("original", "两列乘积"), ("corrected", "三列修正")]
    table = [r"\begin{tabular}{@{}llcrrrrrr@{}}", r"\toprule",
             r"数据 & 方法 & TN/FP/FN/TP & Acc & P & R & F1 & AUROC & AP \\", r"\midrule"]
    for domain, label in DOMAINS.items():
        methods = common + (thyroid if domain == "thyroid" else image)
        if domain != "hazelnut":
            table.append(r"\cmidrule(lr){1-9}")
        peak = max(metrics[domain, method]["f1"] for method, _ in methods)
        for i, (method, title) in enumerate(methods):
            summary = metrics[domain, method]
            counts = "/".join(str(summary[k]) for k in COUNTS)
            values = [f"{summary[k]:.4f}" for k in METRICS]
            if summary["f1"] == peak:
                values[3] = r"\textbf{" + values[3] + "}"
            table.append(" & ".join([label if i == 0 else "", title, counts, *values]) + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (root / "sec" / "benchmark.tex").write_text("\n".join(table) + "\n", encoding="utf-8")


def build_figure(root, entries, metrics):
    font_path = root / "fonts" / "SimHei.ttf"
    font_manager.fontManager.addfont(str(font_path))
    chinese = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", chinese], "font.size": 7.7, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#667380", "axes.labelcolor": "#253644", "axes.linewidth": 0.65,
        "ytick.color": "#354855", "xtick.color": "#354855", "text.color": "#203240",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    })
    fig, (pr, cases) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.03, 1]})
    fig.subplots_adjust(left=0.087, right=0.985, bottom=0.21, top=0.85, wspace=0.43)
    pr.set_title("A   甲状腺测试集的精确率与召回率", loc="left", fontsize=9.4, pad=8)
    cases.set_title("B   两条样本的分数与判定", loc="left", fontsize=9.4, pad=8)
    for method, style, zorder in (("baseline", "--", 2), ("original", "-", 3), ("corrected", (0, (3, 2)), 4)):
        y, _, scores = fields(entries["thyroid", method])
        precision, recall, _ = precision_recall_curve(y, scores, drop_intermediate=True)
        ap = metrics["thyroid", method]["average_precision"]
        # sklearn supplies recall in descending order; post-steps give the AP integral.
        pr.step(recall, precision, where="post", color=COLORS[method], linestyle=style,
                linewidth=1.65 if method == "original" else 1.15, zorder=zorder,
                label=f"{SCHEME_LABELS[method]}   AP {ap:.4f}")
    prevalence = float(np.mean(y))
    pr.axhline(prevalence, color="#8d939b", linestyle=":", linewidth=0.9)
    pr.text(0.02, prevalence + 0.033, f"正例率 {prevalence:.2%}", color="#5f6872", fontsize=7)
    pr.set(xlim=(0, 1.01), ylim=(0, 1.03), xlabel="召回率", ylabel="精确率")
    for axis in (pr.xaxis, pr.yaxis):
        axis.set_ticks(np.linspace(0, 1, 5))
        axis.set_major_formatter(PercentFormatter(1, decimals=0))
    pr.grid(axis="y", color="#e2e7ea", linewidth=0.55)
    pr.legend(loc="lower left", bbox_to_anchor=(0.01, 0.155), frameon=False,
              handlelength=2.5, handletextpad=0.5, fontsize=7.4, labelspacing=0.35)

    by_sample = {method: {r["sample"]: r for r in entries["thyroid", method]}
                 for method in ("original", "corrected")}
    inspected = (("test_row_1499", 1.08), ("test_row_1840", 0.25))
    thresholds = {float(by_sample[method][sample]["threshold"]) for method in by_sample for sample, _ in inspected}
    if len(thresholds) != 1:
        raise ValueError("Case plot requires the same two thresholds")
    cut = thresholds.pop()
    cases.set(xlim=(-0.045, 1.48), ylim=(-0.15, 1.50), xlabel="异常分数")
    cases.axvspan(cut, 1.48, facecolor="#f9f4ee", edgecolor="none", zorder=0)
    cases.axvline(cut, color="#6f6256", linestyle=(0, (3, 2)), linewidth=0.85)
    cases.text(cut + 0.022, 1.43, f"阈值 {cut:.3f}", color="#6f6256", fontsize=7.1, va="top")
    cases.set_xticks([0, 0.4, 0.8, 1.2], ["0", "0.4", "0.8", "1.2"])
    cases.tick_params(axis="y", length=0, pad=4)
    cases.spines["left"].set_visible(False)
    cases.axhline(0.66, color="#dee4e8", linewidth=0.65)
    ticklabels = []
    for sample, center in inspected:
        label = int(by_sample["original"][sample]["label"])
        ticklabels.append(f"{sample.split('_')[-1]}\n真实{'异常' if label else '正常'}")
        for method, offset, marker in (("original", 0.115, "o"), ("corrected", -0.115, "D")):
            row = by_sample[method][sample]
            score, pred = float(row["score"]), int(row["prediction"])
            if pred != int(score > cut) or int(row["label"]) != label:
                raise ValueError(f"{sample}: recorded decision differs from rule or labels")
            diagnosis = "正确" if pred == label else ("误报" if pred else "漏判")
            yloc = center + offset
            color = COLORS[method]
            cases.plot([0, score], [yloc, yloc], color=color, alpha=0.45, linewidth=1.2, zorder=2)
            cases.plot(score, yloc, marker=marker, color=color, markersize=4.2, zorder=3)
            left = score < cut and cut - score < 0.17
            cases.annotate(f"{score:.3f} {diagnosis}", xy=(score, yloc), xytext=(-5 if left else 5, 0),
                           textcoords="offset points", va="center", ha="right" if left else "left",
                           color=color, fontsize=7.2,
                           bbox=dict(facecolor="white", edgecolor="none", pad=0.8) if left else None)
    cases.set_yticks([center for _, center in inspected], ticklabels)
    for label in cases.get_yticklabels():
        label.set_multialignment("right")
    destination = root / "figures" / "thyroid-boundary"
    fig.savefig(destination.with_suffix(".pdf"))
    fig.savefig(destination.with_suffix(".png"), dpi=250)
    plt.close(fig)


def build_extended_table(root, entries, metrics):
    """Cross-check each frozen alternative against the same 1933 labelled rows."""
    label_map = {r["sample"]: int(r["label"]) for r in entries["thyroid", "corrected"]}
    models = [
        ("conditional", "六维条件残差融合"),
        ("generative", "模拟正例提升树"),
        ("crossfit", "五折校准集成"),
        ("tails", "连续尾部分布"),
        ("subgroups", "背景子群校准"),
    ]
    rows = [("三列修正", metrics["thyroid", "corrected"])]
    for folder, label in models:
        source = root / "results" / "extended" / folder
        record = json.loads((source / "course_evaluation.json").read_text(encoding="utf-8"))
        scored = read_csv(source / "scores_before_labels.csv")
        labelled = []
        sample_ids = set()
        for row in scored:
            index = row.get("row_1based") or row.get("sample_1based")
            sample = row["sample"] if index is None else f"test_row_{int(index):04d}"
            if sample in sample_ids or sample not in label_map:
                raise ValueError(f"{folder}: unexpected sample {sample}")
            sample_ids.add(sample)
            labelled.append(dict(row, label=label_map[sample]))
        if sample_ids != set(label_map):
            raise ValueError(f"{folder}: incomplete course evaluation")
        calc = recalculate(labelled)
        for key in (*COUNTS, *METRICS):
            reference = record.get("ap") if key == "average_precision" and key not in record else record[key]
            if not np.isclose(calc[key], reference, rtol=0, atol=1e-12):
                raise ValueError(f"{folder}: metric mismatch for {key}")
        rows.append((label, calc))
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}\toprule",
             r"方法 & FP/FN & P & R & F1 & AP \\\midrule"]
    for label, row in rows:
        rates = " & ".join(f"{row[name]:.4f}" for name in ("precision","recall","f1","average_precision"))
        lines.append(f"{label} & {row['fp']}/{row['fn']} & {rates} " + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    (root / "sec" / "extended.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Report source root containing results, sec, figures and local fonts")
    args = parser.parse_args()
    entries, metrics = read_verified(args.root / "results")
    build_table(args.root, metrics)
    build_extended_table(args.root, entries, metrics)
    build_figure(args.root, entries, metrics)
    print(f"Verified {len(metrics)} official metric rows, built benchmark.tex and thyroid-boundary.pdf/png.")


if __name__ == "__main__":
    main()
