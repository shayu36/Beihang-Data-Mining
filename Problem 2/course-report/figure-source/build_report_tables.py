"""Build LaTeX tables and numerical macros directly from a completed experiment.

Run from the report root:
  python figure-source/build_report_tables.py --run results/run1
The program checks prediction/threshold consistency and recomputes count metrics.
It never fits a model or modifies the saved experiment.
"""
from pathlib import Path
import argparse
import csv
import json
import math

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ("hazelnut", "zipper")
BRANCHES = ("all_normal", "global", "local", "fusion")
CN = {"hazelnut": "榛子", "zipper": "拉链"}
METHOD = {"all_normal": "恒判正常", "global": "全局SVM", "local": "局部最近邻", "fusion": "融合"}


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_metrics(rows, records):
    for category in CATEGORIES:
        group = [r for r in rows if r["category"] == category]
        for method in BRANCHES:
            record = records[(category, method)]
            counts = dict(tn=0, fp=0, fn=0, tp=0)
            for row in group:
                truth = int(row["label"])
                if method == "all_normal":
                    predicted = 0
                else:
                    score = float(row[method + "_score"])
                    threshold = float(row[method + "_threshold"])
                    assert math.isfinite(score) and math.isfinite(threshold)
                    predicted = int(score >= threshold)
                    assert predicted == int(row[method + "_prediction"])
                    assert threshold == float(record["threshold"])
                counts[("tn", "fp", "fn", "tp")[2 * truth + predicted]] += 1
            tn, fp, fn, tp = (counts[k] for k in ("tn", "fp", "fn", "tp"))
            vals = dict(**counts, n=len(group), normal=tn + fp, abnormal=tp + fn,
                        accuracy=(tp + tn) / len(group), precision=tp / max(tp + fp, 1),
                        recall=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn),
                        balanced_accuracy=.5 * (tp / (tp + fn) + tn / (tn + fp)))
            for key, value in vals.items():
                assert math.isclose(value, float(record[key]), rel_tol=0, abs_tol=1e-12), (category, method, key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results" / "run1")
    args = parser.parse_args()
    rows = csv_rows(args.run / "test_predictions.csv")
    metrics = csv_rows(args.run / "test_metrics.csv")
    records = {(r["category"], r["method"]): r for r in metrics}
    assert len(records) == 8 and len({r["path"] for r in rows}) == len(rows)
    check_metrics(rows, records)
    frozen = json.loads((args.run / "frozen.json").read_text(encoding="utf-8"))
    info = json.loads((args.run / "run_info.json").read_text(encoding="utf-8"))
    lines = ["% Generated from test_metrics.csv; do not edit the numerical rows manually.",
             r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"产品&方法&正确数&Accuracy (\%)&Precision&Recall&F1&AUROC&AP\\", r"\midrule"]
    for category in CATEGORIES:
        if category != CATEGORIES[0]:
            lines.append(r"\midrule")
        for method in BRANCHES:
            r = records[(category, method)]
            vals = [f"{100 * float(r['accuracy']):.2f}"] + [f"{float(r[k]):.4f}" for k in ("precision", "recall", "f1", "auroc", "ap")]
            if method == "fusion":
                vals = [r"\textbf{" + v + "}" for v in vals]
            first = CN[category] if method == "all_normal" else ""
            correct = int(r["tn"]) + int(r["tp"])
            lines.append("&".join([first, METHOD[method], f"{correct}/{r['n']}"] + vals) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "sec" / "results-table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    boundaries = []
    for category in CATEGORIES:
        group = [r for r in rows if r["category"] == category]
        boundaries.append(max((r for r in group if int(r["label"]) == 0), key=lambda r: float(r["fusion_score"])))
        boundaries.append(min((r for r in group if int(r["label"]) == 1), key=lambda r: float(r["fusion_score"])))
    lines = ["% Generated from full-precision scores; round only for display.",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"产品及测试编号&真实类别&融合分数$S_c$&阈值$\tau_c$&余量$\Delta$\\", r"\midrule"]
    for row in boundaries:
        score, threshold = float(row["fusion_score"]), float(row["fusion_threshold"])
        lines.append("&".join([CN[row["category"]] + Path(row["path"]).stem,
                                "异常" if int(row["label"]) else "正常", f"{score:.6f}",
                                f"{threshold:.6f}", f"${score - threshold:+.6f}$"]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "sec" / "boundary-table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    macros = {"RunPython": info["python"], "RunTorch": info["packages"]["torch"],
              "RunVision": info["packages"]["torchvision"], "RunSklearn": info["packages"]["scikit-learn"],
              "RunMinutes": f"{info['seconds'] / 60:.2f}", "RunThreads": str(info["cpu_threads"]),
              "RunBatch": str(info["batch_size"]), "TotalTests": str(len(rows))}
    for category, prefix in (("hazelnut", "Haz"), ("zipper", "Zip")):
        config = frozen["categories"][category]["config"]
        macros[prefix + "Threshold"] = f"{config['thresholds']['fusion']:.6f}"
        macros[prefix + "Alpha"] = f"{config['local_weight']:g}"
        macros[prefix + "C"] = f"{config['svm'][0]:g}"
        macros[prefix + "GammaScale"] = f"{config['svm'][1]:g}"
        macros[prefix + "Fraction"] = f"{config['fraction']:g}"
        macros[prefix + "TopK"] = str(math.ceil(2304 * config["fraction"]))
        for row in (r for r in boundaries if r["category"] == category):
            side = "Abnormal" if int(row["label"]) else "Normal"
            score, threshold = float(row["fusion_score"]), float(row["fusion_threshold"])
            macros[prefix + side + "ID"] = Path(row["path"]).stem
            macros[prefix + side + "Score"] = f"{score:.6f}"
            macros[prefix + side + "MarginAbs"] = f"{abs(score - threshold):.6f}"
            for branch, label in (("global", "Global"), ("local", "Local")):
                macros[prefix + side + label + "Score"] = f"{float(row[branch + '_score']):.4f}"
                macros[prefix + side + label + "Threshold"] = f"{float(row[branch + '_threshold']):.4f}"
        for method, suffix in (("global", "Global"), ("local", "Local"), ("fusion", "Fusion")):
            record = records[(category, method)]
            macros[prefix + suffix + "AUC"] = f"{float(record['auroc']):.4f}"
            macros[prefix + suffix + "AP"] = f"{float(record['ap']):.4f}"
    def esc(value):
        return value.replace("_", r"\_").replace("%", r"\%")
    (ROOT / "sec" / "verified-values.tex").write_text(
        "% Generated from the current verified run.\n" +
        "\n".join(r"\newcommand{" + chr(92) + k + "}{" + esc(v) + "}" for k, v in macros.items()) + "\n",
        encoding="utf-8")
    print(f"Checked {len(rows)} predictions and 8 metric rows; generated LaTeX data files.")


if __name__ == "__main__":
    main()
