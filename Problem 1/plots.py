"""Render the frozen Task 1 experiment; labels are used for evaluation plots only.

Run from Problem 1:
    python plots.py --run results/run --archive ../DM_2026_Dataset/Cluster.zip \
        --output course-report/figures

This module does not fit clustering models, choose a seed, or change assignments.
The two embedding panels share a single label-free two-dimensional PCA projection.
Representative images are chosen by distance to predicted-cluster centroids before
their ground-truth labels are used as captions.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


CLASS_ZH = {
    "bottle": "瓶子", "cable": "电缆", "leather": "皮革",
    "pill": "药片", "tile": "瓷砖", "transistor": "晶体管",
}
COLORS = ("#3478AE", "#D97937", "#45956C", "#B45A78", "#8B78B5", "#7D7564")
METHOD_ORDER = {"handcrafted": 0, "deep": 1, "deep_pca": 2, "ward": 3}
METHOD_NAMES = {
    "handcrafted": "手工特征\nK-means",
    "deep": "深度特征\nK-means",
    "deep_pca": "深度特征+PCA\nK-means",
    "ward": "深度特征\nWard",
}


def configure_fonts() -> None:
    preferred = Path("C:/Windows/Fonts/simhei.ttf")
    if preferred.exists():
        font_manager.fontManager.addfont(str(preferred))
        font_family = font_manager.FontProperties(fname=preferred).get_name()
    else:
        available = {font.name for font in font_manager.fontManager.ttflist}
        font_family = next((name for name in ("SimHei", "Noto Sans CJK SC", "Microsoft YaHei")
                            if name in available), None)
        if font_family is None:
            raise RuntimeError("Install SimHei or another supported Chinese font before rendering")
    plt.rcParams.update({
        "font.family": font_family, "font.size": 12,
        "axes.titlesize": 15, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 11, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, folder: Path, name: str) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{extension}", dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def scalar_id(value) -> str:
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def method_key(name: str) -> str:
    lower = name.lower().replace("-", "_")
    if "ward" in lower:
        return "ward"
    if "hand" in lower or "color" in lower:
        return "handcrafted"
    if "pca" in lower:
        return "deep_pca"
    if "deep" in lower or "resnet" in lower:
        return "deep"
    return name


def load_inputs(run: Path, archive_path: Path) -> dict:
    with np.load(run / "features.npz", allow_pickle=False) as source:
        names = np.asarray(source["names"]).astype(str)
        features = np.asarray(source["deep"], dtype=np.float64)
    if features.ndim != 2 or len(features) != len(names) or not np.isfinite(features).all():
        raise ValueError("Expected aligned, finite deep features and filenames")
    if len(set(names.tolist())) != len(names):
        raise ValueError("Feature filenames must be unique")
    predictions = read_csv(run / "predictions.csv")
    if len(predictions) != len(names):
        raise ValueError("Prediction and feature counts differ")
    by_name = {row["filename"]: row for row in predictions}
    if set(by_name) != set(names) or len(by_name) != len(predictions):
        raise ValueError("Prediction filenames do not uniquely match the feature file")
    cluster = np.array([scalar_id(by_name[name]["cluster"]) for name in names])
    with np.load(run / "clusters.npz", allow_pickle=False) as source:
        main = np.array([scalar_id(value) for value in source["deep_s11"]])
    if not np.array_equal(cluster, main):
        raise ValueError("predictions.csv does not describe the fixed deep_s11 assignment")
    with ZipFile(archive_path) as archive:
        official = json.loads(archive.read("Cluster/cluster_labels.json").decode("utf-8-sig"))
    truth = np.array([official[Path(name).name] for name in names])
    if any(by_name[name].get("true_label", official[Path(name).name]) != label
           for name, label in zip(names, truth)):
        raise ValueError("Saved truth labels disagree with the official archive")
    stored = json.loads((run / "metrics.json").read_text(encoding="utf-8-sig"))
    external = stored.get("external", stored)
    classes = list(external["class_names"])
    ids = [scalar_id(value) for value in external["cluster_ids"]]
    table = np.asarray(external["contingency_matrix"], dtype=int)
    actual = np.array([[np.sum((cluster == c) & (truth == t)) for t in classes] for c in ids])
    if not np.array_equal(actual, table):
        raise ValueError("Saved contingency matrix disagrees with the saved assignments")
    if "cluster_to_class" in external:
        mapping = {scalar_id(key): value for key, value in external["cluster_to_class"].items()}
    else:
        from scipy.optimize import linear_sum_assignment
        rows, columns = linear_sum_assignment(table, maximize=True)
        mapping = {ids[r]: classes[c] for r, c in zip(rows, columns)}
    return dict(names=names, features=features, cluster=cluster, truth=truth,
                classes=classes, cluster_ids=ids, mapping=mapping, external=external,
                measurements=read_csv(run / "all_metrics.csv"))


def plot_method(folder: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 4.7))
    ax.set(xlim=(0, 12.4), ylim=(0, 4.7))
    ax.axis("off")
    x_positions = [0.10, 3.25, 6.40, 9.55]
    width, height, top = 2.70, 1.12, 3.18
    texts = ["600 张无标签图像\n统一图像预处理", "冻结 ResNet18\n提取 512 维特征",
             "逐图 L2 归一化\n统一向量尺度", "K-means 聚类\nK=6，预设种子"]
    for x, label in zip(x_positions, texts):
        ax.add_patch(Rectangle((x, top), width, height, facecolor="#F3F6F8",
                               edgecolor="#536672", linewidth=1.2))
        ax.text(x + width/2, top + height/2, label, ha="center", va="center", fontsize=14, linespacing=1.65)
    for a, b in zip(x_positions, x_positions[1:]):
        ax.add_patch(FancyArrowPatch((a + width, top + height/2), (b, top + height/2),
                                    arrowstyle="-|>", mutation_scale=15, color="#536672", linewidth=1.2))
    bottom = 1.35
    for x, label in ((9.55, "保存特征与簇划分\n固定主运行：种子 11"),
                     (6.40, "外部评价\nACC / ARI / NMI"),
                     (3.25, "读取真实类别\n仅进入评价与绘图")):
        ax.add_patch(Rectangle((x, bottom), width, height, facecolor="white",
                               edgecolor="#536672", linewidth=1.2))
        ax.text(x + width/2, bottom + height/2, label, ha="center", va="center", fontsize=14, linespacing=1.65)
    ax.add_patch(FancyArrowPatch((10.90, top), (10.90, bottom + height),
                                arrowstyle="-|>", mutation_scale=15, color="#536672", linewidth=1.2))
    ax.add_patch(FancyArrowPatch((9.55, bottom + height/2), (9.10, bottom + height/2),
                                arrowstyle="-|>", mutation_scale=15, color="#536672", linewidth=1.2))
    ax.add_patch(FancyArrowPatch((5.95, bottom + height/2), (6.40, bottom + height/2),
                                arrowstyle="-|>", mutation_scale=15, color="#536672", linewidth=1.2))
    ax.text(0.10, 0.65, "对照实验", fontsize=14, weight="normal", va="center")
    ax.text(2.0, 0.65, "手工特征 K-means  /  深度特征 + PCA K-means  /  Ward 层次聚类",
            fontsize=13, va="center")
    save_figure(fig, folder, "method-overview")


def plot_embedding(data: dict, folder: Path) -> dict:
    projection = PCA(n_components=2, svd_solver="full").fit(data["features"])
    coordinates = projection.transform(data["features"])
    color = {name: COLORS[i % len(COLORS)] for i, name in enumerate(data["classes"])}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.3), sharex=True, sharey=True)
    for label in data["classes"]:
        mask = data["truth"] == label
        axes[0].scatter(*coordinates[mask].T, s=19, alpha=.78, color=color[label], linewidths=0)
    for cluster_id in data["cluster_ids"]:
        mask = data["cluster"] == cluster_id
        label = data["mapping"].get(cluster_id)
        axes[1].scatter(*coordinates[mask].T, s=19, alpha=.78,
                        color=color.get(label, "#555555"), linewidths=0)
    axes[0].set_title("真实类别（仅用于评价）", pad=12)
    axes[1].set_title("预测簇（按评价匹配着色）", pad=12)
    ratios = projection.explained_variance_ratio_ * 100
    for ax in axes:
        ax.set_xlabel(f"第一主成分（{ratios[0]:.1f}% 方差）")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.15, linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"第二主成分（{ratios[1]:.1f}% 方差）")
    handles = [Line2D([], [], color=color[name], marker="o", linestyle="", markersize=7,
                      label=CLASS_ZH.get(name, name)) for name in data["classes"]]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(.5, .005), columnspacing=1.8)
    fig.subplots_adjust(bottom=.18, wspace=.16, left=.075, right=.99, top=.88)
    save_figure(fig, folder, "cluster-embedding")
    return {"pca_variance_ratio": projection.explained_variance_ratio_.tolist(),
            "color_alignment": "Hungarian matching for presentation only; identical PCA coordinates in both panels"}


def plot_confusion(data: dict, folder: Path) -> None:
    counts = np.asarray(data["external"]["confusion_matrix"], dtype=int)
    rows = counts.sum(axis=1, keepdims=True)
    fraction = np.divide(counts, rows, out=np.zeros_like(counts, dtype=float), where=rows != 0)
    columns = data["external"].get("confusion_columns", data["classes"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    chart = ax.imshow(fraction, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, f"{counts[i,j]}\n{fraction[i,j]:.0%}", ha="center", va="center",
                    fontsize=13, linespacing=1.4, color="white" if fraction[i,j] > .56 else "#263442")
    ax.set_xticks(range(len(columns)), [CLASS_ZH.get(name, name or "未匹配") for name in columns])
    ax.set_yticks(range(len(data["classes"])), [CLASS_ZH.get(name, name) for name in data["classes"]])
    ax.set_xlabel("预测类别（簇标签经一一匹配）", labelpad=10)
    ax.set_ylabel("真实类别", labelpad=10)
    ax.set_title("主方案混淆矩阵：计数 / 类内比例", pad=15)
    ax.set_xticks(np.arange(-.5, counts.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, counts.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(chart, ax=ax, shrink=.82, pad=.035)
    colorbar.set_label("类内比例", labelpad=10)
    fig.tight_layout()
    save_figure(fig, folder, "confusion-matrix")


def plot_examples(data: dict, archive_path: Path, folder: Path) -> list[dict]:
    chosen = []
    for cluster_id in data["cluster_ids"]:
        positions = np.flatnonzero(data["cluster"] == cluster_id)
        if len(positions) < 2:
            raise ValueError("Each predicted cluster must contain at least two examples")
        center = data["features"][positions].mean(axis=0)
        squared = np.square(data["features"][positions] - center).sum(axis=1)
        # Tied distances are resolved using the existing stable filename order.
        order = np.lexsort((data["names"][positions], squared))[:2]
        for row, local_index in enumerate(order):
            index = int(positions[local_index])
            chosen.append(dict(cluster=cluster_id, row=row, index=index,
                               filename=data["names"][index], true_label=data["truth"][index],
                               squared_distance=float(squared[local_index]), cluster_size=len(positions)))
    fig, axes = plt.subplots(2, len(data["cluster_ids"]), figsize=(12.6, 5.6), squeeze=False)
    with ZipFile(archive_path) as archive:
        for sample in chosen:
            column = data["cluster_ids"].index(sample["cluster"])
            ax = axes[sample["row"], column]
            member = "Cluster/dataset/" + Path(sample["filename"]).name
            with Image.open(io.BytesIO(archive.read(member))) as source:
                # Resize for plotting only; clustering representations stay untouched.
                source = source.convert("RGB")
                source.thumbnail((512, 512), Image.Resampling.LANCZOS)
                ax.imshow(source)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if sample["row"] == 0:
                ax.set_title(f"簇 {sample['cluster']}（n={sample['cluster_size']}）", fontsize=12, pad=9)
            ax.set_xlabel(f"{sample['filename']} · {CLASS_ZH.get(sample['true_label'], sample['true_label'])}",
                          fontsize=11, labelpad=6)
    fig.suptitle("各预测簇中最靠近特征中心的两张图像", fontsize=16, y=.98)
    fig.text(.5, .015, "样例按主方案的深度特征距离选取；图下注明文件名与真实类别。", ha="center", fontsize=11)
    fig.subplots_adjust(left=.015, right=.985, top=.86, bottom=.11, wspace=.10, hspace=.25)
    save_figure(fig, folder, "cluster-examples")
    return chosen


def summarize_measurements(measurements: list[dict]) -> list[dict]:
    grouped = {}
    for row in measurements:
        grouped.setdefault(row["method"], []).append(row)
    summaries = []
    for method, records in grouped.items():
        seeds = [int(row["seed"]) for row in records]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Repeated method/seed pairs in all_metrics.csv: {method}")
        values = {}
        for metric in ("acc", "ari", "nmi"):
            numbers = np.array([float(row[metric]) for row in records])
            if not np.isfinite(numbers).all():
                raise ValueError(f"Nonfinite metric in {method}/{metric}")
            values[metric] = {"mean": float(numbers.mean()),
                              "std": float(numbers.std(ddof=1)) if len(numbers) > 1 else None}
        summaries.append(dict(method=method, key=method_key(method), n=len(records), seeds=seeds, metrics=values))
    return sorted(summaries, key=lambda row: (METHOD_ORDER.get(row["key"], 99), row["method"]))


def plot_comparison(summaries: list[dict], folder: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.9), sharey=True)
    x = np.arange(len(summaries))
    labels = [METHOD_NAMES.get(row["key"], row["method"]) for row in summaries]
    colors = [COLORS[METHOD_ORDER.get(row["key"], 0) % len(COLORS)] for row in summaries]
    min_score = min(row["metrics"][metric]["mean"] - (row["metrics"][metric]["std"] or 0)
                    for row in summaries for metric in ("acc", "ari", "nmi"))
    for ax, metric in zip(axes, ("acc", "ari", "nmi")):
        means = np.array([row["metrics"][metric]["mean"] for row in summaries])
        stds = np.array([row["metrics"][metric]["std"] or 0 for row in summaries])
        ax.bar(x, means, color=colors, width=.65, yerr=stds, capsize=4,
               error_kw={"elinewidth": 1.2, "capthick": 1.2})
        for index, (mean, std) in enumerate(zip(means, stds)):
            ax.text(index, mean + std + .025, f"{mean:.3f}", ha="center", fontsize=11)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelsize=10.8, length=0, pad=10)
        ax.set_title(metric.upper(), pad=12)
        ax.set_ylim(min(0, min_score - .08), 1.10)
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.grid(axis="y", alpha=.18, linewidth=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("外部评价指标（越高越好）")
    n_values = {row["n"] for row in summaries if row["key"] != "ward"}
    repeat_text = str(next(iter(n_values))) if len(n_values) == 1 else "实际"
    fig.text(.5, .018, f"K-means：{repeat_text} 个预设种子的均值 ± 样本标准差；Ward：确定性单次运行。",
             ha="center", fontsize=11)
    fig.subplots_adjust(bottom=.23, top=.88, left=.065, right=.99, wspace=.16)
    save_figure(fig, folder, "metrics-comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/run"))
    parser.add_argument("--archive", type=Path, default=Path("../DM_2026_Dataset/Cluster.zip"))
    parser.add_argument("--output", type=Path, default=Path("course-report/figures"))
    args = parser.parse_args()
    configure_fonts()
    data = load_inputs(args.run, args.archive)
    summaries = summarize_measurements(data["measurements"])
    args.output.mkdir(parents=True, exist_ok=True)
    plot_method(args.output)
    embedding = plot_embedding(data, args.output)
    plot_confusion(data, args.output)
    examples = plot_examples(data, args.archive, args.output)
    plot_comparison(summaries, args.output)
    metadata = dict(main_assignment="deep_s11", embedding=embedding, examples=examples,
                    comparison=summaries, source_run=str(args.run.resolve()))
    (args.output / "plot-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved five PDF/PNG figure pairs and plot metadata to {args.output.resolve()}")


if __name__ == "__main__":
    main()
