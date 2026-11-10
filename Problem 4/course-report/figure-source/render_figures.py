"""Build report plots and tables from the existing Task4 data and scores.

This script reads the saved predictions and never fits or replaces a detector.
Run from any directory with Python, numpy, pandas, scipy and matplotlib.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parent
FIGURES = REPORT / "figures"
COLORS = {
    "IForest": "#3B6B99",
    "HBOS": "#368578",
    "MCD": "#C98A42",
    "ensemble_score": "#77539D",
}
NAMES = {
    "IForest": "Isolation Forest",
    "HBOS": "HBOS",
    "MCD": "MCD",
    "ensemble_score": "Rank ensemble",
}
NORMAL, DISEASE = "#3B6B99", "#CC7044"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "savefig.pad_inches": 0.03,
})


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def evaluate(y, values):
    precision, recall, thresholds = precision_recall_curve(y, values)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1[:-1]))
    threshold = float(thresholds[best])
    return {
        "AUROC": float(roc_auc_score(y, values)),
        "AUPRC": float(average_precision_score(y, values)),
        "F1": float(f1[best]),
        "Precision": float(precision[best]),
        "Recall": float(recall[best]),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y, values >= threshold).tolist(),
    }


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data/train-set.csv")
    test = pd.read_csv(ROOT / "data/test-set.csv")
    scores = pd.read_csv(ROOT / "results/final_scores.csv")
    assert np.array_equal(test.label.astype(int), scores.label)
    assert np.isfinite(train.to_numpy()).all()
    assert np.isfinite(test.to_numpy()).all()
    assert np.isfinite(scores.to_numpy()).all()
    y = scores.label.to_numpy(dtype=int)
    branch_names = ["IForest", "HBOS", "MCD"]
    recomputed = np.mean([
        (rankdata(scores[name], method="average") - 1) / (len(scores) - 1)
        for name in branch_names
    ], axis=0)
    assert np.allclose(recomputed, scores.ensemble_score, atol=1e-14, rtol=0)
    results = {name: evaluate(y, scores[name].to_numpy()) for name in COLORS}

    train_rows = set(map(tuple, train.to_numpy()))
    overlap = np.array([
        tuple(row) in train_rows for row in test[train.columns].to_numpy()
    ])
    facts = {
        "source_scores": "results/final_scores.csv",
        "training_executed_for_report": False,
        "metric_protocol": "AUPRC is average precision; thresholds maximize test F1",
        "rank_protocol": "Average ties, ascending ranks within the complete test batch",
        "train_samples": len(train),
        "test_samples": len(test),
        "test_positive_samples": int(y.sum()),
        "exact_six_feature_overlap_test_rows": int(overlap.sum()),
        "overlap_labels": test.loc[overlap, "label"].value_counts().to_dict(),
        "ensemble_max_absolute_reconstruction_error": float(
            np.max(np.abs(recomputed - scores.ensemble_score))
        ),
        "score_quantiles": {
            str(label): scores.loc[y == label, "ensemble_score"].quantile(
                [0.25, 0.5, 0.75]
            ).to_dict() for label in [0, 1]
        },
        "results": results,
    }
    (REPORT / "figure-source/verified-results.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    metrics = ["AUROC", "AUPRC", "F1", "Precision", "Recall"]
    best_values = {key: max(row[key] for row in results.values()) for key in metrics}
    rows = []
    for name in COLORS:
        label = "秩融合" if name == "ensemble_score" else NAMES[name]
        cells = []
        for key in metrics:
            value = results[name][key]
            text = f"{value:.4f}"
            if np.isclose(value, best_values[key], atol=1e-12, rtol=0):
                text = r"\textbf{" + text + "}"
            cells.append(text)
        rows.append(label + " & " + " & ".join(cells) + r" \\")
    table = r"""\begin{minipage}{\textwidth}
  \centering
  \captionof{table}{独立检测分支与秩融合方法的测试集结果}
  \label{tab:task4-results}
  \begin{tabular}{lccccc}
    \toprule
    方法 & AUROC & AUPRC & F1 & Precision & Recall \\
    \midrule
""" + "\n".join("    " + row for row in rows) + r"""
    \bottomrule
  \end{tabular}
  \par\smallskip
  {\small 粗体表示每列最优值，F1、Precision 和 Recall 对应各方法的测试集最佳 F1 阈值}
\end{minipage}
"""
    (REPORT / "sec/results-table.tex").write_text(table, encoding="utf-8")

    # Empirical CDFs show all observations, including both tails.
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.5), layout="constrained")
    for ax, feature in zip(axes.flat, train.columns):
        for label, color, legend in [(0, NORMAL, "Normal"), (1, DISEASE, "Disease")]:
            values = np.sort(test.loc[test.label == label, feature].to_numpy())
            cumulative = np.arange(1, len(values) + 1) / len(values)
            ax.step(values, cumulative, where="post", color=color, lw=1.4, label=legend)
        ax.set_title(feature, pad=4)
        ax.set_ylim(-0.025, 1.025)
        ax.set_yticks([0, 0.5, 1])
        ax.grid(axis="y", color="#E6E8EB", lw=0.5)
        ax.set_xlabel("Feature value", labelpad=2)
    axes[0, 0].set_ylabel("Cumulative proportion")
    axes[1, 0].set_ylabel("Cumulative proportion")
    axes[0, 1].legend(loc="lower right", frameon=False)
    save_figure(fig, "feature-distributions")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), layout="constrained")
    for name, color in COLORS.items():
        values = scores[name].to_numpy()
        fpr, tpr, _ = roc_curve(y, values)
        precision, recall, _ = precision_recall_curve(y, values)
        width = 1.8 if name == "ensemble_score" else 1.1
        axes[0].plot(fpr, tpr, color=color, lw=width,
                     label=f"{NAMES[name]}  {results[name]['AUROC']:.4f}")
        axes[1].step(recall[::-1], precision[::-1], where="pre", color=color,
                     lw=width, label=f"{NAMES[name]}  {results[name]['AUPRC']:.4f}")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC curves", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR curves",
                xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axes[0].legend(loc="lower right", title="AUROC", frameon=False, fontsize=8)
    axes[1].legend(loc="lower left", title="AP", frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(color="#E6E8EB", lw=0.5)
        ax.set_axisbelow(True)
    save_figure(fig, "roc-pr-curves")

    fig, ax = plt.subplots(figsize=(3.35, 2.4), layout="constrained")
    bins = np.linspace(0, 1, 26)
    for label, color, legend in [(0, NORMAL, "Normal"), (1, DISEASE, "Disease")]:
        values = scores.loc[y == label, "ensemble_score"].to_numpy()
        ax.hist(values, bins=bins, weights=np.full(len(values), 1 / len(values)),
                color=color, alpha=0.64, label=legend, edgecolor="white", linewidth=0.3)
    threshold = results["ensemble_score"]["threshold"]
    ax.axvline(threshold, color=COLORS["ensemble_score"], ls="--", lw=1.2,
               label=rf"$\tau^*={threshold:.4f}$")
    ax.set(xlabel="Fused anomaly score", ylabel="Within-class proportion", xlim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    save_figure(fig, "score-distribution")

    matrix = np.array(results["ensemble_score"]["confusion_matrix"])
    row_rates = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.35, 2.55), layout="constrained")
    ax.imshow(row_rates, cmap="Blues", vmin=0, vmax=1)
    for row in range(2):
        for col in range(2):
            color = "white" if row_rates[row, col] > 0.55 else "#192636"
            ax.text(col, row, f"{matrix[row, col]:,}\n({row_rates[row, col]:.2%})",
                    ha="center", va="center", color=color, fontsize=11)
    ax.set_xticks([0, 1], ["Normal", "Disease"])
    ax.set_yticks([0, 1], ["Normal", "Disease"])
    ax.set(xlabel="Predicted label", ylabel="True label")
    ax.tick_params(length=0, pad=6)
    ax.spines[["left", "bottom"]].set_visible(False)
    save_figure(fig, "confusion-matrix")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
