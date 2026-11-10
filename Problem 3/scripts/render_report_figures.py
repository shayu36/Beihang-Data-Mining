"""Render report figures from the existing dataset and saved metrics, without training.

The report covers only the current GRU ensemble + Ridge model. Colors identify
methods consistently. Existing prediction and loss images are never overwritten.
Raw observations use actual timestamps; split distributions follow current code.
"""
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data_utils import load_and_clean, split_chronological

OUT = ROOT / "results" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)
METRICS = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
KEYS = ["Persistence", "Ridge", "GRU-ensemble", "Final(GRU+Ridge)"]
LABELS = ["持久化基线", "Ridge", "GRU 集成", "GRU + Ridge"]
COLORS = ["#8996A8", "#D58B33", "#397AA8", "#18887F"]
plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#77818C",
    "axes.labelcolor": "#253342",
    "text.color": "#253342",
    "xtick.color": "#445566",
    "ytick.color": "#445566",
    "savefig.dpi": 220,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(name)


def metrics_comparison():
    fig, axs = plt.subplots(2, 2, figsize=(11.2, 7.2), layout="constrained")
    specs = [
        ("RMSE", "RMSE（越低越好）", 7.2),
        ("MAE", "MAE（越低越好）", 3.5),
        ("R2", "R²（越高越好）", 1.14),
        ("MAPE(%)", "MAPE / %（越低越好）", 0.81),
    ]
    for ax, (key, title, xmax) in zip(axs.flat, specs):
        values = [METRICS[k][key] for k in KEYS]
        assert all(np.isfinite(values))
        ax.barh(LABELS, values, color=COLORS, height=0.58)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_title(title, loc="left", pad=12)
        ax.grid(axis="x", alpha=0.17)
        ax.set_axisbelow(True)
        for i, v in enumerate(values):
            ax.text(v + xmax * 0.018, i, f"{v:.4f}", va="center", fontsize=10.5)
        if key in ("RMSE", "MAE"):
            ax.set_xlabel("CO₂ 浓度 / ppm")
        if key == "R2":
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    save(fig, "metrics_comparison.png")


def relative_improvement():
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 3.5), layout="constrained")
    final = METRICS[KEYS[-1]]
    for ax, key in zip(axs, ["RMSE", "MAE"]):
        gains = [100 * (METRICS[k][key] - final[key]) / METRICS[k][key] for k in KEYS[:-1]]
        bars = ax.barh(LABELS[:-1], gains,
                       color=["#18887F" if v >= 0 else "#BC654D" for v in gains], height=0.55)
        ax.invert_yaxis()
        ax.axvline(0, color="#566270", lw=0.9)
        ax.set_xlim(-8.2, 16.5)
        ax.set_title(f"最终融合的 {key} 相对下降幅度", loc="left", pad=12)
        ax.set_xlabel("相对下降 / %（负值表示误差增加）")
        ax.grid(axis="x", alpha=0.17)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, gains):
            ax.text(val + (0.35 if val >= 0 else -0.35),
                    bar.get_y() + bar.get_height()/2, f"{val:+.2f}%",
                    va="center", ha="left" if val >= 0 else "right")
    save(fig, "relative_improvement.png")


def data_overview():
    raw = pd.read_csv(ROOT / "data" / "weather.csv", encoding="gbk")
    dates = pd.to_datetime(raw["date"], format="%Y/%m/%d %H:%M")
    target = raw["OT"].mask(raw["OT"] <= -9998)
    fig, ax = plt.subplots(figsize=(11.2, 3.6), layout="constrained")
    ax.plot(dates, target, color="#397AA8", lw=0.6, rasterized=True)
    ax.set_title("原始 CO₂ 浓度观测：按真实日期展示", loc="left", pad=12)
    ax.set_ylabel("CO₂ 浓度 / ppm")
    ax.set_xlabel("真实日期；异常占位值置为空缺，不做插值")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.17)
    save(fig, "ot_observations.png")


def split_distribution():
    df, _ = load_and_clean()
    y = df["OT"].to_numpy(dtype=np.float32)[12:]
    tr, va, te, *_ = split_chronological(len(y))
    bins = np.linspace(float(y.min()), float(y.max()), 46)
    fig, ax = plt.subplots(figsize=(9.6, 3.8), layout="constrained")
    for idx, name, color, style in zip(
        [tr, va, te], ["训练集", "验证集", "测试集"],
        ["#397AA8", "#D58B33", "#18887F"], ["-", "--", "-."]
    ):
        ax.hist(y[idx], bins=bins, density=True, histtype="step", linewidth=1.7,
                linestyle=style, color=color, label=f"{name}（n={len(idx):,}）")
    ax.set_title("当前代码划分下的 CO₂ 浓度标签分布", loc="left", pad=12)
    ax.set_xlabel("CO₂ 浓度 / ppm")
    ax.set_ylabel("概率密度")
    ax.legend(frameon=False)
    ax.grid(alpha=0.17)
    save(fig, "split_target_distribution.png")


def fusion_weights():
    w = float(METRICS["best_weight"])
    assert 0 <= w <= 1
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.4), layout="constrained")
    axs[0].barh(["GRU 集成", "Ridge"], [w*100, (1-w)*100],
                color=[COLORS[2], COLORS[1]], height=0.5)
    axs[0].invert_yaxis()
    axs[0].set_title("最终融合系数", loc="left", pad=12)
    for i, v in enumerate([w*100, (1-w)*100]):
        axs[0].text(v+1.5, i, f"{v:.0f}%", va="center")
    axs[0].set_xlim(0, 108)
    axs[1].barh([f"GRU seed {s}" for s in range(5)], [20]*5,
                color=COLORS[2], height=0.55)
    axs[1].invert_yaxis()
    axs[1].set_title("GRU 集成内部的均值权重", loc="left", pad=12)
    for i in range(5):
        axs[1].text(21, i, "20%", va="center")
    axs[1].set_xlim(0, 30)
    for ax in axs:
        ax.set_xlabel("预测值的线性组合权重 / %")
        ax.grid(axis="x", alpha=0.17)
        ax.set_axisbelow(True)
    save(fig, "fusion_weights.png")


if __name__ == "__main__":
    metrics_comparison()
    relative_improvement()
    data_overview()
    split_distribution()
    fusion_weights()
