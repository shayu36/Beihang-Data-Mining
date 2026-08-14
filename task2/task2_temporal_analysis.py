#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2: Temporal Statistics and Magnitude/Depth Structure
=========================================================
1. Continuous 24-month sequence (2024-01 ~ 2025-12)
2. Annual summary: total, monthly mean/median/max/min
3. Magnitude distribution: 4 groups, yearly count + percentage
4. Depth distribution: 3 groups, yearly count + percentage
5. magType: overall + yearly count + percentage
6. mag_group × depth_group cross-tab (count + row-normalised %)
7. depth == 10 km: count, percentage, explanation

Usage:
    python task2_temporal_analysis.py
    python task2_temporal_analysis.py --input ../task1/earthquakes_prepared.csv --output-dir ./
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys, io
import warnings
warnings.filterwarnings("ignore")

def setup_windows_console():
    """Re-wrap stdout as UTF-8 (called from main() only — wrapping at
    import time breaks pytest's capture machinery)."""
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent

# ── 规定分组 ──
MAG_BINS   = [4.5, 5.0, 6.0, 7.0, np.inf]
MAG_LABELS = ["[4.5,5.0)", "[5.0,6.0)", "[6.0,7.0)", "[7.0,+inf)"]

DEPTH_BINS   = [0, 70, 300, np.inf]
DEPTH_LABELS = ["[0,70)", "[70,300)", "[300,+inf)"]


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2: Temporal and Magnitude/Depth Analysis")
    parser.add_argument(
        "--input", type=Path,
        default=BASE_DIR.parent / "task1" / "earthquakes_prepared.csv",
        help="Path to earthquakes_prepared.csv (from task 1)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR,
        help="Directory for output files",
    )
    return parser.parse_args()


# ================================================================
def main():
    setup_windows_console()
    args = parse_args()
    INPUT_CSV = args.input
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["year"]  = df["time"].dt.year

    # 确保使用规定的分组（task1 已生成则直接用）
    if "mag_group" not in df.columns:
        df["mag_group"] = pd.cut(df["mag"], bins=MAG_BINS, labels=MAG_LABELS, right=False)
    if "depth_group" not in df.columns:
        df["depth_group"] = pd.cut(df["depth"], bins=DEPTH_BINS, labels=DEPTH_LABELS, right=False)

    n = len(df)
    print(f"Loaded {n} records")
    print(f"Time range: {df['time'].min()} ~ {df['time'].max()}")

    # ================================================================
    # (1) 连续 24 个月序列
    # ================================================================
    print("\n=== (1) Monthly Earthquake Count (24 months) ===")

    full_months = pd.period_range("2024-01", "2025-12", freq="M")
    monthly = (
        df.groupby(df["time"].dt.to_period("M"))
        .size()
        .reindex(full_months, fill_value=0)
    )
    monthly.index = monthly.index.astype(str)

    monthly_df = monthly.reset_index()
    monthly_df.columns = ["year_month", "count"]
    monthly_df.to_csv(OUTPUT_DIR / "monthly_counts.csv", index=False, encoding="utf-8-sig")

    print(monthly_df.to_string(index=False))
    print(f"\n  Max: {monthly.idxmax()} ({monthly.max()})")
    print(f"  Min: {monthly.idxmin()} ({monthly.min()})")

    # ── Plot ──
    fig, ax1 = plt.subplots(figsize=(16, 7))
    x = np.arange(len(monthly))
    ax1.bar(x, monthly.values, width=0.6, color="#4A90D9", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Month", fontsize=13)
    ax1.set_ylabel("Earthquake Count", fontsize=13, color="#4A90D9")
    ax1.tick_params(axis="y", labelcolor="#4A90D9")
    ax2 = ax1.twinx()
    ax2.plot(x, monthly.values, color="#E0554A", marker="o", linewidth=2, markersize=6)
    ax2.set_ylabel("Count (line)", fontsize=13, color="#E0554A")
    ax2.tick_params(axis="y", labelcolor="#E0554A")
    labels = [str(m) for m in monthly.index]
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(labels[::2], rotation=45, ha="right", fontsize=9)
    ax1.set_title("Monthly M>=4.5 Earthquake Count (2024.01 - 2025.12)", fontsize=15, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ymax, ymin = monthly.max(), monthly.min()
    xmax, xmin = monthly.values.argmax(), monthly.values.argmin()
    ax1.annotate(f"Max: {ymax}\n{labels[xmax]}", (xmax, ymax), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=9, color="#E0554A", fontweight="bold")
    ax1.annotate(f"Min: {ymin}\n{labels[xmin]}", (xmin, ymin), textcoords="offset points",
                 xytext=(0, -18), ha="center", fontsize=9, color="#333333", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_counts.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: monthly_counts.png")

    # ================================================================
    # (2) 年度汇总：总数、月均、中位数、最大、最少
    # ================================================================
    print("\n=== (2) Annual Monthly Summary ===")

    annual_rows = []
    for yr in [2024, 2025]:
        yr_mask = monthly_df["year_month"].str.startswith(str(yr))
        yr_monthly = monthly_df.loc[yr_mask, "count"]
        annual_rows.append({
            "year": yr,
            "annual_event_count": int(yr_monthly.sum()),
            "monthly_mean": round(yr_monthly.mean(), 1),
            "monthly_median": round(float(yr_monthly.median()), 1),
            "monthly_max": int(yr_monthly.max()),
            "max_month": str(monthly_df.loc[yr_monthly.idxmax(), "year_month"])
                         if len(yr_monthly) > 0 else "",
            "monthly_min": int(yr_monthly.min()),
            "min_month": str(monthly_df.loc[yr_monthly.idxmin(), "year_month"])
                         if len(yr_monthly) > 0 else "",
        })
    annual_summary = pd.DataFrame(annual_rows)
    annual_summary.to_csv(OUTPUT_DIR / "annual_monthly_summary.csv", index=False, encoding="utf-8-sig")
    print(annual_summary.to_string(index=False))

    # ================================================================
    # (3) 震级分布（4 组）：yearly count + percentage
    # ================================================================
    print("\n=== (3) Magnitude Distribution ===")

    mag_all = df["mag_group"].value_counts().reindex(MAG_LABELS, fill_value=0)
    mag_2024 = df[df["year"] == 2024]["mag_group"].value_counts().reindex(MAG_LABELS, fill_value=0)
    mag_2025 = df[df["year"] == 2025]["mag_group"].value_counts().reindex(MAG_LABELS, fill_value=0)

    mag_df = pd.DataFrame({
        "mag_group": MAG_LABELS,
        "all_count": mag_all.values,
        "all_pct": (mag_all.values / n * 100).round(2),
        "2024_count": mag_2024.values,
        "2024_pct": (mag_2024.values / mag_2024.sum() * 100).round(2),
        "2025_count": mag_2025.values,
        "2025_pct": (mag_2025.values / mag_2025.sum() * 100).round(2),
    })
    mag_df.to_csv(OUTPUT_DIR / "magnitude_by_year.csv", index=False, encoding="utf-8-sig")
    print(mag_df.to_string(index=False))

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(MAG_LABELS))
    w = 0.25
    ax.bar(x - w, mag_df["2024_count"], w, color="#50B86A", alpha=0.85, label="2024")
    ax.bar(x,      mag_df["2025_count"], w, color="#E0554A", alpha=0.85, label="2025")
    ax.set_xticks(x)
    ax.set_xticklabels(MAG_LABELS, fontsize=11)
    ax.set_xlabel("Magnitude Group", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Magnitude Distribution by Year", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    for bar_group in ax.containers:
        ax.bar_label(bar_group, fmt="%d", fontsize=8, padding=2)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "magnitude_by_year.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: magnitude_by_year.png")

    # ================================================================
    # (4) 深度分布（3 组）：yearly count + percentage
    # ================================================================
    print("\n=== (4) Depth Distribution ===")

    dep_all = df["depth_group"].value_counts().reindex(DEPTH_LABELS, fill_value=0)
    dep_2024 = df[df["year"] == 2024]["depth_group"].value_counts().reindex(DEPTH_LABELS, fill_value=0)
    dep_2025 = df[df["year"] == 2025]["depth_group"].value_counts().reindex(DEPTH_LABELS, fill_value=0)

    dep_df = pd.DataFrame({
        "depth_group": DEPTH_LABELS,
        "all_count": dep_all.values,
        "all_pct": (dep_all.values / n * 100).round(2),
        "2024_count": dep_2024.values,
        "2024_pct": (dep_2024.values / dep_2024.sum() * 100).round(2),
        "2025_count": dep_2025.values,
        "2025_pct": (dep_2025.values / dep_2025.sum() * 100).round(2),
    })
    dep_df.to_csv(OUTPUT_DIR / "depth_by_year.csv", index=False, encoding="utf-8-sig")
    print(dep_df.to_string(index=False))

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["#4A90D9", "#F5A623", "#E0554A"]
    ax1.pie(dep_df["all_count"], labels=DEPTH_LABELS, autopct="%1.1f%%",
            colors=colors, startangle=140)
    ax1.set_title("Depth Distribution (All)", fontsize=13, fontweight="bold")

    x = np.arange(len(DEPTH_LABELS))
    w = 0.3
    ax2.bar(x - w/2, dep_df["2024_count"], w, color="#50B86A", alpha=0.85, label="2024")
    ax2.bar(x + w/2, dep_df["2025_count"], w, color="#E0554A", alpha=0.85, label="2025")
    ax2.set_xticks(x)
    ax2.set_xticklabels(DEPTH_LABELS, fontsize=11)
    ax2.set_title("Depth Distribution by Year", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "depth_by_year.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: depth_by_year.png")

    # ================================================================
    # (5) magType：总体 + yearly count + percentage
    # ================================================================
    print("\n=== (5) magType Distribution ===")

    mt_all = df["magType"].value_counts()
    mt_2024 = df[df["year"] == 2024]["magType"].value_counts()
    mt_2025 = df[df["year"] == 2025]["magType"].value_counts()

    mt_df = pd.DataFrame({
        "magType": mt_all.index,
        "all_count": mt_all.values,
        "all_pct": (mt_all.values / n * 100).round(2),
    })
    mt_df["2024_count"] = mt_df["magType"].map(mt_2024).fillna(0).astype(int)
    mt_df["2024_pct"]  = (mt_df["2024_count"] / mt_df["2024_count"].sum() * 100).round(2)
    mt_df["2025_count"] = mt_df["magType"].map(mt_2025).fillna(0).astype(int)
    mt_df["2025_pct"]  = (mt_df["2025_count"] / mt_df["2025_count"].sum() * 100).round(2)
    mt_df.to_csv(OUTPUT_DIR / "magtype_by_year.csv", index=False, encoding="utf-8-sig")
    print(mt_df.to_string(index=False))

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    mtypes = mt_df["magType"].tolist()
    ax1.barh(mtypes, mt_df["all_count"], color="#4A90D9", alpha=0.85)
    ax1.set_xlabel("Count", fontsize=12)
    ax1.set_title("magType (Overall)", fontsize=13, fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3, linestyle="--")

    x = np.arange(len(mtypes))
    w = 0.3
    ax2.barh(x - w/2, mt_df["2024_count"], w, color="#50B86A", alpha=0.85, label="2024")
    ax2.barh(x + w/2, mt_df["2025_count"], w, color="#E0554A", alpha=0.85, label="2025")
    ax2.set_yticks(x)
    ax2.set_yticklabels(mtypes)
    ax2.set_xlabel("Count", fontsize=12)
    ax2.set_title("magType by Year", fontsize=13, fontweight="bold")
    ax2.invert_yaxis()
    ax2.legend(fontsize=10)
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "magtype_by_year.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: magtype_by_year.png")

    # ================================================================
    # (6) mag_group × depth_group 交叉表
    # ================================================================
    print("\n=== (6) Magnitude-Depth Cross-tab ===")

    cross_count = pd.crosstab(df["mag_group"], df["depth_group"]).reindex(
        index=MAG_LABELS, columns=DEPTH_LABELS, fill_value=0,
    )
    cross_pct   = (
        pd.crosstab(df["mag_group"], df["depth_group"], normalize="index") * 100
    ).reindex(index=MAG_LABELS, columns=DEPTH_LABELS, fill_value=0.0)

    cross_combined = cross_count.astype(str)
    for mg in MAG_LABELS:
        for dg in DEPTH_LABELS:
            if mg in cross_count.index and dg in cross_count.columns:
                cross_combined.loc[mg, dg] = (
                    f"{cross_count.loc[mg, dg]} ({cross_pct.loc[mg, dg]:.1f}%)"
                )
    cross_combined.to_csv(OUTPUT_DIR / "magnitude_depth_crosstab.csv", encoding="utf-8-sig")
    print(cross_combined.to_string())

    # ================================================================
    # (7) depth == 10 km 统计
    # ================================================================
    print("\n=== (7) Depth == 10 km Statistics ===")

    depth_10 = np.isclose(df["depth"], 10.0)
    depth_10_count = int(depth_10.sum())
    depth_10_pct   = depth_10_count / n * 100

    depth10_df = pd.DataFrame([{
        "depth_km": 10,
        "count": depth_10_count,
        "percentage": round(depth_10_pct, 2),
        "note": ("10 km 处集中可能反映目录中固定深度的使用习惯，"
                 "不意味着这些震源深度均被精确测定为 10 km。"),
    }])
    depth10_df.to_csv(OUTPUT_DIR / "depth_10km_summary.csv", index=False, encoding="utf-8-sig")
    print(f"  depth == 10 km: {depth_10_count} 条 ({depth_10_pct:.2f}%)")
    print("  说明: 10 km 处集中可能反映目录定位中固定深度的使用，"
          "并不意味着这些震源深度均被精确测定为 10 km。")

    # ================================================================
    # Summary
    # ================================================================
    print("\n说明：")
    print("上述月度数量表示当前冻结版本目录中符合检索条件的事件数量。")
    print("2024—2025 年两年的数据不足以判断地震活动的长期趋势，")
    print("也不能用于预测未来地震或评估灾害风险。")
    print("\n" + "=" * 60)
    print("Task 2 Complete!")
    print("=" * 60)
    for f in [
        "monthly_counts.csv", "annual_monthly_summary.csv",
        "magnitude_by_year.csv", "depth_by_year.csv",
        "magtype_by_year.csv", "magnitude_depth_crosstab.csv",
        "depth_10km_summary.csv",
        "monthly_counts.png", "magnitude_by_year.png",
        "depth_by_year.png", "magtype_by_year.png",
    ]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
