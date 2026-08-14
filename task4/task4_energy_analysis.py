#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4: Representative Earthquakes and Relative Energy Analysis
================================================================
1. Top 10 sorted by mag↓, time↑, id↑ — outputs id + all fields
2. Monthly relative energy index: E_rel = 10**(1.5*(M-4.5))
3. Monthly: event_count, relative_energy_sum, max_mag
4. Strongest event's share of total relative energy index
5. Explanation that this is a proxy index, not joules / intensity / risk

Usage:
    python task4_energy_analysis.py
    python task4_energy_analysis.py --input ../task1/earthquakes_prepared.csv --output-dir ./
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

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Task 4: Top 10 and Relative Energy Analysis")
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


def main():
    setup_windows_console()
    args = parse_args()
    INPUT_CSV = args.input
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    n = len(df)
    print(f"Loaded {n} records\n")

    # ================================================================
    # (1) Top 10 — mag↓, time↑, id↑
    # ================================================================
    print("=" * 70)
    print("(1) Top 10 Earthquake Events (mag↓, time↑, id↑)")
    print("=" * 70)

    top10 = (
        df.sort_values(["mag", "time", "id"], ascending=[False, True, True], kind="mergesort")
        .head(10)
        [["id", "time", "latitude", "longitude", "depth", "mag", "magType", "place"]]
        .copy()
    )
    top10["time_str"] = top10["time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    top10 = top10.reset_index(drop=True)
    top10.index = top10.index + 1
    top10.index.name = "Rank"

    print(f"\n{'Rank':<5} {'ID':<18} {'Time':<28} {'Lat':>8} {'Lon':>9} "
          f"{'Depth':>7} {'Mag':>6} {'Type':>8}  Place")
    print("-" * 120)
    for i, row in top10.iterrows():
        print(f"{i:<5} {row['id']:<18} {row['time_str']:<28} {row['latitude']:>8.3f} "
              f"{row['longitude']:>9.3f} {row['depth']:>7.1f} {row['mag']:>6.1f} "
              f"{row['magType']:>8}  {row['place']}")

    top10_out = top10.drop(columns=["time"])
    top10_out.to_csv(OUTPUT_DIR / "top10_events.csv", encoding="utf-8-sig")
    print(f"\n  -> Saved: top10_events.csv")

    # ================================================================
    # (2) 相对能量指数 — E_rel = 10**(1.5*(M-4.5))
    # ================================================================
    print("\n" + "=" * 70)
    print("(2) Monthly Relative Energy Index")
    print("=" * 70)

    # 规定公式: E_rel = 10**(1.5*(M-4.5)) —— M=4.5 → 1.0
    df["relative_energy_index"] = 10 ** (1.5 * (df["mag"] - 4.5))

    # 最强事件占比
    strongest_idx = df["relative_energy_index"].idxmax()
    strongest_event = df.loc[strongest_idx]
    strongest_share_pct = (
        strongest_event["relative_energy_index"]
        / df["relative_energy_index"].sum()
        * 100
    )
    print(f"\n  最强事件: M{strongest_event['mag']:.1f}  {strongest_event['place']}")
    print(f"  最强事件相对能量指数: {strongest_event['relative_energy_index']:.2e}")
    print(f"  占全部指数总和的比例: {strongest_share_pct:.2f}%")

    # 月度汇总
    df["year_month"] = df["time"].dt.to_period("M").astype(str)
    monthly = df.groupby("year_month").agg(
        event_count=("id", "count"),
        relative_energy_sum=("relative_energy_index", "sum"),
        max_mag=("mag", "max"),
    ).reset_index()

    monthly["energy_pct"] = (
        monthly["relative_energy_sum"] / monthly["relative_energy_sum"].sum() * 100
    ).round(2)

    monthly.to_csv(OUTPUT_DIR / "monthly_relative_energy.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'Month':<8} {'Events':>7} {'RelEnergySum':>16} {'Energy%':>8} {'MaxMag':>6}")
    print("-" * 50)
    for _, row in monthly.iterrows():
        print(f"{row['year_month']:<8} {int(row['event_count']):>7} "
              f"{row['relative_energy_sum']:>16.4e} {row['energy_pct']:>7.2f}% "
              f"{row['max_mag']:>6.1f}")

    # ── 说明 ──
    print("""
  ⚠ 重要说明：
  本指标为近似代理指标 (proxy index)，不是实际地震能量（焦耳），
  不是烈度，也不是损害或灾害风险评估。
  公式: E_rel = 10^(1.5 × (M - 4.5))  ——  M=4.5 → E_rel=1。
  该指标仅用于比较不同地震事件和月份的相对强度，不能直接解释为
  物理能量或风险等级。
""")

    # ── Key comparison ──
    month_max_count  = monthly.loc[monthly["event_count"].idxmax(), "year_month"]
    month_max_energy = monthly.loc[monthly["relative_energy_sum"].idxmax(), "year_month"]
    mc = monthly.loc[monthly["year_month"] == month_max_count, "event_count"].values[0]
    me = monthly.loc[monthly["year_month"] == month_max_energy, "energy_pct"].values[0]
    print(f"  地震次数最多月份: {month_max_count} ({int(mc)} events)")
    print(f"  相对能量最高月份: {month_max_energy} ({me:.2f}%)")

    # ── Plot: dual-axis ──
    months = monthly["year_month"].tolist()
    x = np.arange(len(months))
    fig, ax1 = plt.subplots(figsize=(16, 7))
    ax1.bar(x, monthly["event_count"], width=0.55, color="#4A90D9", alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Month", fontsize=13)
    ax1.set_ylabel("Event Count", fontsize=13, color="#4A90D9")
    ax1.tick_params(axis="y", labelcolor="#4A90D9")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(months[::2], rotation=45, ha="right", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, monthly["energy_pct"], color="#E0554A", marker="s", linewidth=2.5, markersize=7)
    ax2.set_ylabel("Relative Energy Index (%)", fontsize=13, color="#E0554A")
    ax2.tick_params(axis="y", labelcolor="#E0554A")

    ymax_c = monthly["event_count"].max()
    xmax_c = monthly["event_count"].values.argmax()
    ax1.annotate(f"Most events: {months[xmax_c]}\n({int(ymax_c)})",
                 (xmax_c, ymax_c), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=9, color="#4A90D9", fontweight="bold")
    ymax_e = monthly["energy_pct"].max()
    xmax_e = monthly["energy_pct"].values.argmax()
    ax2.annotate(f"Highest energy: {months[xmax_e]}\n({ymax_e:.1f}%)",
                 (xmax_e, ymax_e), textcoords="offset points",
                 xytext=(0, -20), ha="center", fontsize=9, color="#E0554A", fontweight="bold")
    ax1.set_title("Monthly Event Count vs Relative Energy Index", fontsize=15, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_relative_energy.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("\n  -> Saved: monthly_relative_energy.png")

    # ── Scatter: count vs mean energy per event ──
    monthly["mean_rei_per_event"] = monthly["relative_energy_sum"] / monthly["event_count"]
    monthly["plot_year"] = [m[:4] for m in months]

    fig, ax = plt.subplots(figsize=(12, 7))
    for yr, color in [("2024", "#50B86A"), ("2025", "#E0554A")]:
        s = monthly[monthly["plot_year"] == yr]
        ax.scatter(s["event_count"], s["mean_rei_per_event"],
                   c=color, s=s["event_count"] / 3, alpha=0.75,
                   edgecolors="#333333", linewidth=0.5, label=f"{yr}", zorder=3)

    # Label notable months
    threshold = monthly["mean_rei_per_event"].quantile(0.80)
    for _, row in monthly.iterrows():
        is_notable = (row["mean_rei_per_event"] >= threshold or
                      row["event_count"] >= monthly["event_count"].quantile(0.80))
        lbl = row["year_month"] if is_notable else row["year_month"].split("-")[1]
        fs = 8 if is_notable else 6.5
        alpha = 0.9 if is_notable else 0.45
        ax.annotate(lbl, (row["event_count"], row["mean_rei_per_event"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=fs, alpha=alpha)

    ax.set_xlabel("Event Count", fontsize=13)
    ax.set_ylabel("Mean Relative Energy Index per Event", fontsize=13)
    ax.set_title("Count vs Mean Relative Energy per Event — 2024 vs 2025", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "count_vs_energy.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: count_vs_energy.png")

    # ── Energy summary CSV ──
    energy_summary = pd.DataFrame([{
        "strongest_event_mag": strongest_event["mag"],
        "strongest_event_place": strongest_event["place"],
        "strongest_event_rel_index": strongest_event["relative_energy_index"],
        "total_rel_index": df["relative_energy_index"].sum(),
        "strongest_share_pct": round(strongest_share_pct, 2),
        "formula": "E_rel = 10^(1.5*(M-4.5))",
        "note": "Proxy index only — not joules, not intensity, not risk.",
    }])
    energy_summary.to_csv(OUTPUT_DIR / "energy_summary.csv", index=False, encoding="utf-8-sig")

    # ================================================================
    print("\n" + "=" * 70)
    print("Task 4 Complete!")
    print("=" * 70)
    print("  top10_events.csv")
    print("  monthly_relative_energy.csv")
    print("  energy_summary.csv")
    print("  monthly_relative_energy.png")
    print("  count_vs_energy.png")


if __name__ == "__main__":
    main()
