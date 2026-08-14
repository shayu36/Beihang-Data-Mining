#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 3: Global Spatial Distribution Analysis
=============================================
1. Global scatter: colour=depth, marker size=mag
2. 10°×10° grid (left-closed, right-open for grid only)
3. summarize_box(df, lon_min, lon_max, lat_min, lat_max) — CLOSED interval
4. Three fixed windows: A=[125,150]×[25,50], B=[-85,-65]×[-60,15], C=[90,145]×[-15,15]
5. global_earthquakes.html with offline plotly + 6 hover fields

Usage:
    python task3_spatial_analysis.py
    python task3_spatial_analysis.py --input ../task1/earthquakes_prepared.csv --output-dir ./
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys, io, json
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
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent

# ── 规定的三个固定窗口 ──
FIXED_WINDOWS = {
    "Window A": (125, 150, 25, 50),
    "Window B": (-85, -65, -60, 15),
    "Window C": (90, 145, -15, 15),
}

REQUIRED_COLUMNS = ["time", "latitude", "longitude", "depth", "mag", "magType", "place", "id"]


def parse_args():
    parser = argparse.ArgumentParser(description="Task 3: Global Spatial Distribution Analysis")
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
# summarize_box — 闭区间坐标窗口统计
# ================================================================
def summarize_box(df, lon_min, lon_max, lat_min, lat_max):
    """
    Compute statistics for earthquakes within a CLOSED interval
    [lon_min, lon_max] × [lat_min, lat_max].

    Parameters
    ----------
    df : pd.DataFrame (must contain: time, latitude, longitude, depth, mag, magType, place, id)
    lon_min, lon_max : float — -180 <= lon_min <= lon_max <= 180
    lat_min, lat_max : float — -90 <= lat_min <= lat_max <= 90

    Returns
    -------
    dict with: lon_min, lon_max, lat_min, lat_max, event_count,
               count_2024, count_2025, count_mag_ge_6, max_mag, median_depth_km
    """
    # 参数校验
    if not (-180 <= lon_min <= lon_max <= 180):
        raise ValueError(f"经度范围无效: [{lon_min}, {lon_max}]")
    if not (-90 <= lat_min <= lat_max <= 90):
        raise ValueError(f"纬度范围无效: [{lat_min}, {lat_max}]")

    # 检查必需列
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame 缺少必需字段: {missing}")

    # 闭区间筛选
    mask = (
        (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
        & (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max)
    )
    sub = df[mask]

    if len(sub) == 0:
        return {
            "lon_min": lon_min, "lon_max": lon_max,
            "lat_min": lat_min, "lat_max": lat_max,
            "event_count": 0,
            "count_2024": 0, "count_2025": 0,
            "count_mag_ge_6": 0,
            "max_mag": None, "median_depth_km": None,
        }

    sub_2024 = sub[sub["time"].dt.year == 2024]
    sub_2025 = sub[sub["time"].dt.year == 2025]

    return {
        "lon_min": lon_min, "lon_max": lon_max,
        "lat_min": lat_min, "lat_max": lat_max,
        "event_count": int(len(sub)),
        "count_2024": int(len(sub_2024)),
        "count_2025": int(len(sub_2025)),
        "count_mag_ge_6": int((sub["mag"] >= 6.0).sum()),
        "max_mag": round(sub["mag"].max(), 2) if len(sub) > 0 else None,
        "median_depth_km": round(sub["depth"].median(), 1) if len(sub) > 0 else None,
    }


# ================================================================
def main():
    setup_windows_console()
    args = parse_args()
    INPUT_CSV = args.input
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    print(f"Loaded {len(df)} records")

    # ================================================================
    # (1) Global scatter — colour=depth, marker size ∝ mag
    # ================================================================
    print("\n=== (1) Global Scatter Map (colour=depth, size=mag) ===")

    # Subsample for static PNG clarity
    plot_n = min(8000, len(df))
    plot_df = df.sample(plot_n, random_state=42)

    fig, ax = plt.subplots(figsize=(16, 9))
    sizes = np.clip((plot_df["mag"] - 4.0) * 8, 1, 80)
    sc = ax.scatter(
        plot_df["longitude"], plot_df["latitude"],
        c=plot_df["depth"], cmap="viridis_r", s=sizes, alpha=0.55,
        norm=LogNorm(vmin=1, vmax=700), edgecolors="none",
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)
    ax.set_title("Global Earthquake Epicenters (2024-2025, M>=4.5)\nColour = Depth, Size = Magnitude",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle="--")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Depth (km)", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "global_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: global_scatter.png")

    # ================================================================
    # (2) 10°×10° spatial grid (left-closed, right-open)
    # ================================================================
    print("\n=== (2) 10°×10° Spatial Grid ===")

    lat_bins = np.arange(-90, 91, 10)
    lon_bins = np.arange(-180, 181, 10)
    hist2d, _, _ = np.histogram2d(df["latitude"], df["longitude"], bins=[lat_bins, lon_bins])

    grid_data = []
    for i in range(len(lat_bins) - 1):
        for j in range(len(lon_bins) - 1):
            lat_min, lat_max = lat_bins[i], lat_bins[i+1]
            lon_min, lon_max = lon_bins[j], lon_bins[j+1]
            mask = (df["latitude"] >= lat_min) & (df["latitude"] < lat_max) & \
                   (df["longitude"] >= lon_min) & (df["longitude"] < lon_max)
            cnt = mask.sum()
            sub = df[mask] if cnt > 0 else df.iloc[:0]
            grid_data.append({
                "lat_min": lat_min, "lat_max": lat_max,
                "lon_min": lon_min, "lon_max": lon_max,
                "count": int(cnt),
                "mean_mag": round(sub["mag"].mean(), 3) if cnt > 0 else None,
                "max_mag": round(sub["mag"].max(), 2) if cnt > 0 else None,
                "mean_depth": round(sub["depth"].mean(), 1) if cnt > 0 else None,
            })

    grid_df = pd.DataFrame(grid_data)
    grid_df.to_csv(OUTPUT_DIR / "spatial_grid_count.csv", index=False, encoding="utf-8-sig")
    nonempty = int((grid_df["count"] > 0).sum())
    print(f"  Non-empty cells: {nonempty} / {len(grid_df)},  count sum = {grid_df['count'].sum()}")

    # ── Heatmap ──
    fig, ax = plt.subplots(figsize=(16, 8))
    mesh = ax.pcolormesh(lon_bins, lat_bins, hist2d, cmap="YlOrRd",
                         norm=LogNorm(vmin=1, vmax=max(hist2d.max(), 10)),
                         edgecolors="lightgray", linewidth=0.3)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlabel("Longitude (°)", fontsize=13)
    ax.set_ylabel("Latitude (°)", fontsize=13)
    ax.set_title("10°×10° Spatial Grid Event Count Map", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.15, linestyle="--")
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.75)
    cbar.set_label("Event Count (log scale)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "spatial_grid_count.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: spatial_grid_count.png")

    # Top 10
    print("\n  Top 10 grid cells:")
    for _, row in grid_df.nlargest(10, "count").iterrows():
        print(f"    Lon [{row['lon_min']:6.0f}, {row['lon_max']:6.0f}]  "
              f"Lat [{row['lat_min']:6.0f}, {row['lat_max']:6.0f}]  "
              f"Count: {int(row['count']):5d}  MaxMag: {row['max_mag']:.1f}")

    # ================================================================
    # (3) summarize_box 函数测试
    # ================================================================
    print("\n=== (3) summarize_box Function ===")
    test = summarize_box(df, 120, 150, 30, 45)
    print(f"  Test (120-150°E, 30-45°N): count={test['event_count']}, "
          f"max_mag={test['max_mag']}, median_depth={test['median_depth_km']} km")
    print("  Function 'summarize_box(df, lon_min, lon_max, lat_min, lat_max)' ready.")

    # ================================================================
    # (4) 三个固定窗口
    # ================================================================
    print("\n=== (4) Three Fixed Windows ===")

    window_results = []
    for name, (lon_min, lon_max, lat_min, lat_max) in FIXED_WINDOWS.items():
        res = summarize_box(df, lon_min, lon_max, lat_min, lat_max)
        window_results.append(res)
        print(f"\n  {name}: Lon [{lon_min}, {lon_max}], Lat [{lat_min}, {lat_max}]")
        print(f"    event_count={res['event_count']}, 2024={res['count_2024']}, 2025={res['count_2025']}")
        print(f"    count_mag_ge_6={res['count_mag_ge_6']}, max_mag={res['max_mag']}, "
              f"median_depth={res['median_depth_km']} km")

    window_df = pd.DataFrame(window_results)
    window_df.to_csv(OUTPUT_DIR / "coordinate_windows.csv", index=False, encoding="utf-8-sig")
    print("\n  -> Saved: coordinate_windows.csv")

    # ── Plot: 3-window overview ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors_win = ["#4A90D9", "#50B86A", "#E0554A"]
    for idx, (name, (lon_min, lon_max, lat_min, lat_max)) in enumerate(FIXED_WINDOWS.items()):
        ax = axes[idx]
        ax.scatter(df["longitude"], df["latitude"], c="#dddddd", s=0.2, alpha=0.3)
        rect = Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                         linewidth=2, edgecolor=colors_win[idx], facecolor="none", linestyle="-")
        ax.add_patch(rect)
        mask = (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max) & \
               (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max)
        sub = df[mask]
        ax.scatter(sub["longitude"], sub["latitude"], c=colors_win[idx], s=3, alpha=0.7)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_title(f"{name}\nn={len(sub)}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "coordinate_windows.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: coordinate_windows.png")

    # ================================================================
    # (5) global_earthquakes.html — offline Plotly
    # ================================================================
    print("\n=== (5) global_earthquakes.html ===")
    try:
        import plotly.express as px
        html_sample = df.sample(min(5000, len(df)), random_state=42)
        fig_px = px.scatter_geo(
            html_sample, lat="latitude", lon="longitude",
            size="mag", color="depth",
            hover_name="place",
            hover_data={
                "id": True, "time": True, "mag": ":.1f",
                "magType": True, "depth": ":.1f", "place": False,
            },
            projection="natural earth",
            title="Latitude/Longitude Interactive Scatter Plot — 2024-2025 Global M4.5+ Earthquakes (sampled 5000)",
            color_continuous_scale="Viridis_r",
            size_max=15,
        )
        fig_px.update_layout(height=650, margin=dict(l=0, r=0, t=50, b=0))
        fig_px.write_html(
            str(OUTPUT_DIR / "global_earthquakes.html"),
            include_plotlyjs=True,  # offline-ready
        )
        print("  -> Saved: global_earthquakes.html (offline, 5000 sampled points)")
    except ImportError:
        print("  (Plotly not installed; skipping interactive HTML)")

    # ================================================================
    print("\n" + "=" * 60)
    print("Task 3 Complete!")
    print("=" * 60)
    print("  global_scatter.png")
    print("  spatial_grid_count.csv")
    print("  spatial_grid_count.png")
    print("  coordinate_windows.csv")
    print("  coordinate_windows.png")
    print("  global_earthquakes.html")


if __name__ == "__main__":
    main()
