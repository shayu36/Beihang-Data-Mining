#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 6: ML Analysis — DBSCAN Spatial Clustering + PCA Feature Analysis
======================================================================
- DBSCAN: density-based clustering using Haversine (great-circle) distance
  on the sphere. Correctly handles date-line wrap-around and latitude-dependent
  degree lengths.
- Multi-parameter comparison: eps = 100, 200, 300, 400, 500 km
  with min_samples = 5, 10, 20
- PCA: dimensionality reduction on numerical features to reveal structure.

Usage:
    python task6_ml_analysis.py
    python task6_ml_analysis.py --input task1_processed_data.csv --output-dir ./outputs
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys, io, warnings
warnings.filterwarnings("ignore")

def setup_windows_console():
    """Re-wrap stdout as UTF-8 (called from main() only — wrapping at
    import time breaks pytest's capture machinery)."""
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent
EARTH_RADIUS_KM = 6371.0088


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 6: ML Analysis — DBSCAN + PCA"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR.parent / "task1" / "earthquakes_prepared.csv",
        help="Path to processed CSV (from task 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR,
        help="Directory for output files",
    )
    return parser.parse_args()


# ================================================================
# Haversine DBSCAN helpers
# ================================================================
def latlon_to_radians(df):
    """Convert (latitude, longitude) degrees to radians."""
    return np.radians(df[["latitude", "longitude"]].to_numpy())


def eps_km_to_rad(eps_km):
    """Convert distance in km to radians on the Earth's sphere."""
    return eps_km / EARTH_RADIUS_KM


def run_haversine_dbscan(coords_rad, eps_km, min_samples):
    """
    Run DBSCAN with Haversine (great-circle) distance.

    Parameters
    ----------
    coords_rad : np.ndarray of shape (n, 2)
        (latitude_rad, longitude_rad)
    eps_km : float
        Neighbourhood radius in kilometres.
    min_samples : int
        Minimum points to form a core point.

    Returns
    -------
    labels : np.ndarray
        Cluster labels (-1 = noise).
    """
    eps_rad = eps_km_to_rad(eps_km)
    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    return db.fit_predict(coords_rad)


def circular_mean_longitude(longitude: pd.Series) -> float:
    """Circular mean of longitudes, correct across ±180° date line."""
    angles = np.radians(longitude.dropna().to_numpy())
    if len(angles) == 0:
        return float("nan")
    mean_sin = np.sin(angles).mean()
    mean_cos = np.cos(angles).mean()
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)))


def cluster_stats(labels, df):
    """Compute per-cluster statistics."""
    rows = []
    unique_labels = sorted(set(labels))
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        sub = df[mask]
        rows.append({
            "cluster_id": int(lbl),
            "count": len(sub),
            "mean_mag": round(sub["mag"].mean(), 2),
            "max_mag": round(sub["mag"].max(), 1),
            "mean_depth": round(sub["depth"].mean(), 1),
            "center_lon": round(circular_mean_longitude(sub["longitude"]), 1),
            "center_lat": round(sub["latitude"].mean(), 1),
            "time_start": str(sub["time"].min().date()),
            "time_end": str(sub["time"].max().date()),
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)


# ================================================================
# Main
# ================================================================
def main():
    setup_windows_console()
    args = parse_args()
    INPUT_CSV = args.input
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Read ──
    df = pd.read_csv(INPUT_CSV)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    print(f"Loaded {len(df)} records")

    # Convert to radians for Haversine distance
    coords_rad = latlon_to_radians(df)

    # ================================================================
    # 1. DBSCAN — Haversine distance with multi-parameter comparison
    # ================================================================
    print("\n=== (1) DBSCAN Spatial Clustering (Haversine distance) ===")
    print("Using great-circle (Haversine) distance on the sphere.")
    print("This correctly handles date-line wrap-around and latitude-dependent")
    print("degree lengths, unlike Euclidean distance on raw (lon, lat).\n")

    # --- Step 1: k-distance graph to guide eps choice ---
    k = 10
    # Use haversine metric for k-distance too
    nbrs = NearestNeighbors(n_neighbors=k, metric="haversine").fit(coords_rad)
    distances_rad, _ = nbrs.kneighbors(coords_rad)
    # Convert to km
    k_dist_km = np.sort(distances_rad[:, -1]) * EARTH_RADIUS_KM

    # --- Step 2: Compare multiple parameter sets ---
    eps_values = [100, 200, 300, 400, 500]
    min_samples_values = [5, 10, 20]

    print("  Parameter comparison (Haversine DBSCAN):")
    print(f"  {'eps(km)':<10} {'min_s':<8} {'n_clusters':<12} {'n_noise':<10} {'noise%':<8} {'silhouette':<12} {'max_cluster%':<14}")
    print("  " + "-" * 75)

    param_results = []
    for eps_km in eps_values:
        for min_s in min_samples_values:
            labels = run_haversine_dbscan(coords_rad, eps_km, min_s)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            noise_pct = n_noise / len(df) * 100

            # Silhouette (exclude noise)
            sil = None
            if n_clusters >= 2:
                mask = labels != -1
                if mask.sum() > 1 and len(set(labels[mask])) >= 2:
                    # Sample for performance
                    n_sample = min(5000, mask.sum())
                    idx = np.where(mask)[0]
                    if len(idx) > n_sample:
                        idx = np.random.RandomState(42).choice(idx, n_sample, replace=False)
                    try:
                        sil = silhouette_score(coords_rad[idx], labels[idx], metric="haversine")
                    except Exception:
                        sil = None

            # Max cluster percentage
            cluster_sizes = pd.Series(labels[labels != -1]).value_counts()
            max_cluster_pct = (cluster_sizes.iloc[0] / len(df) * 100) if len(cluster_sizes) > 0 else 0

            sil_str = f"{sil:.4f}" if sil is not None else "N/A"
            print(f"  {eps_km:<10} {min_s:<8} {n_clusters:<12} {n_noise:<10} {noise_pct:<7.1f}% {sil_str:<12} {max_cluster_pct:<13.1f}%")

            param_results.append({
                "eps_km": eps_km,
                "min_samples": min_s,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": round(noise_pct, 1),
                "silhouette_score": round(sil, 4) if sil is not None else None,
                "max_cluster_pct": round(max_cluster_pct, 1),
            })

    param_df = pd.DataFrame(param_results)
    param_df.to_csv(OUTPUT_DIR / "task6_dbscan_params.csv", index=False, encoding="utf-8-sig")

    # --- Select best parameters ---
    # Prefer: silhouette > 0.3, noise% < 30%, max_cluster% < 60%
    valid = param_df[
        (param_df["silhouette_score"].notna()) &
        (param_df["silhouette_score"] > 0.3) &
        (param_df["noise_pct"] < 30) &
        (param_df["max_cluster_pct"] < 60)
    ]
    if len(valid) > 0:
        best = valid.loc[valid["silhouette_score"].idxmax()]
    else:
        # Fallback: highest silhouette
        valid2 = param_df[param_df["silhouette_score"].notna()]
        best = valid2.loc[valid2["silhouette_score"].idxmax()] if len(valid2) > 0 else param_df.iloc[0]

    best_eps = int(best["eps_km"])
    best_min_samples = int(best["min_samples"])
    print(f"\n  => Recommended: eps={best_eps} km, min_samples={best_min_samples}")
    print(f"     Clusters: {int(best['n_clusters'])}, Noise: {best['noise_pct']}%, "
          f"Silhouette: {best['silhouette_score']}")

    # --- Run best model ---
    best_labels = run_haversine_dbscan(coords_rad, best_eps, best_min_samples)
    df["dbscan_cluster"] = best_labels

    n_clusters_best = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    n_noise_best = (best_labels == -1).sum()

    # --- Visualisation ---
    fig = plt.figure(figsize=(20, 14))

    # (a) k-distance graph
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(k_dist_km)
    ax.set_xlabel("Sorted Points", fontsize=10)
    ax.set_ylabel(f"{k}-th Nearest Neighbour Distance (km)", fontsize=10)
    ax.set_title(f"k-Distance Graph (k={k}, Haversine)", fontsize=11, fontweight="bold")
    for eps_km in [100, 200, 300, 400, 500]:
        ax.axhline(y=eps_km, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.text(len(k_dist_km) * 0.98, eps_km, f" {eps_km} km",
                ha="right", va="bottom", fontsize=7, color="gray")
    ax.grid(alpha=0.3)

    # (b) Parameter heatmap — silhouette
    ax = fig.add_subplot(2, 3, 2)
    pivot = param_df.pivot_table(
        values="silhouette_score", index="eps_km", columns="min_samples", aggfunc="first"
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=0.6)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("min_samples", fontsize=10)
    ax.set_ylabel("eps (km)", fontsize=10)
    ax.set_title("Silhouette Score by Parameters", fontsize=11, fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            text = f"{val:.3f}" if not np.isnan(val) else "N/A"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="black" if (not np.isnan(val) and 0.2 < val < 0.5) else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (c) Best model: global cluster map
    ax = fig.add_subplot(2, 3, 3)
    unique_labels = sorted(set(best_labels))
    n_colors = max(1, len(unique_labels) - (1 if -1 in unique_labels else 0))
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_colors, 20)))

    # Noise first
    noise_mask = best_labels == -1
    ax.scatter(df.loc[noise_mask, "longitude"], df.loc[noise_mask, "latitude"],
               c="#cccccc", s=0.3, alpha=0.3, label=f"Noise ({n_noise_best})")

    color_idx = 0
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = best_labels == lbl
        cluster_pts = df.loc[mask]
        if len(cluster_pts) > 3000:
            cluster_pts = cluster_pts.sample(3000, random_state=42)
        ax.scatter(cluster_pts["longitude"], cluster_pts["latitude"],
                   s=1.5, alpha=0.5, color=colors[color_idx % len(colors)])
        color_idx += 1

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Latitude (°)", fontsize=10)
    ax.set_title(f"Best DBSCAN: eps={best_eps} km, min={best_min_samples}\n"
                 f"{n_clusters_best} clusters, {n_noise_best/len(df)*100:.1f}% noise",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")

    # (d) Noise percentage heatmap
    ax = fig.add_subplot(2, 3, 4)
    pivot_n = param_df.pivot_table(
        values="noise_pct", index="eps_km", columns="min_samples", aggfunc="first"
    )
    im2 = ax.imshow(pivot_n.values, aspect="auto", cmap="YlOrRd_r", vmin=0, vmax=50)
    ax.set_xticks(range(len(pivot_n.columns)))
    ax.set_xticklabels(pivot_n.columns)
    ax.set_yticks(range(len(pivot_n.index)))
    ax.set_yticklabels(pivot_n.index)
    ax.set_xlabel("min_samples", fontsize=10)
    ax.set_ylabel("eps (km)", fontsize=10)
    ax.set_title("Noise % by Parameters", fontsize=11, fontweight="bold")
    for i in range(len(pivot_n.index)):
        for j in range(len(pivot_n.columns)):
            val = pivot_n.values[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7)
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # (e) Max cluster % heatmap
    ax = fig.add_subplot(2, 3, 5)
    pivot_m = param_df.pivot_table(
        values="max_cluster_pct", index="eps_km", columns="min_samples", aggfunc="first"
    )
    im3 = ax.imshow(pivot_m.values, aspect="auto", cmap="YlOrRd_r", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot_m.columns)))
    ax.set_xticklabels(pivot_m.columns)
    ax.set_yticks(range(len(pivot_m.index)))
    ax.set_yticklabels(pivot_m.index)
    ax.set_xlabel("min_samples", fontsize=10)
    ax.set_ylabel("eps (km)", fontsize=10)
    ax.set_title("Max Cluster % by Parameters", fontsize=11, fontweight="bold")
    for i in range(len(pivot_m.index)):
        for j in range(len(pivot_m.columns)):
            val = pivot_m.values[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7)
    plt.colorbar(im3, ax=ax, shrink=0.8)

    # (f) Per-cluster bar chart
    ax = fig.add_subplot(2, 3, 6)
    cluster_df = cluster_stats(best_labels, df)
    top12 = cluster_df.head(12)
    colors_bar = plt.cm.viridis(np.linspace(0.15, 0.85, len(top12)))
    bars = ax.barh(
        [f"C{int(c['cluster_id'])} ({c['center_lon']:.0f}°,{c['center_lat']:.0f}°)" for _, c in top12.iterrows()],
        top12["count"],
        color=colors_bar, edgecolor="white"
    )
    ax.set_xlabel("Event Count", fontsize=10)
    ax.set_title("Top DBSCAN Clusters by Size", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    for bar, (_, c) in zip(bars, top12.iterrows()):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f"n={c['count']}  Mmax={c['max_mag']}",
                va="center", fontsize=6.5)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task6_dbscan_clustering.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  -> Saved: task6_dbscan_clustering.png")

    # Save per-cluster stats
    cluster_df.to_csv(OUTPUT_DIR / "task6_dbscan_clusters.csv", index=False, encoding="utf-8-sig")
    print("  -> Saved: task6_dbscan_clusters.csv")

    # --- Export labeled events CSV (each earthquake with its DBSCAN label) ---
    labeled_df = df[["time", "latitude", "longitude", "depth", "mag", "magType", "place", "id"]].copy()
    labeled_df["dbscan_cluster"] = best_labels
    labeled_df["dbscan_is_noise"] = (best_labels == -1)
    labeled_df.to_csv(OUTPUT_DIR / "task6_dbscan_labeled_events.csv", index=False, encoding="utf-8-sig")
    print("  -> Saved: task6_dbscan_labeled_events.csv")

    # Print cluster details
    print(f"\n  Top clusters (Haversine DBSCAN, eps={best_eps} km, min_samples={best_min_samples}):")
    for _, c in cluster_df.head(8).iterrows():
        print(f"    Cluster {int(c['cluster_id'])}: {int(c['count']):5d} events  "
              f"mean_mag={c['mean_mag']:.2f}  "
              f"center=({c['center_lon']:.0f}°, {c['center_lat']:.0f}°)  "
              f"depth={c['mean_depth']:.0f} km")

    print(f"\n  Methodological notes:")
    print(f"  - Uses Haversine (great-circle) distance, not Euclidean")
    print(f"  - eps={best_eps} km means points within ~{best_eps} km are neighbours")
    print(f"  - Correctly handles date-line and latitude distortion")
    print(f"  - This is a density-based description of spatial patterns only;")
    print(f"    cluster labels have no direct geophysical interpretation without")
    print(f"    plate-boundary / fault-mechanism context.")


    # ================================================================
    # 2. PCA Feature Analysis
    # ================================================================
    print("\n=== (2) PCA Feature Analysis ===")

    feature_cols = ["depth", "mag", "horizontalError", "depthError",
                    "magError", "rms", "gap", "dmin", "nst", "magNst"]

    # Drop rows with any NaN in these features
    pca_df = df[feature_cols].dropna()
    print(f"  Samples for PCA: {len(pca_df)} / {len(df)} "
          f"({len(df) - len(pca_df)} dropped due to NaN)")

    X = StandardScaler().fit_transform(pca_df)

    pca = PCA()
    X_pca = pca.fit_transform(X)

    # --- Explained variance ---
    evr = pca.explained_variance_ratio_
    cumsum = np.cumsum(evr)

    print(f"\n  Explained variance ratio:")
    for i, (v, c) in enumerate(zip(evr[:6], cumsum[:6])):
        print(f"    PC{i+1}: {v*100:5.1f}%  (cumulative: {c*100:5.1f}%)")

    # --- Biplot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # (a) Scree plot
    ax = axes[0]
    colors_bar = ["#4A90D9"] * 5 + ["#cccccc"] * (len(evr) - 5)
    ax.bar(range(1, len(evr)+1), evr * 100, color=colors_bar, edgecolor="white")
    ax.plot(range(1, len(evr)+1), cumsum * 100, "o-", color="#E0554A", linewidth=2, markersize=6)
    ax.set_xlabel("Principal Component", fontsize=11)
    ax.set_ylabel("Explained Variance (%)", fontsize=11)
    ax.set_title("PCA Scree Plot", fontsize=13, fontweight="bold")
    ax.set_xticks(range(1, len(evr)+1))
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=evr[0]*100, color="#4A90D9", linestyle="--", alpha=0.4)
    ax.annotate(f"PC1: {evr[0]*100:.1f}%", (1, evr[0]*100), textcoords="offset points",
                xytext=(20, 5), fontsize=10, color="#4A90D9")

    # (b) Biplot: PC1 vs PC2
    ax = axes[1]
    n_sample = min(5000, len(X_pca))
    idx_sample = np.random.RandomState(42).choice(len(X_pca), n_sample, replace=False)
    sc = ax.scatter(X_pca[idx_sample, 0], X_pca[idx_sample, 1],
                    c=pca_df["mag"].values[idx_sample], cmap="plasma",
                    s=3, alpha=0.5, norm=plt.Normalize(4.5, 9))

    # Loadings as arrows
    loadings = pca.components_.T
    scale_factor = X_pca[:, :2].std(axis=0) * 2.5
    for i, col in enumerate(feature_cols):
        ax.arrow(0, 0,
                 loadings[i, 0] * scale_factor[0],
                 loadings[i, 1] * scale_factor[1],
                 head_width=0.25, head_length=0.3, fc="#333333", ec="#333333", alpha=0.7, linewidth=1)
        ax.text(loadings[i, 0] * scale_factor[0] * 1.15,
                loadings[i, 1] * scale_factor[1] * 1.15,
                col, fontsize=8, color="#333333", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", fontsize=11)
    ax.set_title("PCA Biplot: PC1 vs PC2", fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.3)
    ax.grid(alpha=0.2)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Magnitude", fontsize=9)

    # --- Top contributing features ---
    print(f"\n  Top feature contributions:")
    for pc_idx in range(3):
        pc_loadings = pd.Series(abs(loadings[:, pc_idx]), index=feature_cols).sort_values(ascending=False)
        print(f"    PC{pc_idx+1}: " + " | ".join([f"{f}={v:.3f}" for f, v in pc_loadings.head(4).items()]))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "task6_pca_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("\n  -> Saved: task6_pca_analysis.png")

    # Save PCA loadings
    loadings_df = pd.DataFrame(
        loadings[:, :3],
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(3)]
    )
    loadings_df.to_csv(OUTPUT_DIR / "task6_pca_loadings.csv", encoding="utf-8-sig")
    print("  -> Saved: task6_pca_loadings.csv")


    # ================================================================
    # Summary
    # ================================================================
    print("\n说明：")
    print("本分析仅描述当前冻结目录中历史事件的空间聚类结构，")
    print("不能用于未来地震预测或灾害风险评估。")
    print("\n" + "=" * 60)
    print("Task 6 Complete!")
    print("=" * 60)
    print("Generated files:")
    print("  task6_dbscan_params.csv      - Full parameter comparison table")
    print("  task6_dbscan_clustering.png  - k-distance + heatmaps + cluster map + bar chart")
    print("  task6_dbscan_clusters.csv    - Per-cluster statistics")
    print("  task6_dbscan_labeled_events.csv  - All events with cluster labels")
    print("  task6_pca_analysis.png       - Scree plot + biplot")
    print("  task6_pca_loadings.csv       - Feature loadings per PC")


if __name__ == "__main__":
    main()
