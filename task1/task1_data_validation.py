#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1: Data Reading and Quality Validation
============================================
Read USGS earthquake catalog CSV, validate data, add derived columns,
and export earthquakes_prepared.csv.

Usage:
    python task1_data_validation.py
    python task1_data_validation.py --input path/to/data.csv --output-dir ./outputs
"""

import argparse
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import warnings

warnings.filterwarnings("ignore")

def setup_windows_console():
    """Re-wrap stdout/stderr as UTF-8 (called from main() only — wrapping at
    import time breaks pytest's capture machinery)."""
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Task 1: Data Reading and Quality Validation")
    parser.add_argument(
        "--input", type=Path,
        default=BASE_DIR.parent / "USGS_2024_2025_M4.5plus_earthquakes.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE_DIR,
        help="Directory for output files",
    )
    return parser.parse_args()


EXPECTED_START = pd.Timestamp("2024-01-01T00:00:00.000Z")
EXPECTED_END   = pd.Timestamp("2025-12-31T23:59:59.999Z")


def log(msg: str, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest().upper()


# ============================================================
# 1. 读取 CSV
# ============================================================
def read_and_prepare(path: Path):
    """读取 CSV，按 time 和 id 排序，添加派生字段。"""
    df = pd.read_csv(path)
    df["time"]    = pd.to_datetime(df["time"],    utc=True)
    df["updated"] = pd.to_datetime(df["updated"], utc=True)

    # 按要求：按 time 升序、再按 id 升序
    df = df.sort_values(["time", "id"], ascending=[True, True], kind="mergesort")
    df = df.reset_index(drop=True)

    # 派生字段
    df["year"]  = df["time"].dt.year
    df["month"] = df["time"].dt.strftime("%Y-%m")

    df["mag_group"] = pd.cut(
        df["mag"],
        bins=[4.5, 5.0, 6.0, 7.0, np.inf],
        labels=["[4.5,5.0)", "[5.0,6.0)", "[6.0,7.0)", "[7.0,+inf)"],
        right=False,
    )

    df["depth_group"] = pd.cut(
        df["depth"],
        bins=[0, 70, 300, np.inf],
        labels=["[0,70)", "[70,300)", "[300,+inf)"],
        right=False,
    )

    return df


# ============================================================
# 2. 数据范围验证
# ============================================================
def validate_ranges(df, f):
    log("\n" + "=" * 60, f)
    log("2. 数据范围验证", f)
    log("=" * 60, f)
    issues = []

    t_min, t_max = df["time"].min(), df["time"].max()
    log(f"  时间范围: {t_min}  ~  {t_max}", f)
    t_ok = True
    if t_min < EXPECTED_START:
        msg = f"  ⚠ 最小时间 {t_min} 早于预期起始 {EXPECTED_START}"
        issues.append(msg); log(msg, f); t_ok = False
    if t_max > EXPECTED_END:
        msg = f"  ⚠ 最大时间 {t_max} 晚于预期结束 {EXPECTED_END}"
        issues.append(msg); log(msg, f); t_ok = False
    if t_ok:
        log("  ✓ 时间范围符合要求 (2024-01-01 ~ 2025-12-31)", f)

    lat_ok = df["latitude"].between(-90, 90).all()
    if not lat_ok:
        bad = df[~df["latitude"].between(-90, 90)]
        msg = f"  ⚠ 纬度越界: {len(bad)} 条记录"
        issues.append(msg); log(msg, f)
    else:
        log("  ✓ 纬度全部在 [-90°, 90°] 范围内", f)

    lon_ok = df["longitude"].between(-180, 180).all()
    if not lon_ok:
        bad = df[~df["longitude"].between(-180, 180)]
        msg = f"  ⚠ 经度越界: {len(bad)} 条记录"
        issues.append(msg); log(msg, f)
    else:
        log("  ✓ 经度全部在 [-180°, 180°] 范围内", f)

    depth_ok = (df["depth"] >= 0).all()
    if not depth_ok:
        bad = df[df["depth"] < 0]
        msg = f"  ⚠ 深度为负: {len(bad)} 条记录"
        issues.append(msg); log(msg, f)
    else:
        log("  ✓ 深度全部 >= 0 km", f)

    mag_ok = (df["mag"] >= 4.5).all()
    if not mag_ok:
        bad = df[df["mag"] < 4.5]
        msg = f"  ⚠ 震级 < 4.5: {len(bad)} 条记录"
        issues.append(msg); log(msg, f)
    else:
        log("  ✓ 震级全部 >= 4.5", f)

    if (df["type"] == "earthquake").all():
        log("  ✓ type 字段全部为 'earthquake'", f)
    else:
        other = df[df["type"] != "earthquake"]
        msg = f"  ⚠ type 字段存在非 earthquake 值: {len(other)} 条"
        issues.append(msg); log(msg, f)

    if (df["status"] == "reviewed").all():
        log("  ✓ status 字段全部为 'reviewed'", f)
    else:
        other = df[df["status"] != "reviewed"]
        msg = f"  ⚠ status 字段存在非 reviewed 值: {len(other)} 条"
        issues.append(msg); log(msg, f)

    return issues


# ============================================================
# 3. 记录唯一性检查
# ============================================================
def validate_uniqueness(df, f):
    log("\n" + "=" * 60, f)
    log("3. 记录唯一性检查", f)
    log("=" * 60, f)
    issues = []

    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        msg = f"  ⚠ 存在 {dup_rows} 条完全重复记录"
        issues.append(msg); log(msg, f)
    else:
        log(f"  ✓ 无完全重复记录 (共 {len(df)} 条)", f)

    dup_ids = df["id"].duplicated().sum()
    if dup_ids > 0:
        dup_id_list = df[df["id"].duplicated(keep=False)]["id"].unique()
        msg = f"  ⚠ 存在 {dup_ids} 条重复 ID ({len(dup_id_list)} 个 ID 重复)"
        issues.append(msg); log(msg, f)
        log(f"     重复 ID: {list(dup_id_list[:5])}", f)
    else:
        log(f"  ✓ {len(df)} 个 ID 全部唯一", f)

    core_cols = ["time", "latitude", "longitude", "depth", "mag"]
    dup_core = df.duplicated(subset=core_cols).sum()
    if dup_core > 0:
        msg = f"  ⚠ 核心字段 (time+lat+lon+depth+mag) 存在 {dup_core} 条重复"
        issues.append(msg); log(msg, f)
    else:
        log("  ✓ 核心字段组合无重复", f)

    return issues


# ============================================================
# 4. 数值边界检查
# ============================================================
def validate_numerical_bounds(df, f):
    log("\n" + "=" * 60, f)
    log("4. 数值边界检查（详细描述性统计）", f)
    log("=" * 60, f)

    num_cols = ["latitude", "longitude", "depth", "mag",
                "horizontalError", "depthError", "magError",
                "nst", "gap", "dmin", "rms", "magNst"]

    stats = df[num_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    stats["missing"] = df[num_cols].isna().sum()
    stats["missing_pct"] = (df[num_cols].isna().sum() / len(df) * 100).round(2)
    stats["count"] = stats["count"].astype(int)

    for col in num_cols:
        s = df[col]
        log(f"\n  [{col}]", f)
        log(f"    非空值数: {s.notna().sum()}  /  缺失: {s.isna().sum()} ({s.isna().sum()/len(df)*100:.2f}%)", f)
        log(f"    最小值:   {s.min():.4f}", f)
        log(f"    1% 分位:  {s.quantile(0.01):.4f}", f)
        log(f"    中位数:   {s.median():.4f}", f)
        log(f"    99% 分位: {s.quantile(0.99):.4f}", f)
        log(f"    最大值:   {s.max():.4f}", f)
        log(f"    均值:     {s.mean():.4f}", f)
        log(f"    标准差:   {s.std():.4f}", f)

    issues = []
    for col in ["depth", "mag"]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                msg = f"  ⚠ {col} 存在 {neg} 条负值"
                issues.append(msg); log(msg, f)

    return issues, stats


# ============================================================
# 5. 质量字段缺失检查
# ============================================================
def validate_quality_missing(df, f):
    log("\n" + "=" * 60, f)
    log("5. 质量字段缺失检查", f)
    log("=" * 60, f)

    quality_cols = ["horizontalError", "depthError", "magError", "rms", "gap",
                    "nst", "dmin", "magNst", "magType"]

    missing_summary = {}
    warnings_list = []
    for col in quality_cols:
        miss = df[col].isna().sum()
        pct  = miss / len(df) * 100
        missing_summary[col] = {"missing": miss, "pct": round(pct, 2)}
        status = "⚠" if miss > 0 else "✓"
        log(f"  {status} {col:20s}: 缺失 {miss:5d} / {len(df)} ({pct:.2f}%)", f)
        if miss > 0:
            warnings_list.append(f"质量字段 {col} 缺失 {miss} 条 ({pct:.2f}%)")

    log("\n  --- 关键说明 ---", f)
    log("  质量字段缺失值未用 0 填补，也未因缺失而删除整条记录。", f)
    log("  统计计算使用各字段的有效样本（逐字段 pairwise）。", f)

    return missing_summary, warnings_list


# ============================================================
# 6. 导出结果
# ============================================================
def export_results(df, stats, missing_summary, output_dir):
    # (a) 处理后数据（含派生字段）→ earthquakes_prepared.csv
    output_csv = output_dir / "earthquakes_prepared.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n✓ 处理后数据已导出: {output_csv}")

    # (b) 缺失统计 → validation_summary.csv
    missing_df = pd.DataFrame(missing_summary).T
    missing_df.index.name = "field"
    missing_csv = output_dir / "validation_summary.csv"
    missing_df.to_csv(missing_csv, encoding="utf-8-sig")
    print(f"✓ 缺失统计已导出: {missing_csv}")

    # (c) 数值统计 CSV
    stats_csv = output_dir / "task1_numerical_stats.csv"
    stats.to_csv(stats_csv, encoding="utf-8-sig")
    print(f"✓ 数值统计已导出: {stats_csv}")


# ============================================================
# 主流程
# ============================================================
def main():
    setup_windows_console()
    args = parse_args()
    input_csv = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_log = output_dir / "validation_report.txt"

    if not input_csv.exists():
        print(f"错误: 找不到输入文件 {input_csv}")
        sys.exit(1)

    print("=" * 60)
    print("任务 1：数据读取与质量验证")
    print("USGS 2024-2025 M4.5+ Earthquakes")
    print("=" * 60)

    file_sha256 = calculate_sha256(input_csv)

    with open(validation_log, "w", encoding="utf-8") as f:
        log(f"验证时间: {pd.Timestamp.now()}", f)
        log(f"输入文件: {input_csv}", f)
        log(f"SHA-256: {file_sha256}", f)

        log("\n" + "=" * 60, f)
        log("1. 读取 CSV", f)
        log("=" * 60, f)
        log(f"  文件路径: {input_csv}", f)

        df = read_and_prepare(input_csv)
        log(f"  记录总数: {len(df)}", f)
        log(f"  字段数量: {len(df.columns)}", f)
        log(f"  新增派生字段: year, month, mag_group, depth_group", f)
        log(f"  mag_group 分组: [4.5,5.0) [5.0,6.0) [6.0,7.0) [7.0,+inf)", f)
        log(f"  depth_group 分组: [0,70) [70,300) [300,+inf)", f)
        log(f"  time 类型: {df['time'].dtype}", f)
        log(f"  updated 类型: {df['updated'].dtype}", f)
        log("  ✓ 已按 time, id 排序；time 与 updated 正确区分", f)

        issues_2 = validate_ranges(df, f)
        issues_3 = validate_uniqueness(df, f)
        issues_4, stats = validate_numerical_bounds(df, f)
        missing_summary, quality_warnings = validate_quality_missing(df, f)

        core_issues = issues_2 + issues_3 + issues_4
        log("\n" + "=" * 60, f)
        log("总  结", f)
        log("=" * 60, f)
        if core_issues:
            log(f"  ⚠ 核心验证共发现 {len(core_issues)} 个问题:", f)
            for i, iss in enumerate(core_issues, 1):
                log(f"    {i}. {iss}", f)
        else:
            log("  ✓ 核心筛选与边界验证全部通过！", f)

        if quality_warnings:
            log(f"\n  ⚠ 质量字段存在 {len(quality_warnings)} 项缺失（不影响数据使用）:", f)
            for i, w in enumerate(quality_warnings, 1):
                log(f"    {i}. {w}", f)
        else:
            log("  ✓ 质量字段无缺失", f)

        log(f"\n  最终记录数: {len(df)}", f)
        log(f"  字段数: {len(df.columns)}", f)

    export_results(df, stats, missing_summary, output_dir)

    print("\n" + "=" * 60)
    print("任务 1 完成！")
    print(f"验证日志: {validation_log}")
    print(f"处理后数据: {output_dir / 'earthquakes_prepared.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
