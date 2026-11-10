"""任务 3：时间序列预测 —— 过去 2 小时预测下一时间点室外温度（OT）。

最终方案：GRU 多种子集成 + 岭回归的加权组合（权重在验证集上选择）。
运行： python main.py
"""
import json
import os

import numpy as np
import torch
from sklearn.linear_model import Ridge

from config import RESULTS_DIR, TARGET_COL, WINDOW_SIZE, GRU_SEEDS
from src.data_utils import (load_and_clean, build_windows, split_chronological,
                            scale_data)
from src.models import build_model
from src.train import train_model
from src.evaluate import (regression_metrics, persistence_baseline,
                          ridge_baseline, plot_loss, plot_prediction, plot_scatter)

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_baselines(y_train_raw, y_test_raw, Xtr, Xte):
    res = {}
    gt, pred = persistence_baseline(y_train_raw[-1:], y_test_raw)
    res["Persistence"] = regression_metrics(gt, pred)
    gt, pred = ridge_baseline(Xtr, y_train_raw, Xte, y_test_raw)
    res["Ridge"] = regression_metrics(gt, pred)
    return res


def run_gru_ensemble(Xtr, ytr, Xva, yva, Xte, seeds=GRU_SEEDS):
    """训练多个随机种子的 GRU，返回验证集与测试集的平均预测（标准化量纲）。"""
    pv, pt = [], []
    input_dim = Xtr.shape[-1]
    for s in seeds:
        m = build_model(input_dim, cell="gru")
        m, hist = train_model(m, Xtr, ytr, Xva, yva, verbose=False, seed=s)
        m.eval()
        with torch.no_grad():
            pv.append(m(torch.tensor(Xva)).numpy())
            pt.append(m(torch.tensor(Xte)).numpy())
    if seeds:  # 保存一个损失曲线用于可视化
        plot_loss(hist, os.path.join(RESULTS_DIR, "loss_gru.png"))
    return np.mean(pv, axis=0), np.mean(pt, axis=0)


def run_lstm(Xtr, ytr, Xva, yva, Xte, seed=42):
    m = build_model(Xtr.shape[-1], cell="lstm")
    m, _ = train_model(m, Xtr, ytr, Xva, yva, verbose=False, seed=seed)
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xte)).numpy()


def main():
    df, feature_cols = load_and_clean()
    target_idx = feature_cols.index(TARGET_COL)

    X, y = build_windows(df, feature_cols, WINDOW_SIZE)
    print(f"[data] 滑动窗口 {WINDOW_SIZE} 步 -> 输入 {X.shape}, 目标 {y.shape}")

    tr_idx, va_idx, te_idx, n_tr, n_va, n_te = split_chronological(len(y))
    print(f"[data] 划分 train/val/test = {n_tr}/{n_va}/{n_te}")

    y_train_raw, y_test_raw = y[tr_idx], y[te_idx]
    Xtr, Xva, Xte, ytr, yva, yte, mu, sd = scale_data(
        X[tr_idx], X[va_idx], X[te_idx], y[tr_idx], y[va_idx], y[te_idx],
        target_idx=target_idx)

    metrics = {}
    metrics.update(run_baselines(y_train_raw, y_test_raw, Xtr, Xte))

    # LSTM（单种子，对照）
    p_lstm = run_lstm(Xtr, ytr, Xva, yva, Xte) * sd + mu
    metrics["LSTM"] = regression_metrics(yte * sd + mu, p_lstm)

    # GRU 多种子集成
    gv, gt = run_gru_ensemble(Xtr, ytr, Xva, yva, Xte)
    metrics["GRU-ensemble"] = regression_metrics(yte * sd + mu, gt * sd + mu)

    # Ridge（标准化特征 + 标准化目标）
    Xtr_f, Xva_f, Xte_f = Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1), Xte.reshape(len(Xte), -1)
    ridge = Ridge(alpha=1.0).fit(Xtr_f, ytr)
    rv, rt = ridge.predict(Xva_f), ridge.predict(Xte_f)

    # 在验证集上选择 GRU 与 Ridge 的最优加权系数
    best_w, best_val = 0.5, 1e9
    for w in np.arange(0.0, 1.001, 0.05):
        val_mse = float(np.mean((w * gv + (1 - w) * rv - yva) ** 2))
        if val_mse < best_val:
            best_val, best_w = val_mse, w

    # 最终模型：加权组合
    final = best_w * gt + (1 - best_w) * rt
    metrics["Final(GRU+Ridge)"] = regression_metrics(yte * sd + mu, final * sd + mu)
    print(f"[model] 验证集最优权重 w(GRU)={best_w:.2f}")

    print("\n=== 测试集评估结果（原始量纲）===")
    for name, m in metrics.items():
        print(f"  {name:16s} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
              f"R2={m['R2']:.4f}  MAPE={m['MAPE(%)']:.2f}%")

    # 可视化（最终模型预测）
    y_true_raw = yte * sd + mu
    final_raw = final * sd + mu
    plot_prediction(y_true_raw, final_raw, n=240,
                    path=os.path.join(RESULTS_DIR, "prediction.png"))
    plot_scatter(y_true_raw, final_raw, path=os.path.join(RESULTS_DIR, "scatter.png"))

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({**metrics, "best_weight": best_w}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {RESULTS_DIR}")


if __name__ == "__main__":
    main()
