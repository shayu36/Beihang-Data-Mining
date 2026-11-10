"""评估指标、基线模型与可视化。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import RESULTS_DIR


def regression_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE(%)": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def persistence_baseline(y_train_last, y_test, horizon=1):
    """朴素基线：预测下一时刻温度不变，即 ŷ_{t+1} = OT(t)。

    返回 (ground_truth, prediction)，二者均在原始量纲、长度一致。
    """
    pred = np.concatenate([y_train_last[-horizon:], y_test[:-horizon]])
    return y_test, pred


def ridge_baseline(Xtr, ytr, Xte, yte, alpha=1.0):
    """线性回归基线：把滑动窗口展平后做岭回归。"""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha)
    model.fit(Xtr.reshape(len(Xtr), -1), ytr)
    pred = model.predict(Xte.reshape(len(Xte), -1))
    return yte, pred


def plot_loss(history, path=None):
    if path is None:
        path = os.path.join(RESULTS_DIR, "loss_curve.png")
    plt.figure(figsize=(6.5, 3.6))
    plt.plot(history["train"], label="Train loss")
    plt.plot(history["val"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_prediction(y_true, y_pred, title="Prediction vs ground truth",
                    n=240, path=None):
    if path is None:
        path = os.path.join(RESULTS_DIR, "prediction.png")
    plt.figure(figsize=(9, 3.8))
    plt.plot(y_true[:n], label="Ground truth", lw=1.4, color="#333")
    plt.plot(y_pred[:n], label="Prediction", lw=1.4, color="#d62728", alpha=0.85)
    plt.xlabel("Time step (10 min)")
    plt.ylabel("Outdoor temperature OT")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_scatter(y_true, y_pred, path=None):
    if path is None:
        path = os.path.join(RESULTS_DIR, "scatter.png")
    plt.figure(figsize=(4.6, 4.2))
    plt.scatter(y_true, y_pred, s=6, alpha=0.35, color="#1f77b4")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "--", color="#d62728", lw=1.2, label="y=x")
    plt.xlabel(r"Ground truth CO$_2$ concentration (ppm)")
    plt.ylabel(r"Predicted CO$_2$ concentration (ppm)")
    plt.title("Scatter plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
