"""任务3 最终：GRU 多种子集成 + Ridge 的加权组合（权重在验证集上选择）。"""
import numpy as np
import torch
from sklearn.linear_model import Ridge

from src.data_utils import load_and_clean, build_windows, split_chronological, scale_data
from src.models import RecurrentRegressor
from src.train import train_model
from src.evaluate import regression_metrics
from config import TARGET_COL, WINDOW_SIZE

df, fc = load_and_clean()
X, y = build_windows(df, fc, WINDOW_SIZE)
tr, va, te, *_ = split_chronological(len(y))
ti = fc.index(TARGET_COL)
Xtr, Xva, Xte, ytr, yva, yte, mu, sd = scale_data(
    X[tr], X[va], X[te], y[tr], y[va], y[te], ti)


def gru_ensemble(seeds=(0, 1, 2, 3, 4)):
    pv, pt = [], []
    for s in seeds:
        m = RecurrentRegressor(Xtr.shape[-1], cell="gru")
        m, _ = train_model(m, Xtr, ytr, Xva, yva, verbose=False, seed=s)
        m.eval()
        with torch.no_grad():
            pv.append(m(torch.tensor(Xva)).numpy())
            pt.append(m(torch.tensor(Xte)).numpy())
    return np.mean(pv, axis=0), np.mean(pt, axis=0)


Xtr_f, Xva_f, Xte_f = Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1), Xte.reshape(len(Xte), -1)

gv, gt = gru_ensemble()
ridge = Ridge(alpha=1.0).fit(Xtr_f, ytr)
rv = ridge.predict(Xva_f)
rt = ridge.predict(Xte_f)

# 在验证集上选择最优权重
best_w, best_val = 0.5, 1e9
for w in np.arange(0, 1.001, 0.05):
    pv = w * gv + (1 - w) * rv
    val_mse = float(np.mean((pv - yva) ** 2))
    if val_mse < best_val:
        best_val, best_w = val_mse, w

# 测试集最终
pt = best_w * gt + (1 - best_w) * rt
r = regression_metrics(yte * sd + mu, pt * sd + mu)
r_g = regression_metrics(yte * sd + mu, gt * sd + mu)
r_r = regression_metrics(yte * sd + mu, rt * sd + mu)

print(f"验证集最优权重 w(GRU)={best_w:.2f}")
print(f"GRU 集成     R2={r_g['R2']:.4f} RMSE={r_g['RMSE']:.4f} MAE={r_g['MAE']:.4f}")
print(f"Ridge        R2={r_r['R2']:.4f} RMSE={r_r['RMSE']:.4f}")
print(f"GRU+Ridge    R2={r['R2']:.4f} RMSE={r['RMSE']:.4f} MAE={r['MAE']:.4f}")
