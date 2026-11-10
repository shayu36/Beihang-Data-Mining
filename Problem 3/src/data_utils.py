"""数据加载、清洗、滑动窗口切分与标准化。"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (DATA_PATH, TARGET_COL, WINDOW_SIZE, SENTINEL,
                    TEST_RATIO, VAL_RATIO, SEED)


def load_and_clean(path=DATA_PATH):
    """读取 GBK 编码的 CSV，并把 -9999 这类传感器异常值做时序线性插值。"""
    df = pd.read_csv(path, encoding="gbk")
    df = df.sort_values("date").reset_index(drop=True)

    # 仅对数值特征处理异常占位值：把 -9999 视为缺失，做时间维度的线性插值
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_bad_before = int((df[num_cols] <= SENTINEL + 1).sum().sum())
    for c in num_cols:
        df.loc[df[c] <= SENTINEL + 1, c] = np.nan
    df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
    n_nan_after = int(df[num_cols].isna().sum().sum())

    # 21 维特征全部参与建模（包含待预测的 OT 的历史值，即自回归特征）
    feature_cols = num_cols
    print(f"[data] 共 {len(df)} 条记录，特征维度 {len(feature_cols)}，"
          f"清洗异常占位值 {n_bad_before} 处，插值后剩余缺失 {n_nan_after} 处")
    return df, feature_cols


def build_windows(df, feature_cols, window=WINDOW_SIZE, target_col=TARGET_COL):
    """滑动窗口切分：用过去 window 个时间点的全部特征预测下一时间点的 OT。"""
    X = df[feature_cols].values.astype(np.float32)   # 21 维特征（含 OT 历史）
    y = df[target_col].values.astype(np.float32)

    Xs, ys = [], []
    for i in range(window, len(df)):
        Xs.append(X[i - window:i])                   # (window, n_features)
        ys.append(y[i])                              # 下一时间点的 OT
    return np.stack(Xs), np.stack(ys)


def split_chronological(n, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=SEED):
    """按时间顺序划分训练/验证/测试，保证时序不泄漏。"""
    n_test = int(n * test_ratio)
    n_rest = n - n_test
    n_val = int(n_rest * val_ratio)
    n_train = n_rest - n_val
    idx = np.arange(n)
    return (idx[:n_train], idx[n_train:n_train + n_val],
            idx[n_train + n_val:], n_train, n_val, n_test)


def scale_data(X_train, X_val, X_test, y_train, y_val, y_test, target_idx=-1):
    """仅在训练集上拟合 StandardScaler，同时标准化特征与目标 OT。

    返回标准化后的 X/y 以及目标列的均值、标准差（用于还原原始量纲）。
    """
    n, T, F = X_train.shape
    scaler = StandardScaler().fit(X_train.reshape(-1, F))
    Xtr = scaler.transform(X_train.reshape(-1, F)).reshape(n, T, F)
    Xva = scaler.transform(X_val.reshape(-1, F)).reshape(-1, T, F)
    Xte = scaler.transform(X_test.reshape(-1, F)).reshape(-1, T, F)

    mu = scaler.mean_[target_idx]
    sd = scaler.scale_[target_idx]
    ytr = ((y_train - mu) / sd).astype(np.float32)
    yva = ((y_val - mu) / sd).astype(np.float32)
    yte = ((y_test - mu) / sd).astype(np.float32)
    return Xtr, Xva, Xte, ytr, yva, yte, mu, sd


class WindowDataset:
    """把 (N, T, F) 的数组包装成 PyTorch 数据集。"""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
