"""任务4 收尾：特征选择 + 精选集成 + 自编码器融合。"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.stats import rankdata

from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.kde import KDE
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.mcd import MCD
from sklearn.mixture import GaussianMixture

SEED = 42


def load(feats=None):
    tr = pd.read_csv("data/train-set.csv")
    te = pd.read_csv("data/test-set.csv")
    cols = [c for c in tr.columns if feats is None or c in feats]
    return (tr[cols].values.astype(np.float32),
            te[cols].values.astype(np.float32), te["label"].values.astype(int))


def m(s, y):
    a = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    p, r, _ = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    b = int(np.argmax(f))
    return a, ap, f[b], p[b], r[b]


def rn(s):
    r = rankdata(s)
    return (r - 1) / (len(r) - 1)


class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, max(2, d // 2)), nn.Tanh(),
                                 nn.Linear(max(2, d // 2), 2), nn.Tanh())
        self.dec = nn.Sequential(nn.Linear(2, max(2, d // 2)), nn.Tanh(),
                                 nn.Linear(max(2, d // 2), d))

    def forward(self, x):
        return self.dec(self.enc(x))


def autoencoder(Xtr, Xte, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt = torch.tensor(Xtr)
    model = AE(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(300):
        opt.zero_grad()
        loss = nn.MSELoss()(model(Xt), Xt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        rec = model(torch.tensor(Xte)).numpy()
    return np.mean((Xte - rec) ** 2, axis=1)


def all_scores(Xtr, Xte):
    sc = {}
    sc["CBLOF"] = CBLOF(n_clusters=8, contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)
    sc["IForest"] = IForest(n_estimators=300, contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)
    sc["HBOS"] = HBOS(contamination=0.05).fit(Xtr).decision_function(Xte)
    sc["kNN"] = KNN(n_neighbors=20, contamination=0.05).fit(Xtr).decision_function(Xte)
    sc["KDE"] = KDE(contamination=0.05).fit(Xtr).decision_function(Xte)
    sc["PCA"] = PCA(contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)
    sc["OCSVM"] = OCSVM(nu=0.05).fit(Xtr).decision_function(Xte)
    sc["MCD"] = MCD(contamination=0.05).fit(Xtr).decision_function(Xte)
    sc["GMM"] = -GaussianMixture(n_components=4, random_state=SEED).fit(Xtr).score_samples(Xte)
    sc["AE"] = autoencoder(Xtr, Xte)
    return sc


def main():
    # 特征选择：全 6 维 vs 去掉噪声特征 1、5 的 4 维
    for label, feats in [("6维全特征", None), ("4维(去f1,f5)", None)]:
        if feats is None and label.startswith("4"):
            feats = ["feature_2", "feature_3", "feature_4", "feature_6"]
        Xtr, Xte, y = load(feats)
        sc_ = StandardScaler().fit(Xtr)
        Xtr, Xte = sc_.transform(Xtr), sc_.transform(Xte)
        sc = all_scores(Xtr, Xte)
        S = np.stack([rn(s) for s in sc.values()], axis=1)
        ens = S.mean(axis=1)
        a, ap, f, p, r = m(ens, y)
        print(f"[{label}] 精选集成(10方法) AUROC={a:.4f} AUPRC={ap:.4f} F1={f:.4f}")
        # 各单模型 AUPRC
        for k, v in sorted(sc.items(), key=lambda kv: -average_precision_score(y, kv[1])):
            print(f"    {k:10s} AUROC={roc_auc_score(y, v):.4f} AUPRC={average_precision_score(y, v):.4f}")


if __name__ == "__main__":
    main()
