"""任务4 优化：集成融合 + 自编码器 + 关键超参调优。"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.kde import KDE
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.mcd import MCD
from pyod.models.lof import LOF


def load():
    tr = pd.read_csv("data/train-set.csv")
    te = pd.read_csv("data/test-set.csv")
    Xtr = tr.values.astype(np.float32)
    Xte = te.iloc[:, :-1].values.astype(np.float32)
    yte = te["label"].values.astype(int)
    return Xtr, Xte, yte


def metrics(scores, yte):
    auroc = roc_auc_score(yte, scores)
    auprc = average_precision_score(yte, scores)
    prec, rec, _ = precision_recall_curve(yte, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    b = int(np.argmax(f1))
    return auroc, auprc, f1[b], prec[b], rec[b]


def ranknorm(s):
    """把分数转成 [0,1] 的秩归一化值。"""
    from scipy.stats import rankdata
    r = rankdata(s)
    return (r - 1) / (len(r) - 1)


def get_scores(Xtr, Xte):
    out = {}
    for name, m in [
        ("CBLOF", CBLOF(contamination=0.05)),
        ("IForest", IForest(contamination=0.05, random_state=42)),
        ("HBOS", HBOS(contamination=0.05)),
        ("kNN", KNN(contamination=0.05)),
        ("KDE", KDE(contamination=0.05)),
        ("PCA", PCA(contamination=0.05, random_state=42)),
        ("OCSVM", OCSVM(contamination=0.05)),
        ("MCD", MCD(contamination=0.05)),
    ]:
        m.fit(Xtr)
        out[name] = m.decision_function(Xte)
    return out


class AE(nn.Module):
    def __init__(self, d=6):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 4), nn.Tanh(), nn.Linear(4, 2), nn.Tanh())
        self.dec = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def autoencoder(Xtr, Xte, yte, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt = torch.tensor(Xtr)
    model = AE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(300):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), Xt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        rec = model(torch.tensor(Xte)).numpy()
    rec_err = np.mean((Xte - rec) ** 2, axis=1)
    return rec_err


def main():
    Xtr_raw, Xte_raw, yte = load()
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

    sc = get_scores(Xtr, Xte)

    print(f"{'方法':24s} {'AUROC':>7s} {'AUPRC':>7s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s}")
    rows = []
    for name, s in sc.items():
        a, p, f, pr, rc = metrics(s, yte)
        rows.append((name, s, a, p, f, pr, rc))
        print(f"{name:24s} {a:7.4f} {p:7.4f} {f:7.4f} {pr:7.4f} {rc:7.4f}")

    # 自编码器
    s_ae = autoencoder(Xtr, Xte, yte)
    a, p, f, pr, rc = metrics(s_ae, yte)
    print(f"{'AutoEncoder':24s} {a:7.4f} {p:7.4f} {f:7.4f} {pr:7.4f} {rc:7.4f}")
    sc["AE"] = s_ae

    # ---- 集成 ----
    names = list(sc.keys())
    R = np.stack([ranknorm(sc[n]) for n in names], axis=1)

    for ens_name, ens_s in [
        ("平均(秩归一)", R.mean(axis=1)),
        ("最大(秩归一)", R.max(axis=1)),
        ("平均(前5)", np.mean([ranknorm(sc[n]) for n in
            ["CBLOF", "IForest", "HBOS", "kNN", "KDE"]], axis=0)),
    ]:
        a, p, f, pr, rc = metrics(ens_s, yte)
        print(f"{'集成 ' + ens_name:24s} {a:7.4f} {p:7.4f} {f:7.4f} {pr:7.4f} {rc:7.4f}")


if __name__ == "__main__":
    main()
