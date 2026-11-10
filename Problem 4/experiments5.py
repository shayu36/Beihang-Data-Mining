"""任务4 深入：4 特征下精选集成 + IForest 调参。"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.stats import rankdata

from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.kde import KDE
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from sklearn.mixture import GaussianMixture

SEED = 42
FEATS = ["feature_2", "feature_3", "feature_4", "feature_6"]


def load():
    tr = pd.read_csv("data/train-set.csv")[FEATS].values.astype(np.float32)
    te = pd.read_csv("data/test-set.csv")
    return tr, te[FEATS].values.astype(np.float32), te["label"].values.astype(int)


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


def report(name, s, y):
    a, ap, f, p, r = m(s, y)
    print(f"{name:28s} AUROC={a:.4f} AUPRC={ap:.4f} F1={f:.4f}")
    return a, ap, f, s


def main():
    Xtr_raw, Xte_raw, y = load()
    sc_ = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = sc_.transform(Xtr_raw), sc_.transform(Xte_raw)

    # IForest 调参
    print("=== IForest 调参 ===")
    best_ap = 0
    for ne in [100, 300, 500, 1000]:
        for mf in [0.5, 1.0]:
            s = IForest(n_estimators=ne, max_features=mf, contamination=0.05,
                        random_state=SEED).fit(Xtr).decision_function(Xte)
            a, ap, f, *_ = m(s, y)
            print(f"  IForest(n={ne},maxf={mf}) AUROC={a:.4f} AUPRC={ap:.4f}")
            if ap > best_ap:
                best_ap = ap

    # 各方法分数（4 特征）
    scores = {}
    scores["IForest"] = IForest(n_estimators=500, contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)
    scores["HBOS"] = HBOS(contamination=0.05).fit(Xtr).decision_function(Xte)
    scores["MCD"] = MCD(contamination=0.05).fit(Xtr).decision_function(Xte)
    scores["CBLOF"] = CBLOF(n_clusters=8, contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)
    scores["PCA"] = PCA(contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)
    scores["kNN"] = KNN(n_neighbors=20, contamination=0.05).fit(Xtr).decision_function(Xte)
    scores["KDE"] = KDE(contamination=0.05).fit(Xtr).decision_function(Xte)
    scores["GMM"] = -GaussianMixture(n_components=4, random_state=SEED).fit(Xtr).score_samples(Xte)

    print("\n=== 精选集成（4 特征）===")
    order = ["IForest", "HBOS", "MCD", "CBLOF", "PCA", "kNN", "KDE", "GMM"]
    for k in [2, 3, 4, 5, 6, 8]:
        sub = order[:k]
        S = np.stack([rn(scores[n]) for n in sub], axis=1)
        report(f"秩平均(前{k}: {sub})", S.mean(axis=1), y)


if __name__ == "__main__":
    main()
