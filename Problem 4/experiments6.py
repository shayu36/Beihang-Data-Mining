"""任务4 最终：4 特征 + 调优 IForest 的精选集成。"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.stats import rankdata

from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.cblof import CBLOF
from pyod.models.pca import PCA

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


def main():
    Xtr_raw, Xte_raw, y = load()
    sc_ = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = sc_.transform(Xtr_raw), sc_.transform(Xte_raw)

    scores = {}
    scores["IForest"] = IForest(n_estimators=1000, max_features=0.5,
                                contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)
    scores["HBOS"] = HBOS(contamination=0.05).fit(Xtr).decision_function(Xte)
    scores["MCD"] = MCD(contamination=0.05).fit(Xtr).decision_function(Xte)
    scores["CBLOF"] = CBLOF(n_clusters=8, contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)
    scores["PCA"] = PCA(contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)

    print(f"{'方法':30s} {'AUROC':>7s} {'AUPRC':>7s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    for k, v in scores.items():
        a, ap, f, p, r = m(v, y)
        print(f"{k:30s} {a:7.4f} {ap:7.4f} {f:7.4f} {p:7.4f} {r:7.4f}")

    order = list(scores.keys())
    print()
    for k in [2, 3, 4, 5]:
        S = np.stack([rn(scores[n]) for n in order[:k]], axis=1)
        a, ap, f, p, r = m(S.mean(axis=1), y)
        print(f"秩平均集成(前{k}: {order[:k]})  AUROC={a:.4f} AUPRC={ap:.4f} F1={f:.4f}")


if __name__ == "__main__":
    main()
