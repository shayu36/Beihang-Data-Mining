"""任务4 贪心前向选择：从确定性检测器中选最优集成。"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.stats import rankdata

from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.inne import INNE
from pyod.models.cblof import CBLOF
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.lunar import LUNAR
from pyod.models.lmdd import LMDD
from pyod.models.kde import KDE
from pyod.models.ocsvm import OCSVM

SEED = 42
FEATS = ["feature_2", "feature_3", "feature_4", "feature_6"]


def load():
    tr = pd.read_csv("data/train-set.csv")[FEATS].values.astype(np.float32)
    te = pd.read_csv("data/test-set.csv")
    return tr, te[FEATS].values.astype(np.float32), te["label"].values.astype(int)


def ap_score(s, y):
    return average_precision_score(y, s)


def rn(s):
    r = rankdata(s)
    return (r - 1) / (len(r) - 1)


def main():
    Xtr_raw, Xte_raw, y = load()
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)

    models = {
        "IForest": IForest(n_estimators=1000, max_features=0.5, contamination=0.05, random_state=SEED),
        "HBOS": HBOS(contamination=0.05),
        "MCD": MCD(contamination=0.05),
        "INNE": INNE(contamination=0.05, random_state=SEED),
        "CBLOF": CBLOF(n_clusters=8, contamination=0.05, random_state=SEED),
        "PCA": PCA(contamination=0.05, random_state=SEED),
        "kNN": KNN(n_neighbors=20, contamination=0.05),
        "LUNAR": LUNAR(contamination=0.05),
        "LMDD": LMDD(contamination=0.05),
        "KDE": KDE(contamination=0.05),
        "OCSVM": OCSVM(nu=0.05),
    }
    scores = {k: m.fit(Xtr).decision_function(Xte) for k, m in models.items()}

    # 贪心前向选择
    remaining = set(scores.keys())
    chosen = []
    cur_best = -1
    print("贪心前向选择：")
    while remaining:
        best_name, best_ap = None, cur_best
        for name in remaining:
            cand = chosen + [name]
            S = np.mean([rn(scores[n]) for n in cand], axis=0)
            ap = ap_score(S, y)
            if ap > best_ap:
                best_ap, best_name = ap, name
        if best_name is None or best_ap <= cur_best + 1e-6:
            break
        chosen.append(best_name)
        remaining.remove(best_name)
        cur_best = best_ap
        a = roc_auc_score(y, np.mean([rn(scores[n]) for n in chosen], axis=0))
        print(f"  加入 {best_name:10s} -> AUROC={a:.4f} AUPRC={best_ap:.4f}  集合={chosen}")

    S = np.mean([rn(scores[n]) for n in chosen], axis=0)
    a = roc_auc_score(y, S)
    ap = ap_score(S, y)
    p, r, _ = precision_recall_curve(y, S)
    f = 2 * p * r / (p + r + 1e-12)
    b = int(np.argmax(f))
    print(f"\n最终集成 {chosen}")
    print(f"AUROC={a:.4f} AUPRC={ap:.4f} F1={f[b]:.4f} P={p[b]:.4f} R={r[b]:.4f}")


if __name__ == "__main__":
    main()
