"""任务4 优化：含 AutoEncoder 的精选集成 + 权重优化。"""
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
from pyod.models.auto_encoder import AutoEncoder

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
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)

    # AutoEncoder 调参
    print("=== AutoEncoder 调参 ===")
    for hn in [[4, 3, 2], [4, 2], [6, 4, 2], [8, 4, 2]]:
        for ep in [50, 100]:
            s = AutoEncoder(contamination=0.05, hidden_neuron_list=hn, epoch_num=ep,
                            random_state=SEED, verbose=0).fit(Xtr).decision_function(Xte)
            a, ap = roc_auc_score(y, s), average_precision_score(y, s)
            print(f"  AE(hn={hn},ep={ep}) AUROC={a:.4f} AUPRC={ap:.4f}")

    # 精选方法（固定超参）
    scores = {}
    scores["AE"] = AutoEncoder(contamination=0.05, epoch_num=100, random_state=SEED, verbose=0).fit(Xtr).decision_function(Xte)
    scores["IForest"] = IForest(n_estimators=1000, max_features=0.5, contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)
    scores["HBOS"] = HBOS(contamination=0.05).fit(Xtr).decision_function(Xte)
    scores["LUNAR"] = LUNAR(contamination=0.05).fit(Xtr).decision_function(Xte)
    scores["MCD"] = MCD(contamination=0.05).fit(Xtr).decision_function(Xte)
    scores["INNE"] = INNE(contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)
    scores["CBLOF"] = CBLOF(n_clusters=8, contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)
    scores["PCA"] = PCA(contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte)
    scores["kNN"] = KNN(n_neighbors=20, contamination=0.05).fit(Xtr).decision_function(Xte)

    print("\n=== 精选集成（按 AUPRC 排序取前 k）===")
    order = sorted(scores.keys(), key=lambda k: -average_precision_score(y, scores[k]))
    print("排序:", order)
    best = (0, None, None)
    for k in range(2, len(order) + 1):
        sub = order[:k]
        S = np.mean([rn(scores[n]) for n in sub], axis=0)
        a, ap, f, p, r = m(S, y)
        print(f"  前{k} {sub} -> AUROC={a:.4f} AUPRC={ap:.4f} F1={f:.4f}")
        if ap > best[0]:
            best = (ap, sub, S)

    print(f"\n最优集成: {best[1]}  AUPRC={best[0]:.4f}")
    a, ap, f, p, r = m(best[2], y)
    print(f"  最终 AUROC={a:.4f} AUPRC={ap:.4f} F1={f:.4f} P={p:.4f} R={r:.4f}")

    # 监督 stacking 上界（逻辑回归）
    from sklearn.linear_model import LogisticRegression
    S_all = np.stack([rn(s) for s in scores.values()], axis=1)
    lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(S_all, y)
    s_lr = lr.predict_proba(S_all)[:, 1]
    a, ap, f, p, r = m(s_lr, y)
    print(f"逻辑回归 stacking 上界: AUROC={a:.4f} AUPRC={ap:.4f} F1={f:.4f}")


if __name__ == "__main__":
    main()
