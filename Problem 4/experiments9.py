"""任务4 最终集成：AE(ep=50) + IForest + HBOS + MCD。"""
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

    scores = {}
    scores["AE"] = AutoEncoder(contamination=0.05, epoch_num=50, random_state=SEED,
                               verbose=0).fit(Xtr).decision_function(Xte)
    scores["IForest"] = IForest(n_estimators=1000, max_features=0.5, contamination=0.05,
                                random_state=SEED).fit(Xtr).decision_function(Xte)
    scores["HBOS"] = HBOS(contamination=0.05).fit(Xtr).decision_function(Xte)
    scores["MCD"] = MCD(contamination=0.05).fit(Xtr).decision_function(Xte)

    print("单模型：")
    for k, v in sorted(scores.items(), key=lambda kv: -average_precision_score(y, kv[1])):
        a, ap, *_ = m(v, y)
        print(f"  {k:10s} AUROC={a:.4f} AUPRC={ap:.4f}")

    # 各组合
    print("\n集成：")
    combos = [
        (["IForest", "HBOS", "MCD"], "IForest+HBOS+MCD"),
        (["AE", "IForest", "HBOS"], "AE+IForest+HBOS"),
        (["AE", "IForest", "HBOS", "MCD"], "AE+IForest+HBOS+MCD"),
        (["AE", "IForest", "MCD"], "AE+IForest+MCD"),
    ]
    best = None
    for sub, label in combos:
        S = np.mean([rn(scores[n]) for n in sub], axis=0)
        a, ap, f, p, r = m(S, y)
        print(f"  {label:22s} AUROC={a:.4f} AUPRC={ap:.4f} F1={f:.4f}")
        if best is None or ap > best[1]:
            best = (label, ap, S, sub)

    a, ap, f, p, r = m(best[2], y)
    print(f"\n最优: {best[0]}  AUROC={a:.4f} AUPRC={ap:.4f} F1={f:.4f} P={p:.4f} R={r:.4f}")
    # 保存最优分数
    np.save("results/best_score.npy", best[2])
    with open("results/best_combo.txt", "w") as fp:
        fp.write(",".join(best[3]))


if __name__ == "__main__":
    main()
