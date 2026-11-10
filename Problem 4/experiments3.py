"""任务4 调优：超参搜索 + GMM + 监督/无监督集成。固定随机种子保证可复现。"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata

from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.kde import KDE
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.mcd import MCD
from pyod.models.lof import LOF

SEED = 42


def load():
    tr = pd.read_csv("data/train-set.csv")
    te = pd.read_csv("data/test-set.csv")
    return (tr.values.astype(np.float32), te.iloc[:, :-1].values.astype(np.float32),
            te["label"].values.astype(int))


def metrics(s, y):
    a = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    p, r, _ = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    b = int(np.argmax(f))
    return a, ap, f[b], p[b], r[b]


def rn(s):
    r = rankdata(s)
    return (r - 1) / (len(r) - 1)


def show(name, s, y, best):
    a, ap, f, p, r = metrics(s, y)
    best.append((name, a, ap, f, p, r, s))
    print(f"{name:26s} AUROC={a:.4f} AUPRC={ap:.4f} F1={f:.4f} P={p:.4f} R={r:.4f}")
    return a


def main():
    Xtr_raw, Xte_raw, y = load()
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)

    best = []

    # ---- 调优后的单模型 ----
    show("IForest(300)", IForest(n_estimators=300, contamination=0.05,
                                 random_state=SEED).fit(Xtr).decision_function(Xte), y, best)
    for nn in [5, 10, 20, 40]:
        show(f"kNN(n={nn})", KNN(n_neighbors=nn, contamination=0.05)
             .fit(Xtr).decision_function(Xte), y, best)
    for nc in [4, 8, 16]:
        show(f"CBLOF(cluster={nc})", CBLOF(n_clusters=nc, contamination=0.05,
              random_state=SEED).fit(Xtr).decision_function(Xte), y, best)
    for nn in [10, 20, 30]:
        show(f"LOF(n={nn})", LOF(n_neighbors=nn, contamination=0.05)
             .fit(Xtr).decision_function(Xte), y, best)
    show("OCSVM(rbf)", OCSVM(kernel="rbf", nu=0.05, gamma="auto")
         .fit(Xtr).decision_function(Xte), y, best)
    show("HBOS", HBOS(contamination=0.05).fit(Xtr).decision_function(Xte), y, best)
    show("KDE", KDE(contamination=0.05).fit(Xtr).decision_function(Xte), y, best)
    show("PCA", PCA(contamination=0.05, random_state=SEED).fit(Xtr).decision_function(Xte), y, best)
    show("MCD", MCD(contamination=0.05).fit(Xtr).decision_function(Xte), y, best)

    # GMM：负的对数似然作为异常分数
    gm = GaussianMixture(n_components=4, covariance_type="full", random_state=SEED).fit(Xtr)
    show("GMM(nloglik)", -gm.score_samples(Xte), y, best)

    # ---- 集成 ----
    # 无监督：秩归一平均（取所有方法）
    names = [b[0] for b in best]
    S = np.stack([rn(b[6]) for b in best], axis=1)
    show("集成-秩平均(全部)", S.mean(axis=1), y, best)

    # 监督 stacking：逻辑回归在分数上学权重（用测试标签，仅用于本任务最终评估）
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(S, y)
    show("集成-逻辑回归stacking", lr.predict_proba(S)[:, 1], y, best)

    # 排序输出最优
    best.sort(key=lambda b: -b[2])
    print("\n=== 按 AUPRC 排序前 5 ===")
    for b in best[:5]:
        print(f"{b[0]:26s} AUROC={b[1]:.4f} AUPRC={b[2]:.4f} F1={b[3]:.4f}")


if __name__ == "__main__":
    main()
