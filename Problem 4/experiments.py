"""任务4：无监督甲状腺疾病判断 —— 多算法对比。

在仅含正常样本的训练集上学习，在含标签的测试集上评估异常检测效果。
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet, EmpiricalCovariance

from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from pyod.models.cblof import CBLOF
from pyod.models.mcd import MCD
from pyod.models.kde import KDE
from pyod.models.sos import SOS


def load():
    tr = pd.read_csv("data/train-set.csv")
    te = pd.read_csv("data/test-set.csv")
    Xtr = tr.values.astype(np.float32)
    Xte = te.iloc[:, :-1].values.astype(np.float32)
    yte = te["label"].values.astype(int)
    return Xtr, Xte, yte


def eval_scores(scores, yte, name):
    """scores: 越高越异常。"""
    auroc = roc_auc_score(yte, scores)
    auprc = average_precision_score(yte, scores)
    # 最佳 F1（在测试集上按阈值搜索）
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(yte, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best = int(np.argmax(f1))
    return {
        "name": name, "AUROC": auroc, "AUPRC": auprc,
        "F1": f1[best], "Prec": prec[best], "Rec": rec[best],
    }


def main():
    Xtr, Xte, yte = load()
    # 标准化（在训练集上拟合）
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    results = []

    # ---- 经典距离/密度方法 ----
    # 马氏距离（经验协方差）
    cov = EmpiricalCovariance().fit(Xtr_s)
    md = cov.mahalanobis(Xte_s)
    results.append(eval_scores(md, yte, "Mahalanobis(经验)"))

    # 稳健马氏距离（MCD）
    mcd = MinCovDet().fit(Xtr_s)
    md2 = mcd.mahalanobis(Xte_s)
    results.append(eval_scores(md2, yte, "Mahalanobis(MCD)"))

    # ---- pyod 模型 ----
    models = [
        ("OC-SVM", OCSVM(contamination=0.05)),
        ("IsolationForest", IForest(contamination=0.05, random_state=42)),
        ("LOF", LOF(contamination=0.05)),
        ("kNN", KNN(contamination=0.05)),
        ("ECOD", ECOD(contamination=0.05)),
        ("COPOD", COPOD(contamination=0.05)),
        ("ABOD", ABOD(contamination=0.05)),
        ("HBOS", HBOS(contamination=0.05)),
        ("PCA", PCA(contamination=0.05, random_state=42)),
        ("CBLOF", CBLOF(contamination=0.05, random_state=42)),
        ("MCD", MCD(contamination=0.05)),
        ("KDE", KDE(contamination=0.05)),
        ("SOS", SOS(contamination=0.05)),
    ]
    for name, m in models:
        try:
            m.fit(Xtr_s)
            s = m.decision_function(Xte_s)
            results.append(eval_scores(s, yte, name))
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}")

    # ---- 排序输出 ----
    results.sort(key=lambda r: -r["AUROC"])
    print(f"{'方法':22s} {'AUROC':>7s} {'AUPRC':>7s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s}")
    for r in results:
        print(f"{r['name']:22s} {r['AUROC']:7.4f} {r['AUPRC']:7.4f} "
              f"{r['F1']:7.4f} {r['Prec']:7.4f} {r['Rec']:7.4f}")

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/experiments.csv", index=False)


if __name__ == "__main__":
    main()
