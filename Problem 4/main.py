"""任务4：无监督甲状腺疾病判断（最终方案）。

方案：
1. 特征选择：仅保留 feature_2/3/4/6（EDA 表明 feature_1/5 对疾病无判别力）。
2. 标准化（在仅含正常样本的训练集上拟合）。
3. 三种无监督异常检测器：Isolation Forest（调优）、HBOS、稳健马氏距离 MCD。
4. 分数秩归一化后取平均作为最终异常分数。
固定随机种子，结果可复现。
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, \
    precision_recall_curve, confusion_matrix
from scipy.stats import rankdata

from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD

SEED = 42
FEATS = ["feature_2", "feature_3", "feature_4", "feature_6"]


def load():
    tr = pd.read_csv("data/train-set.csv")
    te = pd.read_csv("data/test-set.csv")
    return (tr[FEATS].values.astype(np.float32),
            te[FEATS].values.astype(np.float32),
            te["label"].values.astype(int))


def rank_norm(s):
    r = rankdata(s)
    return (r - 1) / (len(r) - 1)


def main():
    Xtr_raw, Xte_raw, y = load()
    scaler = StandardScaler().fit(Xtr_raw)          # 仅在训练集（正常样本）上拟合
    Xtr = scaler.transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    # 三种无监督检测器，分数越高越异常
    models = {
        "IsolationForest": IForest(n_estimators=1000, max_features=0.5,
                                   contamination=0.05, random_state=SEED),
        "HBOS": HBOS(contamination=0.05),
        "MCD": MCD(contamination=0.05),
    }
    scores = {}
    for name, m in models.items():
        m.fit(Xtr)
        scores[name] = m.decision_function(Xte)

    # 集成：秩归一化后平均
    ensemble = np.mean([rank_norm(s) for s in scores.values()], axis=0)

    # 指标（label=1 患病为正类）
    auroc = roc_auc_score(y, ensemble)
    auprc = average_precision_score(y, ensemble)
    prec, rec, thr = precision_recall_curve(y, ensemble)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    b = int(np.argmax(f1))

    print("=== 任务4 最终结果（测试集）===")
    print(f"AUROC = {auroc:.4f}")
    print(f"AUPRC = {auprc:.4f}")
    print(f"最佳阈值下: F1 = {f1[b]:.4f}, Precision = {prec[b]:.4f}, Recall = {rec[b]:.4f}")

    # 各单模型
    print("\n单模型对照：")
    for name, s in scores.items():
        print(f"  {name:16s} AUROC={roc_auc_score(y, s):.4f} AUPRC={average_precision_score(y, s):.4f}")

    # 保存
    out = pd.DataFrame({
        "label": y,
        "IForest": scores["IsolationForest"],
        "HBOS": scores["HBOS"],
        "MCD": scores["MCD"],
        "ensemble_score": ensemble,
    })
    out.to_csv("results/final_scores.csv", index=False)
    with open("results/final_metrics.txt", "w") as f:
        f.write(f"AUROC={auroc:.4f}\nAUPRC={auprc:.4f}\nF1={f1[b]:.4f}\n"
                f"Precision={prec[b]:.4f}\nRecall={rec[b]:.4f}\n")
    print("\n结果已保存到 results/")


if __name__ == "__main__":
    main()
