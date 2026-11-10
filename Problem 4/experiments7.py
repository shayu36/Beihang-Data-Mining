"""任务4 广泛方法扫描：约 25 种无监督检测器。"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.kpca import KPCA
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.lunar import LUNAR
from pyod.models.mad import MAD
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.qmcd import QMCD
from pyod.models.rod import ROD
from pyod.models.sod import SOD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE

SEED = 42
FEATS = ["feature_2", "feature_3", "feature_4", "feature_6"]


def load():
    tr = pd.read_csv("data/train-set.csv")[FEATS].values.astype(np.float32)
    te = pd.read_csv("data/test-set.csv")
    return tr, te[FEATS].values.astype(np.float32), te["label"].values.astype(int)


def m(s, y):
    return (roc_auc_score(y, s), average_precision_score(y, s))


def main():
    Xtr_raw, Xte_raw, y = load()
    sc = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)

    models = [
        ("IForest", IForest(n_estimators=1000, max_features=0.5, contamination=0.05, random_state=SEED)),
        ("HBOS", HBOS(contamination=0.05)),
        ("MCD", MCD(contamination=0.05)),
        ("CBLOF", CBLOF(n_clusters=8, contamination=0.05, random_state=SEED)),
        ("PCA", PCA(contamination=0.05, random_state=SEED)),
        ("kNN", KNN(n_neighbors=20, contamination=0.05)),
        ("KDE", KDE(contamination=0.05)),
        ("GMM", GMM(n_components=4, contamination=0.05, random_state=SEED)),
        ("OCSVM", OCSVM(nu=0.05)),
        ("ABOD", ABOD(contamination=0.05)),
        ("COF", COF(contamination=0.05)),
        ("COPOD", COPOD(contamination=0.05)),
        ("ECOD", ECOD(contamination=0.05)),
        ("FeatureBagging", FeatureBagging(contamination=0.05, random_state=SEED)),
        ("INNE", INNE(contamination=0.05, random_state=SEED)),
        ("KPCA", KPCA(contamination=0.05)),
        ("LMDD", LMDD(contamination=0.05)),
        ("LODA", LODA(contamination=0.05)),
        ("LOF", LOF(n_neighbors=30, contamination=0.05)),
        ("LUNAR", LUNAR(contamination=0.05)),
        ("MAD", MAD()),
        ("QMCD", QMCD(contamination=0.05)),
        ("ROD", ROD(contamination=0.05)),
        ("SOD", SOD(contamination=0.05)),
        ("DeepSVDD", DeepSVDD(n_features=4, contamination=0.05, random_state=SEED, epochs=50, verbose=0)),
        ("AutoEncoder", AutoEncoder(contamination=0.05, random_state=SEED, epoch_num=50, verbose=0)),
        ("VAE", VAE(contamination=0.05, random_state=SEED, epoch_num=50, verbose=0)),
    ]

    results = []
    for name, model in models:
        try:
            model.fit(Xtr)
            s = model.decision_function(Xte)
            a, ap = m(s, y)
            results.append((name, a, ap, s))
            print(f"{name:16s} AUROC={a:.4f} AUPRC={ap:.4f}")
        except Exception as e:
            print(f"{name:16s} FAIL {type(e).__name__}")

    results.sort(key=lambda r: -r[2])
    print("\n=== 按 AUPRC 排序 ===")
    for name, a, ap, s in results:
        print(f"{name:16s} AUROC={a:.4f} AUPRC={ap:.4f}")

    # 保存
    os.makedirs("results", exist_ok=True)
    np.save("results/sweep_scores.npy", np.stack([r[3] for r in results], axis=1))
    with open("results/sweep_names.txt", "w") as f:
        f.write("\n".join([r[0] for r in results]))
    pd.DataFrame([(r[0], r[1], r[2]) for r in results], columns=["name", "AUROC", "AUPRC"]).to_csv("results/sweep.csv", index=False)


if __name__ == "__main__":
    import os
    main()
