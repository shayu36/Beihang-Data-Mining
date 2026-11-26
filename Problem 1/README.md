# Problem 1：图像无监督聚类

对课程 `Cluster.zip` 的 600 张图像进行六类聚类。完整报告为 [PDF](course-report/course-report.pdf)，可编辑源码为 [LaTeX](course-report/course-report.tex)。

本次预定的 16 组实验全部得到 **ACC=100%、ARI=1、NMI=1**；传统颜色纹理基线也完全分对，因此没有证据证明深度方法在本数据上具有更高准确率。所有结果、中心和逐图预测已保存并独立核验；12 张固定抽样原图重编码与缓存特征的最大差异为 0。详细解释见 [包含全部题目的讲义](explanation/all-tasks-explained.pdf)。

## 方法与实验约定

- 主方法：冻结 ImageNet ResNet18，取全局平均池化后的 512 维特征，逐样本 L2 归一化，再做 K-means（6 簇）。主结果预先固定为种子 11。
- 传统基线：HSV 三维直方图（8×4×4）与 uniform LBP（8 邻域、半径 1、10 维），各直方图归一化后开方，等权拼接并归一化，总共 138 维。
- 对照：深度特征经 PCA 保留 64 维后再次 L2；同一原始深度特征上的 Ward 层次聚类。
- 三组 K-means 都使用种子 11、23、37、53、71；每个种子 20 次 K-means++ 初始化。层次聚类为确定性单次。共 16 组。
- 不以 ACC/ARI/NMI 选择方法、参数或随机种子。`protocol.json` 先于特征提取保存，所有分组先于真实标签读取写入，随后统一进行外部评价。
- ResNet 的外部预训练有监督；本任务中不微调网络，也不使用课程标签训练。600 张图片全部参与无标签聚类，这是给定数据集上的聚类实验，不是新图像留出测试。

## 安装与运行

使用 Python 3.12，建议创建独立虚拟环境。在本目录运行：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe run.py --archive '../DM_2026_Dataset/Cluster.zip' --output results/reproduction
.venv/Scripts/python.exe verify.py results/reproduction
.venv/Scripts/python.exe plots.py --run results/reproduction --archive '../DM_2026_Dataset/Cluster.zip' --output course-report/figures-reproduction
.venv/Scripts/python.exe -m unittest test_evaluation -v
```

输出目录必须尚不存在，以免覆盖已有实验。默认使用 CPU、4 个线程、批大小 16。首次运行自动下载官方 ResNet18 权重，保存在本目录 `.cache/torch/checkpoints/`；离线可通过 `--weights /path/to/resnet18-f37072fd.pth` 指定权重。权重及数据包不包含在源码交付压缩包中。

已执行的实验位于 `results/run/`，工作区复现环境位于项目根目录 `.venv-task1/`。前述命令是便于迁移的独立安装方法。

## 结果文件和核验

| 文件 | 用途 |
|---|---|
| `protocol.json` | 特征、算法、参数、主结果种子和评价约定 |
| `manifest.json` | 数据检查、环境版本、权重与源代码 SHA256、特征提取耗时 |
| `features.npz` | 文件名、各图哈希和三种特征矩阵 |
| `clusters.npz` | 全部 16 组逐图簇编号 |
| `centers.npz`、`pca_model.npz` | K-means 中心、PCA 变换参数 |
| `sealed_before_evaluation.json` | 标签读取前保存的核心文件哈希 |
| `internal_metrics.csv` | 标签读取前计算的轮廓系数、CH、DB 等 |
| `predictions.csv` | 主方法逐图簇编号与最终关联的真实类别 |
| `all_metrics.csv`、`summary.json` | 16 组完整评价及均值、样本标准差（ddof=1） |
| `metrics.json` | 主方法 ACC、ARI、NMI、匹配关系与混淆矩阵 |
| `verification.json` | 独立核验实际执行记录 |

ACC 采用最优一对一簇—类别匹配。ARI/NMI 无须匹配标签编号。`verify.py` 由列联表独立计算 ARI/NMI，并核对官方标签、保存预测、全部运行指标及封存哈希。统计指标的独立复算不等于在另一台机器重新训练。

跨随机种子稳定性用两次分组之间的 ARI 计算，不需要真实标签。轮廓系数、CH、DB 取决于特征空间，不能把不同表示中的数值直接当作同一尺度的优劣排名。本文所有模型设置均预先声明；评价指标用于报告而不用于回头选取更好结果。

## 代码和报告结构

- `features.py`：无标签图片读取、质量检查与特征提取。
- `run.py`：固定协议、聚类、封存、最终评价。
- `verify.py`、`test_evaluation.py`：独立评价与有意义的边界测试。
- `plots.py`：从保存的同一份结果生成图表，输出 PDF、PNG 与作图元数据。
- `course-report/`：与队友一致的双栏报告、参考文献、图表及编译脚本。

报告使用系统中文字体，不分发商业字体文件。编译方式见 `course-report/build-report.ps1`。图表重绘与正文数值需保持对应；更新实验时请使用新的输出目录并同步报告。

可补充运行 `.venv/Scripts/python.exe verify_features.py results/run`，按 `np.linspace(0,599,12,dtype=int)` 的固定索引，从原 ZIP 抽取 12 张图像核对逐图 SHA256 并重新编码，不读取类别标签。脚本独立构建 ResNet18，仅读取本地官方权重（可用 `--weights` 指定），采用官方预处理和 float64 L2 归一化；传统特征复用 `traditional_descriptor`。两种特征均以 `rtol=0, atol=1e-6` 比较，最大绝对误差和逐图结果保存至 `feature_verification.json`。这是固定样本的编码抽查，不是全量重新训练。
