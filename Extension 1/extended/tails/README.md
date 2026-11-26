# 甲状腺边际尾部实验

第 2 列在正常样本上拟合的 Yeo–Johnson 参数约为 −8.33，较大的正值经变换后会趋于饱和。本实验尝试将超过正常第 90 百分位的定向坐标连续衔接到连续尾部分数，以保留正常最大值以外的排序。拼接点两侧的分数连续；原三列修正中第 3 列的衰减项保持原公式。

使用与主实验相同的 Python 依赖。在 `代码` 或报告的 `implementation` 目录运行以下命令，替换为本机课程数据路径。第一步只读取正常训练文件，要求输出目录尚不存在。第二步读取已冻结的唯一候选，核对模型和协议的 SHA-256 后评价一次。

```powershell
$data = 'D:\data\thyroid.zip'
$dest = Join-Path $env:TEMP ('extension1-tails-' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
python extended/tails/tail_experiment.py train --thyroid $data --code . --output $dest
python extended/tails/tail_experiment.py evaluate --thyroid $data --code . --output $dest
python extended/tails/tail_experiment.py export --thyroid $data --code . --output $dest
```

三种候选分别为仅扩展第 2 列的指数尾、仅扩展第 2 列的广义帕累托尾，以及同时扩展第 2 列上尾和第 6 列下尾的广义帕累托尾。五折训练内比较共留出 1425 条正常样本，另保留 357 条正常样本用于冻结候选的最终校准。每折按联合强弱和单侧偏重生成四类扰动，分别记录原阈值附近的检出率、各强度的检出率以及正常样本误报。扰动中第 1、3、4、5、6 列来自同一真实正常记录。

三种新候选均误报 5/1425，86 条指定的单轴尾部正常记录中均为 0 次误报。按预定训练规则选出的双尾候选，在边界附近的模拟检出率为 20.94%，原三列参照为 22.75%。应用冻结模型到课程测试集，双尾候选误报 9 条、漏检 13 条，F1 为 88.04%。部署采用的三列修正误报 5 条、漏检 1 条，F1 为 96.875%。模拟检出率对应人工扰动，课程测试集历史上曾用于选择原三列方向和配置。

报告证据目录 `results/extended/tails/` 中，`protocol.json` 记录预设规则，`frozen.json` 记录逐折比较、所选模型和阈值，`course_evaluation.json` 对应一次课程测试评价。本地训练同时产生 `model.joblib`。

最后的 export 命令重新载入冻结模型，将 1933 条无标签分数、既定阈值和决策写入 `scores_before_labels.csv`。随后用保存的 CSV 关联课程标签，复核混淆计数、F1、AUROC 等指标，结果见 `score_verification.json`。
