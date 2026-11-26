# 扩展实验复现

以下三项实验使用正常训练划分、校准规则和扰动方案选型，再评价冻结模型。甲状腺实验沿用的列方向与校准比例曾参考课程测试结果。伪异常检出率衡量对指定扰动的敏感性。第四项实验单独比较五折校准与单划分校准。[五折校准的结果与复现命令](crossfit/README.md)

从仓库的 `Extension 1` 目录执行以下 PowerShell 命令。先按主说明准备 Python 3.12 环境，将 `$archives` 改为两个原始课程 ZIP 的所在目录。条件残差和伪异常分类器只需甲状腺 ZIP；多尺度实验还需图像 ZIP 及主实验生成的特征缓存。[环境和主实验](../README.md)

```powershell
$implementation = (Resolve-Path '.').Path
$archives = 'D:\data'
$pythonExe = Join-Path $env:TEMP 'dm-extension1-venv/Scripts/python.exe'
$experiments = Join-Path $implementation 'extended'
$thyroid = Join-Path $archives 'thyroid.zip'
$images = Join-Path $archives 'Image_Anomaly_Detection.zip'
$output = Join-Path $env:TEMP ('dm-extension1-extended-' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
```

## 六列条件残差

用其余五列的近邻预测每一列的正常中位数和离散程度，聚合最大的两个残差。先保留正常误报率不超过 1%、方向扰动检出率达到两列乘积参照 90% 的候选，再按方向扰动与依赖关系扰动两类任务的平均检出率选择。

```powershell
$study = Join-Path $experiments 'conditional/conditional_study.py'
$out = Join-Path $output 'conditional'
& $pythonExe $study freeze --thyroid $thyroid --output $out
& $pythonExe $study evaluate --thyroid $thyroid --output $out
```

原始协议随同代码提供。评价前会核对协议、两份 Python 源码和模型哈希。结果目录包含三折各候选计数、选择原因所需汇总、冻结模型、逐条分数及课程指标。

## 伪异常训练分类器

按第 2、6 列的方向构造六种模拟异常，比较带正则化的逻辑回归和直方图梯度提升分类器。五折正常留出误报率不超过 1% 的候选中，以六种场景的平均模拟检出率选型。工作阈值由原划分中的 357 条正常校准记录确定。

```powershell
$out = Join-Path $output 'generative'
& $pythonExe (Join-Path $experiments 'generative/train.py') --thyroid $thyroid --output $out
& $pythonExe (Join-Path $experiments 'generative/evaluate_once.py') --thyroid $thyroid --output $out
```

训练只读取 ZIP 的正常训练成员，输出可加载的模型文件与五折记录。评价从该输出目录加载模型，先保存 1933 条无标签分数，再核对课程标签。

## 三域同参多尺度近邻

在 5 与 20 两个邻域尺度分别计算距离，用正常参考库估计邻域半径。三个数据域共用一个归一强度，由三折正常训练数据上的坐标替换与稀疏偏移实验选定，校准秩阈值均为 0.05。先按主说明运行一键复现，下面从其默认输出位置选取最近一次包含特征缓存的主运行；若自定义了 `-OutputRoot`，请修改 `$runs`。

```powershell
$runs = Join-Path $env:TEMP 'dm-extension1-results'
$run = Get-ChildItem -LiteralPath $runs -Directory -Filter 'run_*' |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'features.npz') } |
    Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
if (-not $run) { throw '请先完成主实验，或将 $run 设为含 features.npz 的完整主运行目录。' }
$baseline = Join-Path $implementation 'course-report/results/shared/scores_before_labels.csv'
$out = Join-Path $output 'multiscale'
& $pythonExe (Join-Path $experiments 'multiscale/multiscale_reference.py') --run $run --implementation $implementation --output $out --threads 2
& $pythonExe (Join-Path $experiments 'multiscale/evaluate_multiscale.py') --frozen $out --baseline $baseline --implementation $implementation --images $images --thyroid $thyroid
```

输入复用主实验的特征缓存；基线文件为原同参单尺度的无标签分数。评价会核对原始包、冻结记录、4070 行逐样本双方法分数，以及复算单尺度结果与原始基线的一致性。归一强度 1 在训练内伪异常上的平均 AP 为 0.7988，原单尺度为 0.7745。

## 课程测试结果

| 扩展模型 | 数据 | FP / FN | F1 | 对照 F1 |
|---|---|---:|---:|---:|
| 条件残差与两列乘积等权融合 | 甲状腺 | 10 / 5 | 0.9223 | 三列修正 0.9688 |
| 伪异常训练的直方图梯度提升 | 甲状腺 | 5 / 31 | 0.7778 | 三列修正 0.9688 |
| 局部密度归一多尺度近邻 | 榛子 | 3 / 10 | 0.4348 | 同参单尺度 0.9286 |
| 局部密度归一多尺度近邻 | 拉链 | 5 / 1 | 0.8235 | 同参单尺度 0.8750 |
| 局部密度归一多尺度近邻 | 甲状腺 | 107 / 22 | 0.5275 | 同参单尺度 0.6286 |

甲状腺两项扩展均有更多漏检。多尺度近邻在三域的固定工作点 F1 均下降。主实验保留原同参检测器和模态适配模型。仓库提供逐条分数、冻结记录与课程指标。[条件残差](../course-report/results/extended/conditional/) · [伪异常分类器](../course-report/results/extended/generative/) · [多尺度近邻](../course-report/results/extended/multiscale/)

## 校准与尾部对照

三列五折校准、连续尾部分布和背景子群模型的命令分别见 [五折校准](crossfit/README.md)、[尾部分布](tails/README.md) 和 [背景子群](subgroups/README.md)。同一课程测试集上，三列修正的 F1 为 96.875%。子群模型的精确率为 95.60%，比三列修正的 94.90% 略高；误报从 5 条减到 4 条，漏检从 1 条增至 7 条。五折集成和连续尾部方法的 F1 分别为 96.37% 和 88.04%。
