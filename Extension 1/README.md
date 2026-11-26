# 选做一 · 跨模态异常检测

图像缺陷可能只占一小块区域，甲状腺异常则表现为六个数值或其关系偏离正常范围。实验把两种输入都转换为描述子，与正常训练样本建立的参考库比较，再聚合为每个样本的异常分数。图像还可利用训练集中的异常标签进行监督融合。方法、逐域对照和错误分析见[课程报告](course-report/course-report.pdf)。

## 课程测试结果

| 数据 | 最终方法 | 正确数 | 精确率 | 召回率 | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 榛子 | 局部正常参考与全局 SVM 融合 | 55 / 55 | 100% | 100% | 100% |
| 拉链 | 局部正常参考与全局 SVM 融合 | 47 / 47 | 100% | 100% | 100% |
| 甲状腺 | 第 2、3、6 列的正常参考分数 | 1927 / 1933 | 94.90% | 98.94% | 96.875% |

**正式甲状腺模型采用三列修正版。** 仓库已附训练好的模型及其 SHA-256 校验记录。安装依赖后，在 `Extension 1` 目录即可对一条原始六维记录评分。[三列修正模型](course-report/results/correction/thyroid_corrected.joblib) · [模型校验值和完整指标](course-report/results/correction/comparison.json)

```powershell
python predict.py --model course-report/results/correction/thyroid_corrected.joblib --values -1.082333720071735 -0.15280509003829099 0.24374268267193666 -0.1532988415335525 -0.3916414902550237 0.05513767816060435
```

三个数据集使用相同的正常近邻算法与判断规则时，F1 依次为 0.9286、0.8750、0.6286。甲状腺原两列模型漏检 1 条、误报 6 条，三列修正减少了 1 条误报。变换与阈值只使用正常训练记录拟合。两列的特征方向、三列修正和拉链候选范围在开发中曾参考课程测试反馈，三列修正从 96 种历史配置中选出，因此表中数值属于课程开发结果。两类图像在各自测试集中均只有 15 张异常图像。

另用正常训练数据完成六项追加对照。局部密度归一多尺度近邻在榛子、拉链、甲状腺上的 F1 分别为 0.4348、0.8235 和 0.5275，均低于原同参近邻。甲状腺各变体的误报、漏检和排序指标见报告表 5。[追加实验的命令与逐域对照](extended/README.md)

## 一键复现

从仓库根目录进入 `Extension 1`，使用 Python 3.12 和 PowerShell。将下面两个 ZIP 路径替换为本机课程数据的位置，保留压缩包内部原始结构。首次运行会下载图像编码器的 ResNet18 权重。已有离线权重时可在最后一行追加 `-Weights 'D:\models\resnet18-f37072fd.pth'`。[官方权重](https://download.pytorch.org/models/resnet18-f37072fd.pth)

```powershell
Set-Location -LiteralPath 'Extension 1'
$envDir = Join-Path $env:TEMP 'dm-extension1-venv'
python -m venv $envDir
$pythonExe = Join-Path $envDir 'Scripts\python.exe'
& $pythonExe -m pip install -r requirements.txt

$images = 'D:\data\Image_Anomaly_Detection.zip'
$thyroid = 'D:\data\thyroid.zip'
$out = Join-Path $env:TEMP 'dm-extension1-results'
.\一键复现.ps1 -Python $pythonExe -Images $images -Thyroid $thyroid -OutputRoot $out -Threads 4
```

一键脚本先训练三个数据集，并从 602 张原图重提特征进行核验。它会打印两处本地输出。主目录 `run_<时间戳>` 含图像模型、两列甲状腺模型、特征、逐样本预测和指标。配对的 `supplements_<同一时间戳>` 含三列修正模型、统一算法对照、图像嵌套五折和甲状腺训练内模拟结果。

## 单条预测

在完成上述一键运行后，下面的命令选取最近一次主运行及其配对输出。甲状腺模型接收按原始列序排列的六个数。图像示例中的路径需替换为本地 PNG，榛子图像使用 `hazelnut.joblib`。结果的 `prediction` 为 0 表示正常，为 1 表示异常。

```powershell
$run = Get-ChildItem -LiteralPath $out -Directory -Filter 'run_*' | Sort-Object Name -Descending | Select-Object -First 1
$runDir = $run.FullName
$extraDir = Join-Path $out ($run.Name -replace '^run_', 'supplements_')
$corrected = Join-Path $extraDir 'correction\thyroid_corrected.joblib'

& $pythonExe predict.py --model $corrected --values -1.082333720071735 -0.15280509003829099 0.24374268267193666 -0.1532988415335525 -0.3916414902550237 0.05513767816060435
& $pythonExe predict.py --model (Join-Path $runDir 'zipper.joblib') --image 'D:\images\zipper.png'
```

## 其他实验与核验

以下示例用于单独复现一项分析。脚本和本 README 在同一目录，报告源文件、插图和实验记录位于 `course-report/`。本地运行可重建训练模型及 `features.npz`。直接阅读 PDF 即可查看图表；重新编译 LaTeX 时需在本地提供模板使用的授权中文和西文字体。[报告源码](course-report/course-report.tex)

```powershell
$check = Join-Path $out ('checks_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $pythonExe shared_reference.py --run $runDir --images $images --thyroid $thyroid --output (Join-Path $check 'shared') --threads 4
& $pythonExe nested_image_cv.py --features (Join-Path $runDir 'features.npz') --output (Join-Path $check 'nested_image_cv.json') --threads 4
& $pythonExe train_only_challenge.py --thyroid $thyroid --output (Join-Path $check 'training_challenge.json')
& $pythonExe compare_correction.py --thyroid $thyroid --output (Join-Path $check 'correction')
& $pythonExe verify_correction.py --run (Join-Path $check 'correction') --thyroid $thyroid --reference 'course-report/results/correction/predictions.csv'
```

`shared_reference.py` 复测完全共享的检测器。`nested_image_cv.py` 只使用训练图像，报告外层留出的预测表现。`train_only_challenge.py` 在正常训练记录和人工扰动上检验特征设计，输出中的模拟检出率仅针对构造的扰动。`verify_correction.py` 将可加载模型的 1933 条预测与历史逐条记录对照，并调用单条预测入口核验正常、异常两种输出。

若需复查固定三列分数的所有阈值，可让训练脚本直接把主结果写入 `$check\run2`。上面的 `$check\correction` 保存同次对照预测，两者共同作为审计输入。该步骤计算的是特定冻结分数在课程集上的事后上限。

```powershell
& $pythonExe run.py --images $images --thyroid $thyroid --output (Join-Path $check 'run2') --threads 4
& $pythonExe threshold_audit.py --thyroid $thyroid --results $check --output (Join-Path $check 'threshold_audit.json')
```

## 追加模型的比较

[追加实验及运行方法](extended/README.md)覆盖六维条件残差、模拟正例、多尺度参考、五折校准、尾部建模和背景子群校准。各模型先在训练数据上定型。课程测试中，背景子群精确率达到 95.60%，召回率为 92.55%；按 F1 评价，三列修正仍以 96.875% 领先。逐方法计数与 AP 见报告表 5。
