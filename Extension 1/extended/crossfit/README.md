# 甲状腺三列分数的五折正常校准

该实验检验固定三列方法对校准样本划分的敏感性。对 1,782 条去重正常记录作五折划分，每次用四折拟合原有变换和三列分数，用留出的一折确定上尾阈值。每条待测记录得到五个“分数／本折阈值”，中位数严格大于 1 时判为异常。单折沿用 α=0.005。五个模型的参考数据有重叠，集成的错误率由实验测量。

## 复现

从仓库的 `Extension 1` 目录执行以下 PowerShell 命令。按主说明创建 Python 3.12 环境，再将 `$archive` 改为本机原始甲状腺 ZIP 的位置。这项实验可直接从该 ZIP 开始。[环境安装](../../README.md)

```powershell
$study = (Resolve-Path 'extended/crossfit').Path
$archive = 'D:\data\thyroid.zip'
$python = Join-Path $env:TEMP 'dm-extension1-venv/Scripts/python.exe'
$out = Join-Path $env:TEMP ('thyroid_crossfit_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $python (Join-Path $study 'run_crossfit.py') freeze --thyroid $archive --output $out
& $python (Join-Path $study 'run_crossfit.py') evaluate --thyroid $archive --output $out
```

`freeze` 仅读取正常训练 CSV，输出五折外层留出诊断、各折阈值和模型文件。`evaluate` 先核对协议、源码及模型哈希，保存逐样本分数和判定后再读取测试标签。模型文件由本地训练生成。将 `$study` 加入 Python 模块搜索路径后，可用 `joblib.load` 读取 `$out/crossfit_model.joblib`，调用 `model.predict(x)`。输入形状为 `(样本数, 6)`，列顺序与原始数据一致。[逐样本分数与课程指标](../../course-report/results/extended/crossfit/)

## 实测

同一批外层正常留出记录上，单划分三列方法误报 6/1,782，五折集成误报 3/1,782。三种模拟定向扰动的平均检出率由 60.70% 降至 58.27%。课程测试上，五折集成出现 6 次误报和 1 次漏判，F1 为 96.37%；原单划分三列方法出现 5 次误报和 1 次漏判，F1 为 96.875%。上游特征与 α 曾使用课程测试集开发，模拟检出率只描述所设定的扰动。
