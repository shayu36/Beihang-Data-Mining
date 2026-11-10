# 图表与实测记录重建

本目录用于从包内 `results/run1/` 的完整实测记录生成报告图表。性能图不再读取旧报告转录数值；预测、模型配置、边界样本和图像响应均取自本次完整训练、测试与核验后保存的结果。

使用Python 3.12，在工程根目录的独立绘图环境中执行：

```powershell
python -m pip install numpy==2.3.5 matplotlib==3.11.2 Pillow==13.0.0
python figure-source/sync_results.py --run results/run1
python figure-source/render_figures.py
```

以上为本次绘图环境的依赖版本，不是模型训练环境。模型训练依赖及操作见工程的实验代码说明。若仅重绘已同步的数据，可直接运行最后一条命令。绘图运行只依赖工程内的 JSON、字体和相对路径素材，不使用来源记录中的绝对路径。

## 文件与数据来源

- `sync_results.py` 读取 `test_predictions.csv`、`test_metrics.csv`、`frozen.json`、`run_info.json` 和 `scores_before_labels.csv`，生成 `verified-results.json`。它逐类复算全局、局部和融合分支的混淆计数、Accuracy、Precision、Recall、F1、BA、AUROC、AP，并核对恒判正常基线、固定阈值和读标签前的预测一致性。出现不一致即停止。
- `verified-results.json` 包含全部102行预测、8行指标、模型配置、数据计数、真实边界样本、来源文件 SHA256，以及绘图素材索引。全部分数和余量保留原始浮点精度，显示时才舍入。
- `recomputed/` 包含从 `results/run1/visuals/` 复制的4张边界原图、4个48×48最近邻距离数组及3层实际通道均值数组。
- `render_figures.py` 是统一入口，调用 `render_verified_figures.py` 生成8组 PDF 与 PNG。
- `build_method_figure.py` 单独绘制中文方法框架图。融合权重、最高距离聚合数量、记忆库规模和训练图数读取实测配置。图中的矩阵、分类边界和小柱形是运算示意，不表示实测特征值。
- `originals/` 中的四张训练样例图用于展示课程数据；测试原图、距离数组与网络响应统一读取本轮 `recomputed/` 素材。

字体优先读取工程 `fonts/` 中的 Times New Roman、宋体和黑体。PDF 中的文字、曲线和框架为矢量元素；图像与特征响应保留真实位图或数组内容。PNG 预览为320 dpi，方法图为400 dpi。

## 八幅图的显示规则

1. `method-overview`：与参考报告一致的中文阶段标题、训练拟合区、共享特征、双评分分支、折外校准和融合判断。
2. `training-samples`：课程训练图像中的正常与异常样例。
3. `feature-responses`：本次从榛子 `train/good/002.png` 提取的各层通道均值。每层独立做最小值—最大值归一化，色标0–1；跨层颜色不表示同一绝对响应。
4. `roc-pr-curves`：两个产品分别绘制ROC与PR，共四个面板。曲线根据原始逐图分数及真实标签计算，同分样本一起纳入；AP使用阶梯加权。重合曲线用不同线型和线宽区分，不偏移或扰动曲线坐标。
5. `confusion-matrices`：全局、局部与融合分支的总体混淆矩阵，按照真实类别行归一化着色，格内同时显示整数计数和行百分比。各产品仍采用自身冻结阈值。
6. `error-complementarity`：动态列出至少一个分支误判的样本，FP为误报、FN为漏检；不存在硬编码样本列表。
7. `score-distribution`：各产品的融合分数分布。正常与异常直方图分别归一化为类内比例，等宽分箱以冻结阈值为一个箱边界；轴下短线对应每个真实分数。紫色虚线是训练内确定的阈值。
8. `qualitative-heatmaps`：每类最高分正常图与最低分异常图的原图和实际局部距离图。距离除以该图距离数组的99%分位数（下限1e-8），再截断到[0,1]；这是逐图独立归一化，统一色标不表示跨图绝对距离相同。直接显示48×48网格，不通过平滑、美化或改值突出缺陷。图像级标签不能用于证明像素定位精度。

当前测试集在项目早期曾用于比较版本。完整重跑和核验说明本次结果可重复，不会把这组固定课程数据变成未接触过的独立测试集。
