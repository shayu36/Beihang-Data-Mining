# Problem 2

图像异常检测。最终报告见 [PDF](course-report/course-report.pdf) 与 [LaTeX](course-report/course-report.tex)。

安装 `requirements.txt` 后，使用课程数据运行：

```powershell
python run.py experiment --archive 'D:\data\Image_Anomaly_Detection.zip' --output 'D:\results\task2-new'
python verify.py --run 'D:\results\task2-new' --archive 'D:\data\Image_Anomaly_Detection.zip'
```

请替换数据和输出路径；输出目录须尚不存在。首次运行会下载 Wide ResNet50-2 权重，离线时可用 `--weights` 指定本地权重。仓库不包含课程数据和训练权重。报告源码保留样式、文献和插图；编译所需系统字体未随公开仓库分发，请从原课程模板或已授权字体安装中提供。
