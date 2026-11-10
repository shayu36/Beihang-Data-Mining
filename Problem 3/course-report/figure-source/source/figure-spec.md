# Model overview

## 当前交付：中文可编辑版本

数学记号以 `course-report/sec/task3-method.tex` 为准，验证集选权以
`course-report/sec/task3-experiments.tex` 的 `eq:fusion_selection` 为准。
采用带样本索引的 `g_i`、`r_i`、标准差 `s_j,s_y`、成员输出
`f_{\theta_k}(\widetilde{\mathbf X}_i)`，以及论文中的两项独立训练目标。
反标准化使用与 `eq:final_prediction` 等价的紧凑表达式；LaTeX 文件同时保留完整原式。

- `模型主图_中文版.pptx`：16:9，一页，411 个原生对象，其中 39 个原生 Office 数学对象，无位图。
- `results/report_figures/model_overview_zh.png`：由 PowerPoint 实际渲染的 3840×2160 中文主图。
- `模型主图_数学符号与公式.tex`：36 组公式/数学标签、51 项符号定义，可由 XeLaTeX 独立编译。

重建脚本为 `build_chinese_ppt.py`；公式由 `native_math.py` 转为原生可编辑数学对象。
`verify_chinese_ppt.py` 检查单页结构、原生公式、公式源码覆盖与当前论文关键公式一致性。
已检查 PowerPoint 实际渲染和 XeLaTeX 编译结果。

## 早先英文图的生成记录

Source: `main.py`, `src/models.py`, `config.py`, `results/metrics.json`, and the CO₂ field mapping established in `Problem3_当前模型方法与实验详解.md`.

Destination: main method figure; English paper typography; landscape 2:1; raster illustration. Output: `results/report_figures/model_overview.png`.

Style references: the three user-supplied architecture screenshots. Borrow pastel semantic regions, thin dashed boundaries, left-to-right arrows, serif type, and a bottom training band. Their unrelated scientific modules and images are not part of this method.

Topology: a standardized 12×21 input fans out to five independently fitted two-layer GRUs (hidden 128, head 128→64→1 with ReLU and Dropout 0.2), whose scalar predictions are averaged, and a separate flatten-to-252 Ridge branch (alpha 1). Validation MSE selects a weight from {0, 0.05, …, 1}. Stored weight 0.95 produces z=0.95g+0.05r. Inverse standardization produces a scalar CO₂ concentration in ppm. GRU MSE, Ridge squared residuals plus L2, and validation weight selection are separate stages. No joint total loss.

The input is labelled a 12-record window because the current code's string ordering does not establish a strictly chronological two-hour window. The figure shows model computation, without unverified performance plots.

Generation: built-in image-generation tool. Full generation prompt is retained below. Final output is a PNG raster image (1774×887); text and shapes are not separate editable objects. Tensor grids are schematic; numerical dimensions are given by the adjacent labels.

Final topology clarification: the purple validation-control arrow terminates at the `w* = 0.95` parameter box. No purple control path crosses the GRU or Ridge prediction paths. The two black multiplication nodes use 0.95 and 0.05 respectively. Scientific checks covered all five GRU/head paths, prediction mean, separate Ridge input, fusion, inverse scaling, and three distinct fitting/selection objectives. The final image was visually inspected for readable labels and unambiguous connectors.

## Prompt

Create a polished scientific model architecture main figure for a research paper. Use case: infographic-diagram. It explains the EXISTING model: five independently trained GRU regressors, prediction averaging, and a Ridge regression branch, combined using a validation-selected weight, for next-record CO2 concentration prediction. This must be a precise computation diagram, with no invented modules or experimental curves.

VISUAL DIRECTION: wide 2:1 landscape, target 2400 x 1200 pixels or larger at same ratio, clean white background, Times New Roman-style serif type and clean mathematical typography. Match the visual grammar of academic architecture figures: pale pastel rounded semantic enclosures, thin colored dashed boundaries, black crisp short arrows, neat small tensor slabs, generous yet compact spacing, bold slightly italic group headings. No headline above the figure, no stage numbers, no glossy effects, no cars or camera images. Use pale blue for input (#EDF4FB), pale teal for GRU (#EAF6F1), pale amber for Ridge (#FFF5DA), pale lavender for fusion (#F2ECFA), pale pink for training (#FBECEF). All label text dark charcoal and highly readable at journal two-column width. Thick enough strokes to print. The scientific flow is left to right. Keep aligned region tops and bottoms. Dominant center region with real parallel branches, not four independent narrative cards.

COMPOSITION:
Top approximately 77% of canvas is inference. Below it, a compact single pink band takes 20% height, separated by whitespace, titled "Training and validation" at its upper edge. Outer margins about 40px. Top has three semantic regions with widths about 22% / 48% / 27%, with small inter-region gaps.
LEFT region title "Multivariate input". Within it vertically show a 12-row by 21-column thin outlined pastel matrix, with a highlighted last column. Label it "12 records × 21 variables" and "CO₂ history included". Put tiny bracket labels "records" and "variables", no numerical data in cells. Below matrix a downward arrow to rounded operator "Train-fitted standardization". Below that a compact equation "X̃ = (X − μ) / σ". Below that small tidy standardized tensor slabs labelled "X̃ ∈ ℝ¹²ˣ²¹". A single arrow from this output goes right and branches to BOTH the GRU lane and the Ridge lane. This is a schematic tensor, not experimental evidence. Do NOT label this input strictly chronological or two hours, because the current implementation has a documented row-ordering issue.

CENTER region overall title "GRU ensemble with a linear branch".
Its upper two thirds is a teal dashed enclosure titled "Five GRU regressors". Show five clearly parallel rows, each a chain of TWO blocks and a scalar output. For row k, first block "GRU k" and second block "Head k", output "gₖ". Write actual k=1,2,3,4,5 in the five rows (GRU 1 ... GRU 5 and Head 1 ... Head 5, outputs g₁ ... g₅). Feed the SAME shared standardized input to all five GRU blocks using a clean branching bus. Connect all five scalar g outputs to ONE vertical mean aggregation operator on the right, labelled "Mean", with equation below "g = ⅕ Σₖ gₖ". Output g leads to the right fusion region. Five separately trained models, no chain between the five models.
Place two compact common structure annotations ABOVE the five rows inside the teal enclosure, aligned with their respective block columns: above GRU blocks "2 layers · H = 128" and above heads "Last state → 128 → 64 → 1". Beneath the head annotation, in legible secondary type, "ReLU · Dropout 0.2". Avoid expanding all gate formulas. Layer dropout=0.2 can be shown as a compact note near GRU common annotation if space permits.
Center bottom third is an amber rounded enclosure titled "Ridge branch". Feed the same X̃ into "Flatten" → "q ∈ ℝ²⁵²" → "Ridge, α = 1" → output "r". Include a compact formula "r = βᵀq + b" inside the Ridge area. This parallel lane uses the same standardized target as the GRU and never receives GRU hidden states or predictions.

RIGHT region title "Weighted fusion and forecast", pale lavender.
The GRU output g enters a distinct multiplication circle marked "×" with adjacent coefficient "0.95". The Ridge output r enters a second multiplication circle marked "×" with adjacent coefficient "0.05". These TWO scaled paths converge into ONE addition circle marked "+". Then an arrow to a formula block "z = 0.95g + 0.05r". Then downward arrow to "Inverse standardization". Then formula "ŷ = μᵧ + σᵧ z". Then a small concentrated output glyph (a single scalar dot with clean arrow, NO made-up waveform or measurement) with label "Next-record CO₂" and unit "(ppm)".
Show a small lavender annotation near fusion, "w* = 0.95", with dashed purple control connector FROM bottom validation-weight selection UP to the fusion coefficient area. The weight is fixed at inference.

BOTTOM PINK BAND: three horizontally arranged compact cells, all readable and connected by no spurious arrows. Title "Training and validation". Left cell title "GRU fitting", formula "Lₖ = mean[(gₖ − ỹ)²]", short note "Adam · validation early stopping". Middle cell title "Ridge fitting", formula "Lᵣ = Σ(r − ỹ)² + α‖β‖₂²", short note "Training set · α = 1". Right cell title "Validation weight selection", formula "w* = argmin_w MSEᵥₐₗ(wg + (1−w)r)", below "w ∈ {0, 0.05, …, 1}". The dashed control arrow goes from this right cell to fusion. These are three separate stages; NO single joint total loss. No training gradient from fused predictions to GRUs. No test labels in training. Thin solid small line sample labelled "Prediction flow" and dashed purple sample labelled "Validation control" may be a compact legend at bottom right only if it fits cleanly.

QUALITY AND SCIENTIFIC CONSTRAINTS: Exact labels, numbers and equations above. Exactly five GRU members, two layers each, hidden width 128, head 128→64→1. Independent GRU fitting with MSE and early stopping; Ridge trained separately; 21-weight validation grid and fixed weighted sum. Use output CO₂ ppm, never temperature, OT, degrees C, multi-step forecasts, attention, LSTM, decoder branches, residual learning, or joint total loss. No empirical metric numbers. All 5 input branches and 5 mean-input paths explicit and non-overlapping. Reserve whitespace for edges. No arrows through text. No clipped labels or tiny illegible equations. This should read like a carefully typeset professional computer-science method figure, strongly inspired by pastel dashed-container paper diagrams with a horizontal inference mechanism and a bottom supervision band.
