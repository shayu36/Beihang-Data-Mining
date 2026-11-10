"""全局配置：数据路径、滑动窗口、划分比例与模型超参数。"""

import os

# ---- 路径 ----
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "weather.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

# ---- 数据 ----
TARGET_COL = "OT"                 # 待预测的室外温度
WINDOW_SIZE = 12                  # 过去 2 小时 = 12 个时间点（每 10 min 一条）
SENTINEL = -9999.0                # 传感器异常占位值，需要清洗

# ---- 时序划分（严格按时间先后，禁止打乱）----
TEST_RATIO = 0.20                 # 最后 20% 作为测试集
VAL_RATIO = 0.15                  # 剩余部分的后 15% 作为验证集（早停用）

# ---- 训练 ----
SEED = 42
BATCH_SIZE = 256
EPOCHS = 200
LR = 5e-4
PATIENCE = 25                     # 验证集早停耐心
LR_PATIENCE = 8                   # 学习率衰减耐心
HIDDEN_SIZE = 128                 # LSTM 隐层维度
NUM_LAYERS = 2                    # LSTM 层数
DROPOUT = 0.2
DEVICE = "cpu"                    # 无 GPU 环境，纯 CPU 训练
GRU_SEEDS = [0, 1, 2, 3, 4]       # GRU 多种子集成的随机种子
