"""时间序列预测模型：LSTM / GRU。"""
import torch
import torch.nn as nn

from config import HIDDEN_SIZE, NUM_LAYERS, DROPOUT


class RecurrentRegressor(nn.Module):
    """多层 RNN（LSTM/GRU）+ 全连接回归头，输出下一时间点的室外温度。"""

    def __init__(self, input_dim, hidden=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 dropout=DROPOUT, cell="lstm"):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        cell_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = cell_cls(input_size=input_dim, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        # x: (batch, window, input_dim)
        out, _ = self.rnn(x)
        last = out[:, -1, :]                 # 取最后一个时间步的隐状态
        return self.head(last).squeeze(-1)   # (batch,)


def build_model(input_dim, cell="lstm"):
    return RecurrentRegressor(input_dim, cell=cell)
