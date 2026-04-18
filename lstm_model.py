import torch
import torch.nn as nn
import numpy as np


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = StockLSTM(7, 128, 2, 3)
batch = torch.randn(32, 20, 7)

# See each step
lstm_out, _ = model.lstm(batch)
print("After LSTM:", lstm_out.shape)

last_day = lstm_out[:, -1, :]
print("Last day:", last_day.shape)

final = model.fc(last_day)
print("Final output:", final.shape)