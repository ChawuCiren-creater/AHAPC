import torch
import torch.nn as nn
import torch.optim as optim

class BiLSTM(nn.Module):
    def __init__(self, input_size=495, hidden_size=128, output_size=1, num_layers=2, dropout_rate=0.5):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out