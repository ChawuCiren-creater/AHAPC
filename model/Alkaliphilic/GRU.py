import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1, dropout_rate=0.5):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=3, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        gru_out, _ = self.gru(x)
        last_hidden_state = gru_out[:, -1, :]
        output = self.fc(last_hidden_state)
        output = self.sigmoid(output)
        return output