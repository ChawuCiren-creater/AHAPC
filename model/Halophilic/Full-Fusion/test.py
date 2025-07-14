import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

checkpoint_path = "2324_BiLSTM_best_model.pt"
checkpoint = torch.load(checkpoint_path, weights_only=False)

model = BiLSTM().to(device)
model.load_state_dict(checkpoint["model_state"])
scaler = checkpoint["scaler"]
print(scaler)
model.eval()
test_csv_file = "ESM2_650M+AAC+ProtBert_Full_test.csv"
test_data = pd.read_csv(test_csv_file, header=1)

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

seed = 36

np.random.seed(seed)
num_features = X_test.shape[1]
feature_indices = np.random.permutation(num_features)
X_test = X_test[:, feature_indices]

X_test = scaler.transform(X_test)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

with torch.no_grad():
    outputs = model(X_test).cpu().numpy()
    y_test_np = y_test.cpu().numpy()

predicted = (outputs >= 0.5).astype(int)
accuracy = accuracy_score(y_test_np, predicted)
precision = precision_score(y_test_np, predicted, zero_division=0)
recall = recall_score(y_test_np, predicted)
specificity = recall_score(y_test_np, predicted, pos_label=0)
f1 = f1_score(y_test_np, predicted)
auc = roc_auc_score(y_test_np, outputs)
mcc = matthews_corrcoef(y_test_np, predicted)
balanced_acc = (recall + specificity) / 2

tn, fp, fn, tp = confusion_matrix(y_test_np, predicted).ravel()

fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print(f"Performance:")
print(f"ACC:  {accuracy:.4f}")
print(f"P:    {precision:.4f}")
print(f"R:    {recall:.4f}")
print(f"Sp:   {specificity:.4f}")
print(f"F1:   {f1:.4f}")
print(f"AUC:  {auc:.4f}")
print(f"MCC:  {mcc:.4f}")
print(f"BA:   {balanced_acc:.4f}")
print(f"FPR:  {fpr:.4f}")
print(f"FNR:  {fnr:.4f}")