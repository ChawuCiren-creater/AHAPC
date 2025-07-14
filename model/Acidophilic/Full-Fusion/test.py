import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, num_features=2704, dropout_rate=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        fc_input_dim = (num_features // 8) * 128
        self.fc = nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x.view(-1, 1)


checkpoint_path = "2704_CNN_best_model.pt"
checkpoint = torch.load(checkpoint_path, weights_only=False)


model = CNN().to(device)
model.load_state_dict(checkpoint["model_state"])
scaler = checkpoint["scaler"]
print(scaler)
model.eval()
print("模型和 StandardScaler 加载完成！")

test_csv_file = "ESM2_650M+PSSM_DPC+ProtT5_Full_test.csv"
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