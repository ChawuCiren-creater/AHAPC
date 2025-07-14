import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM  = "the number of features"
CLASS_NAMES  = ["acidophilic", "halophilic", "alkaliphilic"]
MODEL_DIR    = r"BiLSTM_best_models"
TEST_CSV     = r"vertical_merged_test_fixed.csv"

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1, num_layers=2, dropout_rate=0.5):
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

df = pd.read_csv(TEST_CSV)
y_true = df["Label"].values
X      = df.drop("Label",axis=1).values.astype(np.float32)
X_t    = torch.from_numpy(X).to(device)

probs = {}
for cls, ModelClass in zip(CLASS_NAMES, [BiLSTM, BiLSTM, BiLSTM]):
    path = os.path.join(MODEL_DIR, f"{cls}_best_model.pt")
    model = ModelClass(FEATURE_DIM).to(device) if cls!="salt" else ModelClass(FEATURE_DIM).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    with torch.no_grad():
        probs[cls] = model(X_t).cpu().numpy().ravel()

# —— Soft voting —— #
prob_mat = np.stack([probs[c] for c in CLASS_NAMES], axis=1)
y_pred   = np.argmax(prob_mat, axis=1)

acc = accuracy_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)
cm  = confusion_matrix(y_true, y_pred)
print("=== Overall Metrics ===")
print("Accuracy:", f"{acc:.4f}")
print("MCC:     ", f"{mcc:.4f}")
print("Confusion Matrix:\n", cm)

print("\n=== Per-Class Sn & Sp ===")
for i, cls in enumerate(CLASS_NAMES):
    TP = cm[i,i]
    FN = cm[i,:].sum() - TP
    FP = cm[:,i].sum() - TP
    TN = cm.sum() - (TP + FN + FP)
    Sn = TP/(TP+FN) if (TP+FN)>0 else 0
    Sp = TN/(TN+FP) if (TN+FP)>0 else 0
    print(f"{cls} (class {i}): Sn = {Sn:.4f}, Sp = {Sp:.4f}")
