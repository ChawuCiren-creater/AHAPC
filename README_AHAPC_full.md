# AHAPC: Extremophilic Protein Classification Framework

## 🔍 Overview

**AHAPC** is a unified deep learning framework for classifying extremophilic proteins, including:
- **Acidophilic**, **Halophilic**, and **Alkaliphilic** types.

It integrates protein language model embeddings (ESM2, ProtT5, ProtBert), handcrafted descriptors (AAC, PSSM), and SHAP-based feature selection for both binary and multiclass classification.

---

## ⚙️ Models

| Type        | Architecture | Full Dim | Reduced Dim |
|-------------|--------------|----------|-------------|
| Acidophilic | CNN          | 2704     | 315         |
| Halophilic  | BiLSTM       | 2324     | 495         |
| Alkaliphilic| GRU          | 2324     | 405         |
| Multiclass  | BiLSTM×3     | 3768     | 1325        |

---

## 🚀 Quick Start

Install dependencies:
```bash
pip install torch pandas scikit-learn shap xgboost matplotlib
```

### 🔬 Run Binary Classification Tests

**Acidophilic (Full & Reduced):**
```bash
cd model/Acidophilic/Full-Fusion
python test.py

cd ../Reduced-Fusion(315D)
python test.py
```

**Halophilic (Full & Reduced):**
```bash
cd model/Halophilic/Full-Fusion
python test.py

cd ../Reduced-Fusion(495D)
python test.py
```

**Alkaliphilic (Full & Reduced):**
```bash
cd model/Alkaliphilic/Full-Fusion
python test.py

cd ../Reduced-Fusion(405D)
python test.py
```

### 🔄 Run Multiclass Classification

**Full Feature (3768D):**
```bash
cd model/Three_class/Full-Fusion
python test.py
```

**Reduced Feature (1325D):**
```bash
cd model/Three_class/Reduced-Fusion
python test.py
```

---

### 📈 SHAP Feature Visualization

**Single-class (e.g., Acidophilic):**
```bash
cd model/Acidophilic/XGBoost
python XGBoost_figure.py
```

**Multiclass:**
```bash
cd model/Three_class/XGBoost
python test.py
```

---

## 📊 Metrics

- Accuracy (ACC), MCC, F1, AUC
- Sensitivity / Specificity (Sn/Sp)
- Balanced Accuracy (BA), FPR/FNR

---

## 📁 Features

- **Embeddings**: ESM2, ProtT5, ProtBert  
- **Descriptors**: AAC, PSSM_AAC, PSSM_DPC  
- **Reduced Features**: Selected via SHAP + XGBoost

---

## 📄 Citation

> Mingxian, L. et al. (2025). AHAPC: Cross-Model Fusion for Extremophilic Protein Classification.!

---

## 🛠 Contact

**Author**: Mingxian Lu

**Email**: cirenchawu161@gmail.com

## 📂 Data & Model Availability

This repository only contains testing scripts and model definitions for reproducibility purposes.

The following contents are **not included**:
- Original or preprocessed datasets;
- Embedding features (e.g., ESM, ProtT5, PSSM, etc.);
- Trained model checkpoints.

To access these resources, please contact the authors at: cirenchawu161@gmail.com