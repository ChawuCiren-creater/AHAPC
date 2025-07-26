# AHAPC: Extremophilic Protein Classification Framework

## Overview

**AHAPC** is a unified deep learning framework for classifying extremophilic proteins, including:
- **Acidophilic**, **Halophilic**, and **Alkaliphilic** types.

It integrates protein language model embeddings (ESM2, ProtT5, ProtBert), handcrafted descriptors (AAC, PSSM), and SHAP-based feature selection for both binary and multiclass classification.

---

## Models

| Type        | Architecture | Full Dim | Reduced Dim |
|-------------|--------------|----------|-------------|
| Acidophilic | CNN          | 2704     | 315         |
| Halophilic  | BiLSTM       | 2324     | 495         |
| Alkaliphilic| GRU          | 2324     | 405         |
| Multiclass  | BiLSTM×3     | 3768     | 1325        |

---

## Metrics

- Accuracy (ACC), MCC, F1, AUC
- Sensitivity / Specificity (Sn/Sp)
- Balanced Accuracy (BA), FPR/FNR

---

## Features

- **Embeddings**: ESM2, ProtT5, ProtBert  
- **Descriptors**: AAC, PSSM_AAC, PSSM_DPC  
- **Reduced Features**: Selected via SHAP + XGBoost

---

## Contact

**Author**:Lu Mingxian
**Email**:cirenchawu161@gmail.com
**Link to the full model file**:https://drive.google.com/file/d/1KxA6ZZRbgfYpiGopxs3qSzPCMgic1H19/view?usp=drive_link
