# 🧪 TransPolymer Model Registry

This document tracks the performance and metadata of trained models and data assets specifically optimized for adhesive/PSA development.

## 0. Unified 6-Property Dashboard (Latest UI)
- **Status**: **Operational**
- **Architecture**: Dual-Model Runtime integration.
- **Combined Properties**: `['Tg', 'FFV', 'Tc', 'Density', 'Rg', 'Conductivity']`
- **Features**: Single SMILES input results in simultaneous prediction from Multi-Task Model (v1.0) and Legacy Conductivity Model.

## 1. Unified Multi-Task Model (v1.0)
- **File**: `ckpt/model_multi_best.pt`
- **Scaler**: `ckpt/scaler_multi.joblib`
- **Target Properties**: `['Tg', 'FFV', 'Tc', 'Density', 'Rg']`
- **Architecture**: RoBERTa-base + Multi-output MLP Regressor
- **Training Epochs**: 20
- **Overall Performance**: Avg Test R² = **0.7500**

### Property-wise Performance Dashboard
| Property | Test R² | Test RMSE | Unit |
| :--- | :--- | :--- | :--- |
| **Tg** | 0.6484 | 60.4664 | °C |
| **FFV** | 0.8235 | 0.0113 | - |
| **Tc** | 0.7696 | 0.0422 | W/mK |
| **Density** | 0.8851 | 0.0477 | g/cm³ |
| **Rg** | 0.6235 | 3.0278 | Å |

---

## 2. Single-Task Conductivity Model (Legacy/Archived)
- **File**: `ckpt/model_conductivity_best.pt`
- **Scaler**: `ckpt/scaler_conductivity.joblib`
- **Target Property**: `Conductivity`
- **Test R²**: 0.3436
- **Test RMSE**: 0.0035

---

- `run_multi_train.bat`: Ready-to-use batch script for retraining.

## 4. [NEW] POINT2 Tg-Boost Assets (Industrial Grade)
The following assets were prepared to achieve high-precision Glass Transition Temperature prediction for adhesive design.
- **Dataset**: `data/merged/train_Multi_POINT2.csv` (7,210 samples)
- **Config**: `config_finetune_Multi_Boost.yaml`
- **Utility**: `merge_point2.py`
- **Script**: `run_tg_boost.bat`

## 5. User-Provided Core Datasets
Official data assets provided for high-precision industrial optimization.
- **NeurIPS 2025 (Open Polymer Prediction)**: 5-property MD simulation data. Used for MTL v1.0.
- **POINT2-Dataset**: 7,210 high-quality Tg samples. Used for industrial-grade Tg-Boost training.
