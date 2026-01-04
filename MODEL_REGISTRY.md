# 🧪 TransPolymer Model Registry

This document tracks the performance and metadata of trained models and data assets specifically optimized for adhesive/PSA development.

## 0. Unified 6-Property Dashboard (Latest UI)
- **Status**: **Operational**
- **Architecture**: Dual-Model Runtime integration.
- **Combined Properties**: `['Tg', 'FFV', 'Tc', 'Density', 'Rg', 'Conductivity']`
- **Features**: Single SMILES input results in simultaneous prediction from Multi-Task Model (v1.0) and Legacy Conductivity Model.

## 1. Unified Multi-Task Model (v1.1 - Tg-Boost)
`POINT2`와 `NeurIPS 2025` 데이터를 통합하여 유리 전이 온도(Tg) 예측 성능을 극대화한 최신 모델입니다.

- **Checkpoint**: `ckpt/model_multi_boost_best.pt`
- **Scaler**: `ckpt/scaler_multi_boost.joblib`
- **Target Properties**: `['Tg', 'FFV', 'Tc', 'Density', 'Rg']`
- **Performance (Overall Test RMSE: 0.4584)**:
  | Property | Test R² | Test RMSE | Unit |
  | :--- | :--- | :--- | :--- |
  | **Tg** | **0.7586** | 53.2650 | °C |
  | **FFV** | 0.7789 | 0.0126 | - |
  | **Tc** | 0.6849 | 0.0493 | W/mK |
  | **Density** | 0.4976 | 0.0997 | g/cm³ |
  | **Rg** | 0.5270 | 3.3938 | Å |

---

## 2. Ionic Conductivity Model (v1.0)
리튬 이온 전도체 전용 데이터로 학습된 예측 모델입니다.

- **Checkpoint**: `ckpt/model_conductivity_best.pt`
- **Test RMSE**: 0.0035 (Standardized)
- **Unit**: **S/cm (Ionic Conductivity)**
- **Note**: 일반 고분자의 절연 특성이 아닌 이온 전도 성능을 예측합니다.

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
