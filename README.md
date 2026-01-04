[한국어 버전은 아래에 있으며, 영어 원문은 문서 하단에 보존되어 있습니다.]
[The Korean version is below, and the original English version is preserved at the bottom of the document.]

---

## TransPolymer (트랜스폴리머) ##

#### npj Computational Materials [[논문]](https://www.nature.com/articles/s41524-023-01016-5) [[arXiv]](https://arxiv.org/abs/2209.01307) [[PDF]](https://www.nature.com/articles/s41524-023-01016-5.pdf) </br>
[Changwen Xu](https://changwenxu98.github.io/), [Yuyang Wang](https://yuyangw.github.io/), [Amir Barati Farimani](https://www.meche.engineering.cmu.edu/directory/bios/barati-farimani-amir.html) </br>
카네기 멜런 대학교 (Carnegie Mellon University) </br>

<img src="figs/pipeline.png" width="500">

이 레포지토리는 <strong><em>TransPolymer</em></strong>의 공식 구현체입니다: ["TransPolymer: a Transformer-based language model for polymer property predictions"](https://www.nature.com/articles/s41524-023-01016-5). 이 연구에서는 트랜스포머 기반의 언어 모델을 활용하여, 라벨이 없는 대규모 데이터셋(약 500만 개의 고분자 시나리오)에 대해 자기 지도 학습(Masked Language Modeling)을 수행하고, 이를 바탕으로 하위 작업(Downstream tasks)에서 고분자 물성을 정확하고 효율적으로 예측하는 모델을 제안합니다. 연구에 도움이 되었다면 아래를 인용해 주세요:

```
@article{xu2023transpolymer,
  title={TransPolymer: a Transformer-based language model for polymer property predictions},
  author={Xu, Changwen and Wang, Yuyang and Barati Farimani, Amir},
  journal={npj Computational Materials},
  volume={9},
  number={1},
  pages={64},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

> [!TIP]
> **MacOS (Apple Silicon) 지원 안내:**
> 본 프로젝트는 M1/M2/M3 등 Apple Silicon 칩셋의 **MPS (Metal Performance Shaders)** 가속을 완벽하게 지원합니다. 별도의 설정 없이 자동으로 GPU를 감지하여 사용합니다.

## 시작하기

### 빠른 시작 (MacOS/Apple Silicon 데모용)
  
  초기 환경 설정을 위해 **원클릭 설치 스크립트**를 제공합니다.
  터미널에서 아래 명령어를 실행하면 파이썬 가상환경 생성 및 라이브러리 설치가 자동으로 진행됩니다.
  
  ```bash
  bash setup_mac.sh
  ```
  
  설치 완료 후:
  ```bash

  # 가상환경 생성 
  bash setup_mac.sh

  # 가상환경 활성화
  source .venv/bin/activate
  
  # 모델 학습 시연
  python Pretrain.py --config configs/config.yaml
  
  # 파인튜닝 (기본)
  bash run_finetune.sh
  
  # Tg-Boost (심화 학습)
  bash run_tg_boost.sh

  # 데모 실행 (스트림릿 앱)
  streamlit run app.py
  
  ```
  
  ### 설치 방법 (수동)

Conda 환경을 설정하고 github 레포지토리를 클론합니다.

```bash
# 새로운 환경 생성
$ conda create --name TransPolymer python=3.9
$ conda activate TransPolymer

# 필수 패키지 설치
$ conda install pytorch==1.12.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
$ pip install transformers==4.20.1
$ pip install PyYAML==6.0
$ pip install fairscale==0.4.6
$ conda install -c conda-forge rdkit=2022.3.4
$ conda install -c conda-forge scikit-learn==0.24.2
$ conda install -c conda-forge tensorboard==2.9.1
$ conda install -c conda-forge torchmetrics==0.9.2
$ conda install -c conda-forge packaging==21.0
$ conda install -c conda-forge seaborn==0.11.2
$ conda install -c conda-forge opentsne==0.6.2

# 소스 코드 클론
$ git clone https://github.com/ChangwenXu98/TransPolymer.git
$ cd TransPolymer
```

> [!NOTE]
> **협업자 참고 사항:** PyTorch 등에서 사용하는 대용량 바이너리 파일(`*.dll`, `*.lib` 등)은 레포지토리 경량화를 위해 제외되었습니다. 위의 설치 단계를 따라 로컬 환경을 구축하면 필요한 라이브러리와 함께 자동으로 설치됩니다.


### 데이터셋

사전 학습(Pretraining) 데이터셋은 ["PI1M: A Benchmark Database for Polymer Informatics"](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00726) 논문의 데이터를 채택했습니다. 데이터 증강(Data augmentation)을 통해 각 시나리오를 5개로 늘렸으며, 더 작은 크기의 데이터셋은 PI1M에서 무작위로 추출하여 구성했습니다.

하위 작업(Downstream tasks)에는 고분자 전해질 전도도, 밴드 갭, 전자 친화도, 이온화 에너지, 결정화 경향, 유전 상수, 굴절률, p형 고분자 OPV 전력 변환 효율 등 10가지 데이터셋이 사용되었습니다. 데이터 처리 및 증강은 파인튜닝 단계 전에 수행됩니다. 원본 데이터셋 출처는 다음과 같습니다:

PE-I: ["AI-Assisted Exploration of Superionic Glass-Type Li(+) Conductors with Aromatic Structures"](https://pubs.acs.org/doi/10.1021/jacs.9b11442)

PE-II: ["Database Creation, Visualization, and Statistical Learning for Polymer Li+-Electrolyte Design"](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.0c04767)

---

### 추가된 핵심 데이터셋
본 프로젝트의 발전을 위해 추가된 핵심 데이터셋입니다:

1. **NeurIPS - Open Polymer Prediction 2025**
   - **출처**: [Kaggle Competition](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data)
   - **물성**: Tg, FFV, Tc, Density, Rg (통합 멀티태스크 학습의 기반)
   - **설명**: MD 시뮬레이션 기반의 고품질 고분자 물성 데이터셋입니다.

2. **POINT2-Dataset (Polymer Property Tg)**
   - **출처**: [Kaggle Dataset](https://www.kaggle.com/datasets/fridaycode/point2-dataset-polymer-property-tg-smiles), [GITHUB](https://github.com/Jiaxin-Xu/POINT2)
   - **물성**: 유리 전이 온도 (Tg) - 약 7,210개 샘플
   - **설명**: 접착제 개발의 정밀도를 높이기 위한 대용량 Tg 전용 데이터셋입니다.

---

### 토큰화 (Tokenization)
`PolymerSmilesTokenization.py`는 [huggingface](https://github.com/huggingface/transformers/tree/v4.21.2)의 RobertaTokenizer를 기반으로 하며, 화학 구조 인식을 위해 특별히 설계된 정규 표현식을 사용합니다.

### 체크포인트 (Checkpoints)
사전 학습된 모델은 `ckpt` 폴더에서 확인할 수 있습니다.

### [신규] 프로젝트 구조
- **`app.py`**: 통합 6개 물성 예측 Streamlit 대시보드.
- **`Downstream.py`**: 멀티태스크 및 단일 태스크 파인튜닝 핵심 로직.
- **`configs/`**: YAML 설정 파일 중앙 저장소.
- **`utils/`**: 데이터 처리 및 분석 도구 (t-SNE, Attention 맵핑 등).
- **`maintenance/`**: 환경 설정 및 진단 스크립트.
- **`ckpt/` & `data/`**: 모델 가중치 및 데이터셋 저장소.

## 모델 실행하기

### 사전 학습 (Pretraining)
설정 파일은 `configs/config.yaml`에서 확인할 수 있습니다.
```bash
# MacOS / Single Process
$ python Pretrain.py --config configs/config.yaml

# Linux / Multi-GPU (Legacy)
$ python -m torch.distributed.launch --nproc_per_node=2 Pretrain.py --config configs/config.yaml
```
빠른 학습을 위해 *DistributedDataParallel*이 사용됩니다. 학습 결과는 `ckpt/pretrain.pt`에 저장됩니다.

### 파인튜닝 (Finetuning)
기본 설정 파일은 `configs/config_finetune.yaml`입니다.
```bash
$ python Downstream.py --config configs/config_finetune.yaml
```
멀티태스크(5개 물성 통합) 학습 시:
```bash
$ python Downstream.py --config configs/config_finetune_Multi.yaml
```

## 시각화

### Attention 시각화
해석 가능성을 위한 Attention 스코어 시각화 설정은 `configs/config_attention.yaml`에 있습니다.
```bash
$ python utils/Attention_vis.py --config configs/config_attention.yaml
```

### t-SNE 시각화
화학적 공간 분포 시각화 설정은 `configs/config_tSNE.yaml`에 있습니다.
```bash
$ python utils/tSNE.py --config configs/config_tSNE.yaml
```

---

## 고분자 물성 통합 대시보드 (Streamlit UI)

본 레포지토리의 핵심 기능인 **통합 6종 물성 예측 시스템**은 사용자가 별도의 코딩 없이 웹 인터페이스를 통해 고분자의 성질을 즉시 분석할 수 있도록 설계되었습니다.

### 주요 특징
- **동시 예측**: 하나의 SMILES 입력으로 6가지 핵심 물성(Tg, FFV, Tc, Density, Rg, Conductivity)을 한 번에 예측합니다.
- **Attention 시각화**: 인공지능이 화학 구조의 어느 부분에 주목하여 수치를 계산했는지 인터랙티브 히트맵으로 보여줍니다.
- **다국어 지원**: 한국어와 영어를 모두 지원하며, 물리 단위 정보가 포함되어 있습니다.

### 실행 방법
1. 터미널에서 다음 명령어를 실행합니다:
   ```bash
   $ streamlit run app.py
1.  터미널에서 다음 명령어를 실행합니다:
    ```bash
    $ streamlit run app.py
    ```
2.  웹 브라우저가 열리면 SMILES 입력칸에 분석하려는 고분자의 구조식을 입력하고 "모든 물성 예측하기" 버튼을 누릅니다.

### 예측 가능한 물성 목록
| 물성명 (KR) | 영문명 & 기호 | 단위 |
| :--- | :--- | :--- |
| **유리 전이 온도** | Glass Transition (Tg) | °C | (v1.1 Boost 적용) |
| **자유 부피비** | Free Volume (FFV) | - |
| **열전도도** | Thermal Cond (Tc) | W/mK |
| **밀도** | Density | g/cm³ |
| **회전 반경** | Radius of Gyration (Rg) | Å |
| **이온 전도도** | Ionic Conductivity | S/cm |

---

## 접착제/점착제 특화 Tg-Boost 모델

산업계(Adhesive/PSA) 요구사항을 반영하여, **POINT2 데이터셋(7,210개)**을 활용한 고정밀 Tg 예측 모델 학습 환경을 구축해 두었습니다. 이는 기존 학술 데이터셋보다 약 10배 많은 양을 학습하여 훨씬 정밀한 구조-물성 상관관계를 도출합니다.

### Tg-Boost 학습 실행
제공된 배치 파일을 통해 간편하게 학습을 시작할 수 있습니다:
```bash
$ .\run_tg_boost.bat
```
*(결과물은 `ckpt/model_multi_boost_best.pt`로 저장됩니다.)*

---

## English Version (Original) ##

## TransPolymer ##

#### npj Computational Materials [[Paper]](https://www.nature.com/articles/s41524-023-01016-5) [[arXiv]](https://arxiv.org/abs/2209.01307) [[PDF]](https://www.nature.com/articles/s41524-023-01016-5.pdf) </br>
[Changwen Xu](https://changwenxu98.github.io/), [Yuyang Wang](https://yuyangw.github.io/), [Amir Barati Farimani](https://www.meche.engineering.cmu.edu/directory/bios/barati-farimani-amir.html) </br>
Carnegie Mellon University </br>

<img src="figs/pipeline.png" width="500">

This is the official implementation of <strong><em>TransPolymer</em></strong>: ["TransPolymer: a Transformer-based language model for polymer property predictions"](https://www.nature.com/articles/s41524-023-01016-5). In this work, we introduce TransPolymer, a Transformer-based language model, for representation learning of polymer sequences by pretraining on a large unlabeled dataset (~5M polymer sequences) via self-supervised masked language modeling and making accurate and efficient predictions of polymer properties in downstream tasks by finetuning. If you find our work useful in your research, please cite:
```
@article{xu2023transpolymer,
  title={TransPolymer: a Transformer-based language model for polymer property predictions},
  author={Xu, Changwen and Wang, Yuyang and Barati Farimani, Amir},
  journal={npj Computational Materials},
  volume={9},
  number={1},
  pages={64},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## Getting Started

### 🔥 Quick Start (MacOS/Apple Silicon Demo)

For a quick setup on a clean MacBook (Apple Silicon), use the provided **one-click setup script**.
Run the following command in your terminal to automatically create a virtual environment and install all dependencies.

```bash
bash setup_mac.sh
```

After setup:
```bash
# Activate virtual environment
source .venv/bin/activate

# Run Demo App
streamlit run app.py

# Or Run Training Demo
python Pretrain.py --config configs/config.yaml

# Run Finetuning
bash run_finetune.sh

# Run Tg-Boost
bash run_tg_boost.sh
```

### Installation

Set up conda environment and clone the github repo

```
# create a new environment
$ conda create --name TransPolymer python=3.9
$ conda activate TransPolymer

# install requirements
$ conda install pytorch==1.12.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch
$ pip install transformers==4.20.1
$ pip install PyYAML==6.0
$ pip install fairscale==0.4.6
$ conda install -c conda-forge rdkit=2022.3.4
$ conda install -c conda-forge scikit-learn==0.24.2
$ conda install -c conda-forge tensorboard==2.9.1
$ conda install -c conda-forge torchmetrics==0.9.2
$ conda install -c conda-forge packaging==21.0
$ conda install -c conda-forge seaborn==0.11.2
$ conda install -c conda-forge opentsne==0.6.2

# clone the source code of TransPolymer
$ git clone https://github.com/ChangwenXu98/TransPolymer.git
$ cd TransPolymer
```

> [!NOTE]
> **For Collaborators:** Large binary files (e.g., `*.dll`, `*.lib` from PyTorch) are excluded from the repository to keep it lightweight. Please follow the installation steps above to set up your local environment. These files will be automatically installed with the required libraries.


### Dataset

The pretraining dataset is adopted from the paper ["PI1M: A Benchmark Database for Polymer Informatics"](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00726). Data augmentation is applied by augmenting each sequence to five. Pretraining data with smaller sizes are obtained by randomly picking up data entries from PI1M dataset.

Ten datasets, concerning different polymer properties including polymer electrolyte conductivity, band gap, electron affinity, ionization energy, crystallization tendency, dielectric constant, refractive index, and p-type polymer OPV power conversion efficiency, are used for downstream tasks. Data processing and augmentation are implemented before usage in the finetuning stage. The original datasets and their sources are listed below:

PE-I: ["AI-Assisted Exploration of Superionic Glass-Type Li(+) Conductors with Aromatic Structures"](https://pubs.acs.org/doi/10.1021/jacs.9b11442)

PE-II: ["Database Creation, Visualization, and Statistical Learning for Polymer Li+-Electrolyte Design"](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.0c04767)

Egc, Egb, Eea, Ei, Xc, EPS, Nc: ["Polymer informatics with multi-task learning"](https://www.sciencedirect.com/science/article/pii/S2666389921000581)

---

### [User-Provided Primary Datasets]
Key datasets provided by the user for project enhancement:

1. **NeurIPS - Open Polymer Prediction 2025**
   - **Source**: [Kaggle Competition](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data)
   - **Properties**: Tg, FFV, Tc, Density, Rg (Core Multi-Task data)
2. **POINT2-Dataset (Polymer Property Tg)**
   - **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/fridaycode/point2-dataset-polymer-property-tg-smiles), [GITHUB](https://github.com/Jiaxin-Xu/POINT2)
   - **Properties**: Glass Transition Temperature (Tg) - ~7,210 samples

---

### Tokenization
`PolymerSmilesTokenization.py` is adapted from RobertaTokenizer from [huggingface](https://github.com/huggingface/transformers/tree/v4.21.2) with a specially designed regular expression for tokenization with chemical awareness.

### Checkpoints
Pretrained model can be found in `ckpt` folder.

### [NEW] Project Structure
- **`app.py`**: Unified 6-Property Streamlit Dashboard.
- **`Downstream.py`**: Core logic for multi-task and single-task fine-tuning.
- **`configs/`**: Centralized storage for YAML configurations.
- **`utils/`**: Data processing and analytical tools (t-SNE, Attention mapping).
- **`maintenance/`**: Environment setup and diagnostic scripts.
- **`ckpt/` & `data/`**: Model weights and dataset storage.

## Run the Model

### Pretraining
To pretrain TransPolymer, the configuration can be found in `configs/config.yaml`.
```bash
$ python -m torch.distributed.launch --nproc_per_node=2 Pretrain.py --config configs/config.yaml
```
*DistributedDataParallel* is used for faster pretraining. The pretrained model can be found in `ckpt/pretrain.pt`

### Finetuning
To finetune the pretrained TransPolymer, find configurations in `configs/config_finetune.yaml`.
```bash
$ python Downstream.py --config configs/config_finetune.yaml
```
For Multi-task (5 properties):
```bash
$ python Downstream.py --config configs/config_finetune_Multi.yaml
```

### Attention Visualization
To visualize attention scores, configurations are in `configs/config_attention.yaml`.
```bash
$ python utils/Attention_vis.py --config configs/config_attention.yaml
```

### t-SNE Visualization
To visualize the chemical space, configurations are in `configs/config_tSNE.yaml`.
```bash
$ python utils/tSNE.py --config configs/config_tSNE.yaml
```

## Acknowledgement
- PyTorch implementation of Transformer: [https://github.com/huggingface/transformers.git](https://github.com/huggingface/transformers.git)
