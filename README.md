# BAGLE: Brain Analysis with Graph Learning and Enhancement

이 프로젝트는 뇌 네트워크 데이터를 위한 그래프 신경망 기반 분류 및 데이터 증강 시스템입니다.

## 프로젝트 구조

```
bagle/
├── classifier/          # 분류 모델 및 실험
│   ├── main.py         # 메인 실험 실행 스크립트
│   ├── utils/          # 유틸리티 함수들
│   │   ├── loader.py   # 데이터 로더 및 adjacency 처리
│   │   └── train.py    # 훈련 함수들
│   ├── models/         # 모델 정의
│   └── scripts/        # 실험 스크립트들
├── generator/          # 데이터 생성 및 증강
│   ├── main.py         # SMOTE 기반 증강
│   └── utils/          # 생성 관련 유틸리티
├── data/              # 데이터셋
│   ├── ADNI_CT/       # ADNI CT 데이터
│   ├── ADNI_FDG/      # ADNI FDG 데이터
│   ├── ADNI_Amy/      # ADNI Amyloid 데이터
│   └── ADNI_Tau/      # ADNI Tau 데이터
└── summary/           # 실험 결과 요약
    └── experiment_summary.xlsx
```

## 환경 설정

### 1. Conda 환경 생성
```bash
conda create -n bagle_env python=3.10 -y
conda activate bagle_env
```

### 2. PyTorch 및 CUDA 설치
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

### 3. 필요한 패키지 설치
```bash
pip install torch-geometric wandb einops openpyxl imbalanced-learn
conda install numpy scipy pandas scikit-learn matplotlib seaborn -y
```

## 주요 기능

### 1. 데이터 증강 (SMOTE)
- **min**: 가장 적은 클래스 수만큼 증강
- **full**: 가장 많은 클래스 수만큼 증강

### 2. Adjacency Assignment
- **random**: 기존 샘플에서 랜덤하게 adjacency 할당
- **average**: 같은 클래스의 평균 adjacency 계산하여 할당

### 3. 지원 모델
- **MLP**: 다층 퍼셉트론 (adjacency 사용 안함)
- **MLP-A**: MLP + Adjacency features
- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network

## 실험 실행

### 단일 실험
```bash
cd classifier
python main.py --data adni_ct --model gat --augmentation SMOTE --aug_level full --adj_assignment average --device 0
```

### 배치 실험
```bash
chmod +x run_adni_ct_smote_experiments.sh
./run_adni_ct_smote_experiments.sh
```

## 매개변수

- `--data`: 데이터셋 (adni_ct, adni_fdg, adni_amy, adni_tau)
- `--model`: 모델 (mlp, mlp-a, gcn, gat)
- `--augmentation`: 증강 방법 (NoAug, SMOTE)
- `--aug_level`: 증강 레벨 (min, full)
- `--adj_assignment`: Adjacency 할당 방법 (random, average)
- `--device`: GPU 번호

## 결과 확인

실험 결과는 `summary/experiment_summary.xlsx`에 자동으로 저장됩니다.

컬럼 구조:
- Date, Data, Model, Augmentation, Adj_Assignment
- Avg_Accuracy, Avg_Precision, Avg_Recall, Avg_F1
- Avg_AUROC, Avg_Macro_F1, Avg_Macro_AUROC, Notes

## 참고사항

- MLP 모델은 adjacency assignment 옵션을 사용하지 않습니다
- GPU 메모리 사용량이 높으므로 실험 간 충분한 간격을 두세요
- 5-fold cross validation으로 실행됩니다
