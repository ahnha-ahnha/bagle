#!/bin/bash
# bagle_env 환경 설정 스크립트

echo "=== bagle_env 환경 설정 시작 ==="

# 1. Conda 환경 생성
echo "1. Conda 환경 생성..."
conda create -n bagle_env python=3.10 -y

# 2. 환경 활성화
echo "2. 환경 활성화..."
source activate bagle_env

# 3. CUDA 및 PyTorch 설치
echo "3. CUDA 및 PyTorch 설치..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# 4. 기본 패키지들 설치
echo "4. 기본 패키지들 설치..."
conda install numpy scipy pandas scikit-learn matplotlib seaborn ipython jupyter -y

# 5. PyTorch Geometric 설치
echo "5. PyTorch Geometric 설치..."
pip install torch-geometric

# 6. 추가 패키지들 설치
echo "6. 추가 패키지들 설치..."
pip install wandb einops openpyxl imbalanced-learn

# 7. 시스템 모니터링 도구 설치
echo "7. 시스템 모니터링 도구 설치..."
conda install gpustat nvidia-ml-py psutil -c conda-forge -y

# 8. 환경 확인
echo "8. 환경 확인..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo "=== bagle_env 환경 설정 완료 ==="
echo "환경 활성화 방법: conda activate bagle_env"
