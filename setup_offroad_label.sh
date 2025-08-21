#!/bin/bash

# offroad_label conda 환경 설정 및 viewer.py 실행 스크립트

echo "=== SNU Mountain Dataset Viewer Setup ==="

# conda가 설치되어 있는지 확인
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# 환경 이름
ENV_NAME="offroad_label"

echo "Creating conda environment: $ENV_NAME"

# 기존 환경이 있으면 제거
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Removing existing environment: $ENV_NAME"
    conda env remove -n $ENV_NAME -y
fi

# 새 환경 생성 (Python 3.9 사용)
conda create -n $ENV_NAME python=3.9 -y

if [ $? -ne 0 ]; then
    echo "Error: Failed to create conda environment"
    exit 1
fi

echo "Environment created successfully!"

# 환경 활성화
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

echo "Installing required packages..."

# 기본 패키지들 설치
conda install -c conda-forge -y \
    numpy \
    opencv \
    matplotlib \
    pyyaml \
    pillow

# PyQt6 설치 (GUI용)
conda install -c conda-forge -y pyqt

# 추가 패키지들 pip로 설치
pip install \
    pathlib \
    typing-extensions

echo "Package installation completed!"

# 환경 정보 출력
echo ""
echo "=== Environment Information ==="
echo "Environment name: $ENV_NAME"
echo "Python version: $(python --version)"
echo "Installed packages:"
conda list | grep -E "(numpy|opencv|matplotlib|pyyaml|pyqt|pillow)"


