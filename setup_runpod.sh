#!/bin/bash
set -e

echo "============================================"
echo "  FastV 复现环境安装 (RunPod A100)"
echo "============================================"

# 1. 系统依赖
echo "[1/5] 安装系统依赖..."
apt-get update && apt-get install -y git wget unzip

# 2. 创建 conda 环境
echo "[2/5] 创建 conda 环境..."
if ! command -v conda &> /dev/null; then
    echo "Conda 未安装，使用 pip 直接安装..."
    pip install --upgrade pip
else
    conda create -n fastv python=3.10 -y
    conda activate fastv
fi

# 3. 安装 PyTorch (CUDA 11.8)
echo "[3/5] 安装 PyTorch..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装核心依赖
echo "[4/5] 安装核心依赖..."
pip install transformers==4.37.2 \
    accelerate==0.25.0 \
    bitsandbytes==0.41.3 \
    peft==0.7.1 \
    datasets==2.16.1 \
    sentencepiece==0.1.99 \
    protobuf==4.25.1 \
    pillow==10.2.0 \
    requests==2.31.0 \
    tqdm==4.66.1 \
    einops==0.7.0 \
    timm==0.9.12 \
    openai==1.6.1 \
    httpx==0.26.0 \
    gradio==4.12.0 \
    matplotlib==3.8.2 \
    seaborn==0.13.1 \
    scipy==1.12.0 \
    scikit-learn==1.4.0

# 5. 安装 LLaVA
echo "[5/5] 安装 LLaVA..."
if [ ! -d "third_party/LLaVA" ]; then
    mkdir -p third_party
    cd third_party
    git clone https://github.com/haotian-liu/LLaVA.git
    cd LLaVA
    pip install -e .
    cd ../..
else
    echo "LLaVA 已存在，跳过..."
fi

# 6. 安装评测工具
pip install pycocoevalcap lmms-eval

echo "============================================"
echo "  安装完成！"
echo "  运行 python demo_fastv.py 验证安装"
echo "============================================"
