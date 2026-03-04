#!/bin/bash
set -e

echo "============================================"
echo "  FastV 复现环境安装 (RunPod A100)"
echo "============================================"

# 1. 系统依赖
echo "[1/6] 安装系统依赖..."
apt-get update && apt-get install -y git wget unzip

# 2. 创建纯净 venv（不继承系统包，彻底避免冲突）
echo "[2/6] 创建 Python 虚拟环境 (纯净隔离)..."
VENV_DIR="$(pwd)/venv"
if [ -d "$VENV_DIR" ]; then
    echo "  删除旧 venv..."
    rm -rf "$VENV_DIR"
fi
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# 3. 安装 PyTorch
echo "[3/6] 安装 PyTorch..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装所有依赖，精确匹配 LLaVA 1.2.2 要求的版本
echo "[4/6] 安装核心依赖 (精确匹配 LLaVA 要求)..."
pip install \
    "transformers==4.37.2" \
    "tokenizers==0.15.1" \
    "accelerate==0.21.0" \
    "bitsandbytes==0.41.3" \
    "peft==0.7.1" \
    "datasets==2.16.1" \
    "sentencepiece==0.1.99" \
    "protobuf==4.25.1" \
    "einops==0.6.1" \
    "timm==0.6.13" \
    "httpx==0.24.0" \
    "scikit-learn==1.2.2" \
    "pillow>=10.0.0" \
    "requests>=2.28.0" \
    "tqdm>=4.60.0" \
    "matplotlib>=3.7.0" \
    "seaborn>=0.13.0" \
    "scipy>=1.10.0" \
    "huggingface_hub>=0.19.0"

# 5. 安装 LLaVA
echo "[5/6] 安装 LLaVA..."
if [ ! -d "third_party/LLaVA" ]; then
    mkdir -p third_party
    cd third_party
    git clone https://github.com/haotian-liu/LLaVA.git
    cd LLaVA
    pip install -e ".[train]" --no-deps
    # 手动装 LLaVA 运行时需要的额外依赖
    pip install shortuuid fastapi uvicorn markdown2 \
        gradio==3.35.2 gradio_client==0.2.9 \
        pydantic httpcore anyio open_clip_torch wavedrom 2>/dev/null || true
    cd ../..
else
    echo "  LLaVA 已存在，跳过 clone..."
    cd third_party/LLaVA
    pip install -e ".[train]" --no-deps
    cd ../..
fi

# 6. 安装评测工具
echo "[6/6] 安装评测工具..."
pip install pycocoevalcap 2>/dev/null || true

# 验证安装
echo ""
echo "============================================"
echo "  验证安装..."
echo "============================================"
python3 -c "
import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
    print(f'  显存:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB')

import transformers
print(f'  Transformers: {transformers.__version__}')

import tokenizers
print(f'  Tokenizers:   {tokenizers.__version__}')

import accelerate
print(f'  Accelerate:   {accelerate.__version__}')

import einops
print(f'  Einops:       {einops.__version__}')

import timm
print(f'  Timm:         {timm.__version__}')

try:
    from llava.model.builder import load_pretrained_model
    print(f'  LLaVA:        OK')
except Exception as e:
    print(f'  LLaVA:        FAIL - {e}')
"

echo ""
echo "============================================"
echo "  安装完成！"
echo "============================================"
echo ""
echo "  每次新开终端需要先激活环境:"
echo "    source venv/bin/activate"
echo ""
echo "  然后运行:"
echo "    bash scripts/download_model.sh"
echo "    python demo_fastv.py --model-path models/llava-v1.5-7b"
echo ""
