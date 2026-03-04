#!/bin/bash
set -e

echo "============================================"
echo "  FastV 复现环境安装 (RunPod A100)"
echo "============================================"

# 1. 系统依赖
echo "[1/6] 安装系统依赖..."
apt-get update && apt-get install -y git wget unzip

# 2. 创建隔离的 venv（避免和 RunPod 预装包冲突）
echo "[2/6] 创建 Python 虚拟环境..."
VENV_DIR="$(pwd)/venv"
python3 -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"

# 3. 先装 PyTorch（检测 RunPod 已有的就跳过）
echo "[3/6] 检查 PyTorch..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null && echo "  PyTorch 已存在，跳过" || {
    echo "  安装 PyTorch..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
}

# 4. 安装 LLaVA 兼容的固定版本依赖（关键：版本必须匹配）
echo "[4/6] 安装核心依赖 (LLaVA 兼容版本)..."
pip install \
    "transformers==4.37.2" \
    "tokenizers==0.15.1" \
    "accelerate==0.21.0" \
    "bitsandbytes==0.41.3" \
    "peft==0.7.1" \
    "datasets==2.16.1" \
    "sentencepiece==0.1.99" \
    "protobuf==4.25.1" \
    "pillow>=10.0.0" \
    "requests>=2.28.0" \
    "tqdm>=4.60.0" \
    "einops>=0.7.0" \
    "timm==0.9.12" \
    "httpx>=0.25.0" \
    "matplotlib>=3.7.0" \
    "seaborn>=0.13.0" \
    "scipy>=1.10.0" \
    "scikit-learn>=1.3.0" \
    "huggingface_hub>=0.19.0"

# 5. 安装 LLaVA（从源码，不让它覆盖我们的版本）
echo "[5/6] 安装 LLaVA..."
if [ ! -d "third_party/LLaVA" ]; then
    mkdir -p third_party
    cd third_party
    git clone https://github.com/haotian-liu/LLaVA.git
    cd LLaVA
    pip install -e ".[train]" --no-deps
    # 手动装 LLaVA 缺的依赖（但跳过会冲突的）
    pip install shortuuid fastapi uvicorn markdown2 gradio==3.35.2 \
        gradio_client==0.2.9 pydantic httpcore anyio \
        open_clip_torch wavedrom 2>/dev/null || true
    cd ../..
else
    echo "  LLaVA 已存在，跳过..."
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
    print(f'  显存:         {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB')

import transformers
print(f'  Transformers: {transformers.__version__}')

import accelerate
print(f'  Accelerate:   {accelerate.__version__}')

try:
    from llava.model.builder import load_pretrained_model
    print(f'  LLaVA:        OK')
except Exception as e:
    print(f'  LLaVA:        {e}')
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
echo "    python demo_fastv.py"
echo ""
