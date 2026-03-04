#!/bin/bash
set -e

echo "============================================"
echo "  FastV 复现环境安装 (RunPod A100)"
echo "============================================"

# 1. 系统依赖
echo "[1/5] 安装系统依赖..."
apt-get update && apt-get install -y git wget unzip

# 2. 创建 venv，继承系统包（保留 RunPod 已有的 PyTorch + CUDA + numpy）
echo "[2/5] 创建 Python 虚拟环境..."
VENV_DIR="$(pwd)/venv"
if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
fi
python3 -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"

# 验证系统 PyTorch 可用
echo "  检查系统 PyTorch..."
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 3. 强制安装 LLaVA 要求的精确版本（覆盖系统的新版本）
echo "[3/6] 安装 LLaVA 兼容的依赖版本..."
pip install --force-reinstall --no-deps \
    "transformers==4.37.2" \
    "tokenizers==0.15.1" \
    "accelerate==0.21.0"

pip install \
    "peft==0.7.1" \
    "datasets==2.16.1" \
    "sentencepiece==0.1.99" \
    "protobuf==4.25.1" \
    "einops==0.6.1" \
    "timm==0.6.13" \
    "httpx==0.24.0" \
    "scikit-learn==1.2.2" \
    "matplotlib>=3.7.0" \
    "seaborn>=0.13.0" \
    "huggingface_hub>=0.19.0" \
    "numpy<2.0.0"

# 4. 用空壳 bitsandbytes 覆盖系统坏掉的版本
#    问题: 系统 bnb 0.41.3 缺 libcusparse.so.11，无法卸载 (在 venv 外)
#    解决: 在 venv site-packages 里创建一个空壳 bitsandbytes 包
#         venv 的包优先级高于系统包，所以 import 会找到我们的空壳
echo "[4/6] 创建 bitsandbytes 空壳包 (替代系统坏掉的版本)..."
BNB_STUB="$VENV_DIR/lib/python3.11/site-packages/bitsandbytes"
mkdir -p "$BNB_STUB"
cat > "$BNB_STUB/__init__.py" << 'STUBEOF'
# Stub: shadows broken system bitsandbytes
# FastV only needs inference, no quantization needed
class _Stub:
    def __getattr__(self, name):
        return _Stub()
    def __call__(self, *args, **kwargs):
        return _Stub()
    def __bool__(self):
        return False

nn = _Stub()
functional = _Stub()
optim = _Stub()
STUBEOF
echo "  已创建空壳包: $BNB_STUB"

# 5. 安装 LLaVA
echo "[5/6] 安装 LLaVA..."
mkdir -p third_party
if [ ! -d "third_party/LLaVA" ]; then
    cd third_party
    git clone https://github.com/haotian-liu/LLaVA.git
    cd ..
fi
cd third_party/LLaVA
pip install -e ".[train]" --no-deps
# LLaVA 运行时的额外依赖（跳过 bitsandbytes）
pip install shortuuid fastapi uvicorn markdown2 \
    pydantic httpcore anyio open_clip_torch 2>/dev/null || true
cd ../..

# 6. 安装评测工具
echo "[6/6] 安装评测工具..."
pip install pycocoevalcap 2>/dev/null || true

# 验证安装
echo ""
echo "============================================"
echo "  验证安装..."
echo "============================================"
python3 << 'PYEOF'
import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
    print(f'  显存:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB')

import numpy as np
print(f'  NumPy:        {np.__version__}')

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

print()
print('  所有依赖验证通过!' if 'OK' else '')
PYEOF

echo ""
echo "============================================"
echo "  安装完成！"
echo "============================================"
echo ""
echo "  每次新开终端先激活环境:"
echo "    source venv/bin/activate"
echo ""
echo "  下载模型并运行 demo:"
echo "    bash scripts/download_model.sh"
echo "    python demo_fastv.py --model-path models/llava-v1.5-7b"
echo ""
