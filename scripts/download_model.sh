#!/bin/bash
set -e

echo "============================================"
echo "  下载 LLaVA-1.5 模型"
echo "============================================"

# 默认 7B，加 --13b 参数下载 13B
if [ "$1" == "--13b" ]; then
    MODEL="liuhaotian/llava-v1.5-13b"
    MODEL_NAME="llava-v1.5-13b"
else
    MODEL="liuhaotian/llava-v1.5-7b"
    MODEL_NAME="llava-v1.5-7b"
fi

echo "模型: $MODEL"
echo ""

# 检查 huggingface_hub
pip install -q huggingface_hub

python3 << PYEOF
from huggingface_hub import snapshot_download
import os

model_id = "$MODEL"
local_dir = os.path.join("models", "$MODEL_NAME")

if os.path.exists(local_dir) and any(f.endswith(('.bin', '.safetensors')) for f in os.listdir(local_dir)):
    print(f"模型已存在: {local_dir}，跳过下载")
else:
    print(f"下载 {model_id} -> {local_dir}")
    print("这可能需要 10-20 分钟 (7B ≈ 14GB, 13B ≈ 26GB)...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print("下载完成!")
PYEOF

echo ""
echo "============================================"
echo "  下载完成!"
echo "  模型路径: models/$MODEL_NAME"
echo ""
echo "  运行 demo:"
echo "    python demo_fastv.py --model-path models/$MODEL_NAME"
echo "============================================"
