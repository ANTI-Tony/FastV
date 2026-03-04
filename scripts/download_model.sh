#!/bin/bash
set -e

echo "============================================"
echo "  下载 LLaVA-1.5 模型"
echo "============================================"

MODEL_7B="liuhaotian/llava-v1.5-7b"
MODEL_13B="liuhaotian/llava-v1.5-13b"

# 默认下载 7B，加 --13b 参数下载 13B
MODEL=$MODEL_7B
MODEL_NAME="llava-v1.5-7b"

if [ "$1" == "--13b" ]; then
    MODEL=$MODEL_13B
    MODEL_NAME="llava-v1.5-13b"
fi

echo "下载模型: $MODEL"
echo "这可能需要一些时间..."

python3 -c "
from huggingface_hub import snapshot_download
import os

model_id = '$MODEL'
local_dir = os.path.join('models', '$MODEL_NAME')

print(f'下载 {model_id} 到 {local_dir}')
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print('下载完成!')
"

echo "============================================"
echo "  模型已下载到 models/$MODEL_NAME"
echo "============================================"
