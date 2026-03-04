#!/bin/bash
# 自动激活 venv 并运行命令
# 用法: bash run.sh python demo_fastv.py --model-path models/llava-v1.5-7b
#       bash run.sh python scripts/benchmark_latency.py

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "错误: venv 未找到，请先运行 bash setup_runpod.sh"
    exit 1
fi

exec "$@"
