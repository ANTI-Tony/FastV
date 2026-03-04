#!/bin/bash
set -e

# 自动激活 venv
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$(dirname "$SCRIPT_DIR")/venv"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo "============================================"
echo "  FastV 完整评测流程"
echo "============================================"

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

echo "模型: $MODEL_PATH"
echo "最大样本数: $MAX_SAMPLES (设置 MAX_SAMPLES=100 进行快速测试)"
echo ""

# 确保数据已下载
if [ ! -d "data/textvqa" ]; then
    echo "数据未下载，请先运行: bash scripts/download_data.sh"
    exit 1
fi

# 1. Latency Benchmark
echo ""
echo "========== [1/4] 延迟基准测试 =========="
python scripts/benchmark_latency.py --model-path $MODEL_PATH --num-runs 5

# 2. TextVQA - Vanilla
echo ""
echo "========== [2/4] TextVQA - Vanilla =========="
python scripts/eval_textvqa.py \
    --model-path $MODEL_PATH \
    --fastv-r 0.0 \
    --max-samples $MAX_SAMPLES

# 3. TextVQA - FastV R=50%
echo ""
echo "========== [3/4] TextVQA - FastV K=2, R=50% =========="
python scripts/eval_textvqa.py \
    --model-path $MODEL_PATH \
    --fastv-k 2 \
    --fastv-r 0.5 \
    --max-samples $MAX_SAMPLES

# 4. TextVQA - FastV R=75%
echo ""
echo "========== [4/4] TextVQA - FastV K=2, R=75% =========="
python scripts/eval_textvqa.py \
    --model-path $MODEL_PATH \
    --fastv-k 2 \
    --fastv-r 0.75 \
    --max-samples $MAX_SAMPLES

echo ""
echo "============================================"
echo "  评测完成! 结果在 results/ 目录"
echo "============================================"
echo ""
echo "查看结果:"
echo "  cat results/latency_benchmark.json"
echo "  cat results/textvqa_vanilla.json | python -m json.tool | head -20"
echo "  cat results/textvqa_K2_R50.json | python -m json.tool | head -20"
echo "  cat results/textvqa_K2_R75.json | python -m json.tool | head -20"
