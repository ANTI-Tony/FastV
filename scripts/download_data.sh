#!/bin/bash
set -e

echo "============================================"
echo "  下载评测数据集"
echo "============================================"

DATA_DIR="data"
mkdir -p $DATA_DIR

# 1. TextVQA
echo "[1/3] 下载 TextVQA..."
mkdir -p $DATA_DIR/textvqa
if [ ! -f "$DATA_DIR/textvqa/TextVQA_0.5.1_val.json" ]; then
    wget -q -P $DATA_DIR/textvqa \
        https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
    echo "  TextVQA 标注已下载"
else
    echo "  TextVQA 标注已存在，跳过"
fi

if [ ! -d "$DATA_DIR/textvqa/train_images" ]; then
    wget -q -P $DATA_DIR/textvqa \
        https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
    cd $DATA_DIR/textvqa && unzip -q train_val_images.zip && rm train_val_images.zip && cd ../..
    echo "  TextVQA 图片已下载"
else
    echo "  TextVQA 图片已存在，跳过"
fi

# 2. GQA
echo "[2/3] 下载 GQA..."
mkdir -p $DATA_DIR/gqa
if [ ! -f "$DATA_DIR/gqa/testdev_balanced_questions.json" ]; then
    wget -q -P $DATA_DIR/gqa \
        https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
    cd $DATA_DIR/gqa && unzip -q questions1.2.zip && rm questions1.2.zip && cd ../..
    echo "  GQA 已下载"
else
    echo "  GQA 已存在，跳过"
fi

# 3. MME
echo "[3/3] 下载 MME benchmark..."
mkdir -p $DATA_DIR/mme
echo "  MME 需要手动从 https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models 申请"
echo "  下载后放到 $DATA_DIR/mme/ 目录"

echo "============================================"
echo "  数据集下载完成!"
echo "  目录结构:"
echo "  data/"
echo "  ├── textvqa/"
echo "  ├── gqa/"
echo "  └── mme/"
echo "============================================"
