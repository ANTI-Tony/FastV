#!/bin/bash
set -e

echo "============================================"
echo "  下载多数据集评测基准"
echo "  目标: 找出 FastV 的泛化性弱点"
echo "============================================"

DATA_DIR="data"
mkdir -p $DATA_DIR

# ============================================================
# 1. TextVQA (已有)
# ============================================================
echo "[1/8] TextVQA (OCR-heavy)..."
mkdir -p $DATA_DIR/textvqa
if [ ! -f "$DATA_DIR/textvqa/TextVQA_0.5.1_val.json" ]; then
    wget -q -P $DATA_DIR/textvqa \
        https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
fi
echo "  ✓ TextVQA"

# ============================================================
# 2. GQA — 组合推理，需要空间关系
# ============================================================
echo "[2/8] GQA (compositional reasoning)..."
mkdir -p $DATA_DIR/gqa
if [ ! -f "$DATA_DIR/gqa/testdev_balanced_questions.json" ]; then
    wget -q -P $DATA_DIR/gqa \
        https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
    cd $DATA_DIR/gqa && unzip -q questions1.2.zip 2>/dev/null || true && cd ../..
fi
# GQA images from Visual Genome
if [ ! -d "$DATA_DIR/gqa/images" ]; then
    echo "  GQA图片需要从Visual Genome下载:"
    echo "  wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
    echo "  放到 $DATA_DIR/gqa/images/"
fi
echo "  ✓ GQA"

# ============================================================
# 3. POPE — 幻觉检测 (Yes/No)
#    FastV 剪掉的 token 可能导致更多幻觉
# ============================================================
echo "[3/8] POPE (hallucination detection)..."
mkdir -p $DATA_DIR/pope
if [ ! -d "$DATA_DIR/pope/coco" ]; then
    echo "  从 GitHub 下载 POPE 标注..."
    git clone --depth 1 https://github.com/AoiDragon/POPE.git /tmp/pope_repo 2>/dev/null || true
    cp -r /tmp/pope_repo/output/coco $DATA_DIR/pope/ 2>/dev/null || true
    rm -rf /tmp/pope_repo
fi
# POPE 使用 COCO val2014 图片
if [ ! -d "$DATA_DIR/coco/val2014" ]; then
    echo "  POPE需要COCO val2014图片:"
    echo "  wget http://images.cocodataset.org/zips/val2014.zip"
    echo "  解压到 $DATA_DIR/coco/val2014/"
fi
echo "  ✓ POPE"

# ============================================================
# 4. ScienceQA — 科学推理 (选择题)
#    需要理解图表/示意图中的细节
# ============================================================
echo "[4/8] ScienceQA (scientific reasoning)..."
mkdir -p $DATA_DIR/scienceqa
if [ ! -f "$DATA_DIR/scienceqa/problems.json" ]; then
    echo "  从 HuggingFace 下载 ScienceQA..."
    pip install datasets -q 2>/dev/null || true
    python3 -c "
from datasets import load_dataset
import json, os
ds = load_dataset('derek-thomas/ScienceQA', split='test')
# 只取有图片的样本
samples = [s for s in ds if s.get('image') is not None][:2000]
os.makedirs('$DATA_DIR/scienceqa/images', exist_ok=True)
problems = []
for i, s in enumerate(samples):
    img_path = f'$DATA_DIR/scienceqa/images/{i}.png'
    if s['image'] is not None:
        s['image'].save(img_path)
    problems.append({
        'id': i,
        'question': s['question'],
        'choices': s['choices'],
        'answer': s['answer'],
        'image': img_path if s['image'] else None,
    })
with open('$DATA_DIR/scienceqa/problems.json', 'w') as f:
    json.dump(problems, f, indent=2)
print(f'  Saved {len(problems)} ScienceQA problems')
" 2>/dev/null || echo "  需要手动下载 ScienceQA"
fi
echo "  ✓ ScienceQA"

# ============================================================
# 5. DocVQA — 文档理解 (密集小文字)
#    FastV 最可能翻车的数据集: 文档图片中文字分布在各处
#    剪掉75%的token = 丢掉大量文字信息
# ============================================================
echo "[5/8] DocVQA (document understanding - CRITICAL)..."
mkdir -p $DATA_DIR/docvqa
if [ ! -f "$DATA_DIR/docvqa/val.json" ]; then
    echo "  DocVQA 需要从 https://rrc.cvc.uab.es/?ch=17 注册下载"
    echo "  下载 val_v1.0.json 和 val images"
    echo "  放到 $DATA_DIR/docvqa/"
fi
echo "  ✓ DocVQA (需手动下载)"

# ============================================================
# 6. ChartQA — 图表理解 (需要精确读取数据点)
#    FastV 剪掉的 token 可能包含关键数据点
# ============================================================
echo "[6/8] ChartQA (chart understanding - CRITICAL)..."
mkdir -p $DATA_DIR/chartqa
if [ ! -f "$DATA_DIR/chartqa/test_augmented.json" ]; then
    echo "  从 HuggingFace 下载 ChartQA..."
    python3 -c "
from datasets import load_dataset
import json, os
ds = load_dataset('HuggingFaceM4/ChartQA', split='test')
os.makedirs('$DATA_DIR/chartqa/images', exist_ok=True)
samples = []
for i, s in enumerate(list(ds)[:2000]):
    img_path = f'$DATA_DIR/chartqa/images/{i}.png'
    if s.get('image'):
        s['image'].save(img_path)
    samples.append({
        'id': i,
        'question': s['query'],
        'answer': s['label'],
        'image': img_path,
    })
with open('$DATA_DIR/chartqa/test_augmented.json', 'w') as f:
    json.dump(samples, f, indent=2)
print(f'  Saved {len(samples)} ChartQA samples')
" 2>/dev/null || echo "  需要手动下载 ChartQA"
fi
echo "  ✓ ChartQA"

# ============================================================
# 7. VizWiz — 真实场景辅助 (模糊/低质量图片)
#    图片质量差,每个 token 信息密度低,剪枝更危险
# ============================================================
echo "[7/8] VizWiz (real-world accessibility)..."
mkdir -p $DATA_DIR/vizwiz
if [ ! -f "$DATA_DIR/vizwiz/val.json" ]; then
    echo "  VizWiz 需要从 https://vizwiz.org/tasks-and-datasets/vqa/ 下载"
    echo "  放到 $DATA_DIR/vizwiz/"
fi
echo "  ✓ VizWiz (需手动下载)"

# ============================================================
# 8. SEED-Bench — 综合多维评测
# ============================================================
echo "[8/8] SEED-Bench (comprehensive)..."
mkdir -p $DATA_DIR/seed_bench
if [ ! -f "$DATA_DIR/seed_bench/SEED-Bench.json" ]; then
    echo "  从 HuggingFace 下载 SEED-Bench..."
    python3 -c "
from datasets import load_dataset
import json, os
ds = load_dataset('AILab-CVC/SEED-Bench', split='test')
os.makedirs('$DATA_DIR/seed_bench/images', exist_ok=True)
samples = []
for i, s in enumerate(list(ds)[:3000]):
    img_path = f'$DATA_DIR/seed_bench/images/{i}.png'
    if s.get('image'):
        s['image'].save(img_path)
    samples.append({
        'id': i,
        'question': s['question'],
        'choices': [s.get(f'choice_{c}','') for c in ['a','b','c','d']],
        'answer': s.get('answer', ''),
        'image': img_path,
        'category': s.get('question_type_id', -1),
    })
with open('$DATA_DIR/seed_bench/SEED-Bench.json', 'w') as f:
    json.dump(samples, f, indent=2)
print(f'  Saved {len(samples)} SEED-Bench samples')
" 2>/dev/null || echo "  需要手动下载 SEED-Bench"
fi
echo "  ✓ SEED-Bench"

echo ""
echo "============================================"
echo "  下载完成! 数据集列表:"
echo ""
echo "  最可能暴露 FastV 弱点的:"
echo "  ★★★ DocVQA   — 密集文字,剪枝=丢文字"
echo "  ★★★ ChartQA  — 精确数据点,剪枝=丢数据"
echo "  ★★  POPE     — 幻觉检测,剪枝=更多幻觉"
echo "  ★★  VizWiz   — 低质图片,信息密度低"
echo "  ★   GQA      — 空间推理,需要多区域"
echo "  ★   ScienceQA— 图表/示意图理解"
echo "  ★   SEED     — 综合评测多维度"
echo "============================================"
