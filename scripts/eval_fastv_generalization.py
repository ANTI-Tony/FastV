"""
FastV 泛化性测试 — 跨数据集评测

目的: 找出 FastV 在哪些数据集上掉点严重
      如果 FastV 在某些任务上 accuracy drop > 5%, 就是 ReVoC 的突破口

测试的数据集及其特点:
  TextVQA   — OCR密集, 文字分布在图像各处
  GQA       — 组合推理, 需要空间关系理解
  POPE      — 幻觉检测, 剪枝可能增加幻觉
  DocVQA    — 文档理解, 密集小文字 (最可能翻车)
  ChartQA   — 图表理解, 需要精确读取数据点
  ScienceQA — 科学推理, 图表/示意图
  VizWiz    — 真实低质图片
  SEED-Bench — 综合多维

跑法:
  bash run.sh python scripts/eval_fastv_generalization.py --dataset all
  bash run.sh python scripts/eval_fastv_generalization.py --dataset docvqa --max-samples 500
  bash run.sh python scripts/eval_fastv_generalization.py --dataset chartqa,pope --fastv-r 0.5,0.75

结果产出: results/generalization_report.json — 汇总表
"""

import argparse
import json
import os
import sys
import time
from tqdm import tqdm

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastv.core import load_model, prepare_input, run_vanilla, run_fastv


# ============================================================
# 各数据集的加载器和评测器
# ============================================================

class DatasetEvaluator:
    """Base class for dataset-specific evaluation."""
    name = "base"
    metric_name = "accuracy"

    def load_data(self, data_path, image_dir, max_samples):
        raise NotImplementedError

    def format_prompt(self, sample):
        raise NotImplementedError

    def get_image_path(self, sample, image_dir):
        raise NotImplementedError

    def compute_accuracy(self, pred, sample):
        raise NotImplementedError


class TextVQAEvaluator(DatasetEvaluator):
    name = "TextVQA"

    def load_data(self, data_path, image_dir, max_samples):
        with open(data_path) as f:
            data = json.load(f)['data']
        return data[:max_samples] if max_samples > 0 else data

    def format_prompt(self, sample):
        return sample['question'] + "\nAnswer the question using a single word or phrase."

    def get_image_path(self, sample, image_dir):
        return os.path.join(image_dir, f"{sample['image_id']}.jpg")

    def compute_accuracy(self, pred, sample):
        pred = pred.strip().lower()
        counts = {}
        for a in sample.get('answers', []):
            a = a.strip().lower()
            counts[a] = counts.get(a, 0) + 1
        return min(counts.get(pred, 0) / 3.0, 1.0)


class GQAEvaluator(DatasetEvaluator):
    name = "GQA"

    def load_data(self, data_path, image_dir, max_samples):
        with open(data_path) as f:
            data = json.load(f)
        samples = [{'id': k, **v} for k, v in data.items()]
        return samples[:max_samples] if max_samples > 0 else samples

    def format_prompt(self, sample):
        return sample['question'] + "\nAnswer the question using a single word or phrase."

    def get_image_path(self, sample, image_dir):
        return os.path.join(image_dir, f"{sample['imageId']}.jpg")

    def compute_accuracy(self, pred, sample):
        pred = pred.strip().lower()
        gt = sample['answer'].strip().lower()
        return 1.0 if pred == gt else 0.0


class POPEEvaluator(DatasetEvaluator):
    """POPE hallucination benchmark — Yes/No questions."""
    name = "POPE"

    def load_data(self, data_path, image_dir, max_samples):
        samples = []
        # POPE has multiple files: popular, random, adversarial
        for variant in ['popular', 'random', 'adversarial']:
            fpath = os.path.join(data_path, f"coco_pope_{variant}.json")
            if not os.path.exists(fpath):
                # try alternate path
                fpath = os.path.join(data_path, f"coco/{variant}.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    for line in f:
                        item = json.loads(line.strip())
                        item['variant'] = variant
                        samples.append(item)
        return samples[:max_samples] if max_samples > 0 else samples

    def format_prompt(self, sample):
        return sample['text']  # Already formatted as question

    def get_image_path(self, sample, image_dir):
        img_name = sample.get('image', '')
        return os.path.join(image_dir, img_name)

    def compute_accuracy(self, pred, sample):
        pred = pred.strip().lower()
        gt = sample['label'].strip().lower()
        # Match yes/no
        pred_yn = 'yes' if 'yes' in pred else ('no' if 'no' in pred else pred)
        return 1.0 if pred_yn == gt else 0.0


class DocVQAEvaluator(DatasetEvaluator):
    """DocVQA — document understanding with dense text."""
    name = "DocVQA"

    def load_data(self, data_path, image_dir, max_samples):
        with open(data_path) as f:
            data = json.load(f)
        samples = data.get('data', data) if isinstance(data, dict) else data
        return samples[:max_samples] if max_samples > 0 else samples

    def format_prompt(self, sample):
        return sample['question'] + "\nAnswer the question using the text in the document."

    def get_image_path(self, sample, image_dir):
        img = sample.get('image', sample.get('image_id', ''))
        if not img.endswith(('.png', '.jpg')):
            img = img + '.png'
        return os.path.join(image_dir, img)

    def compute_accuracy(self, pred, sample):
        pred = pred.strip().lower()
        answers = sample.get('answers', [])
        if isinstance(answers, str):
            answers = [answers]
        for a in answers:
            if a.strip().lower() in pred or pred in a.strip().lower():
                return 1.0
        return 0.0


class ChartQAEvaluator(DatasetEvaluator):
    """ChartQA — chart understanding."""
    name = "ChartQA"

    def load_data(self, data_path, image_dir, max_samples):
        with open(data_path) as f:
            data = json.load(f)
        return data[:max_samples] if max_samples > 0 else data

    def format_prompt(self, sample):
        q = sample.get('question', sample.get('query', ''))
        return q + "\nAnswer briefly."

    def get_image_path(self, sample, image_dir):
        return sample.get('image', os.path.join(image_dir, f"{sample['id']}.png"))

    def compute_accuracy(self, pred, sample):
        pred = pred.strip().lower()
        gt = str(sample.get('answer', sample.get('label', ''))).strip().lower()
        if pred == gt:
            return 1.0
        # Relaxed match: contained
        if gt in pred or pred in gt:
            return 1.0
        return 0.0


class ScienceQAEvaluator(DatasetEvaluator):
    """ScienceQA — multiple choice with images."""
    name = "ScienceQA"

    def load_data(self, data_path, image_dir, max_samples):
        with open(data_path) as f:
            data = json.load(f)
        # Only keep samples with images
        data = [s for s in data if s.get('image')]
        return data[:max_samples] if max_samples > 0 else data

    def format_prompt(self, sample):
        q = sample['question']
        choices = sample.get('choices', [])
        opts = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
        return f"{q}\n{opts}\nAnswer with the letter of the correct option."

    def get_image_path(self, sample, image_dir):
        return sample.get('image', '')

    def compute_accuracy(self, pred, sample):
        pred = pred.strip().upper()
        gt_idx = sample.get('answer', 0)
        gt_letter = chr(65 + gt_idx)
        # Check if prediction contains the correct letter
        if gt_letter in pred[:3]:
            return 1.0
        return 0.0


class SEEDBenchEvaluator(DatasetEvaluator):
    """SEED-Bench — comprehensive multi-dimensional evaluation."""
    name = "SEED-Bench"

    def load_data(self, data_path, image_dir, max_samples):
        with open(data_path) as f:
            data = json.load(f)
        data = [s for s in data if s.get('image')]
        return data[:max_samples] if max_samples > 0 else data

    def format_prompt(self, sample):
        q = sample['question']
        choices = sample.get('choices', [])
        opts = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices) if c)
        return f"{q}\n{opts}\nAnswer with the letter only."

    def get_image_path(self, sample, image_dir):
        return sample.get('image', '')

    def compute_accuracy(self, pred, sample):
        pred = pred.strip().upper()
        gt = str(sample.get('answer', '')).strip().upper()
        if gt in pred[:3]:
            return 1.0
        return 0.0


# ============================================================
# 数据集注册表
# ============================================================

DATASETS = {
    'textvqa': {
        'evaluator': TextVQAEvaluator(),
        'data_path': 'data/textvqa/TextVQA_0.5.1_val.json',
        'image_dir': 'data/textvqa/train_images',
        'risk': '★',
        'reason': 'OCR-heavy, text spread across image',
    },
    'gqa': {
        'evaluator': GQAEvaluator(),
        'data_path': 'data/gqa/testdev_balanced_questions.json',
        'image_dir': 'data/gqa/images',
        'risk': '★',
        'reason': 'Compositional reasoning, spatial relations',
    },
    'pope': {
        'evaluator': POPEEvaluator(),
        'data_path': 'data/pope',
        'image_dir': 'data/coco/val2014',
        'risk': '★★',
        'reason': 'Pruning may increase hallucination rate',
    },
    'docvqa': {
        'evaluator': DocVQAEvaluator(),
        'data_path': 'data/docvqa/val.json',
        'image_dir': 'data/docvqa/images',
        'risk': '★★★',
        'reason': 'Dense small text everywhere — pruning = losing text',
    },
    'chartqa': {
        'evaluator': ChartQAEvaluator(),
        'data_path': 'data/chartqa/test_augmented.json',
        'image_dir': 'data/chartqa/images',
        'risk': '★★★',
        'reason': 'Precise data points — pruning = losing data',
    },
    'scienceqa': {
        'evaluator': ScienceQAEvaluator(),
        'data_path': 'data/scienceqa/problems.json',
        'image_dir': 'data/scienceqa/images',
        'risk': '★★',
        'reason': 'Diagrams/charts require detail understanding',
    },
    'seed_bench': {
        'evaluator': SEEDBenchEvaluator(),
        'data_path': 'data/seed_bench/SEED-Bench.json',
        'image_dir': 'data/seed_bench/images',
        'risk': '★',
        'reason': 'Multi-dimensional comprehensive benchmark',
    },
}


# ============================================================
# 主评测循环
# ============================================================

def evaluate_dataset(
    evaluator, data, image_dir, model, tokenizer, image_processor, device,
    fastv_r=0.0, fastv_k=2, max_new_tokens=128,
):
    """Run evaluation on one dataset with one FastV config."""
    correct = 0.0
    total = 0
    errors = 0
    total_time = 0.0
    per_sample = []

    for sample in tqdm(data, desc=f"{evaluator.name}", leave=False):
        img_path = evaluator.get_image_path(sample, image_dir)
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            prompt = evaluator.format_prompt(sample)
            input_ids, image_tensor = prepare_input(
                tokenizer, image_processor, image, prompt, device
            )

            t0 = time.time()
            if fastv_r == 0.0:
                pred = run_vanilla(model, tokenizer, input_ids, image_tensor,
                                   device, max_new_tokens)
            else:
                pred = run_fastv(model, tokenizer, input_ids, image_tensor,
                                 device, fastv_k=fastv_k, fastv_r=fastv_r,
                                 max_new_tokens=max_new_tokens)
            total_time += time.time() - t0

            acc = evaluator.compute_accuracy(pred, sample)
            correct += acc
            total += 1

            per_sample.append({
                'id': sample.get('id', sample.get('image_id', total)),
                'pred': pred[:200],
                'acc': acc,
            })

        except Exception as e:
            errors += 1
            continue

    accuracy = correct / total * 100 if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'errors': errors,
        'avg_time': avg_time,
        'per_sample': per_sample,
    }


def main():
    parser = argparse.ArgumentParser(description='FastV Generalization Test')
    parser.add_argument('--model-path', type=str, default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Comma-separated: textvqa,gqa,pope,docvqa,chartqa,scienceqa,seed_bench or "all"')
    parser.add_argument('--fastv-r', type=str, default='0.0,0.5,0.75',
                        help='Comma-separated pruning ratios to test')
    parser.add_argument('--fastv-k', type=int, default=2)
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Max samples per dataset (for speed)')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    ratios = [float(r) for r in args.fastv_r.split(',')]

    # Determine datasets to evaluate
    if args.dataset == 'all':
        dataset_names = list(DATASETS.keys())
    else:
        dataset_names = [d.strip() for d in args.dataset.split(',')]

    # Filter to available datasets
    available = []
    for name in dataset_names:
        if name not in DATASETS:
            print(f"⚠ Unknown dataset: {name}")
            continue
        info = DATASETS[name]
        dp = info['data_path']
        if not os.path.exists(dp):
            print(f"⚠ {name}: data not found at {dp}, skipping")
            continue
        available.append(name)

    if not available:
        print("没有可用的数据集! 请先运行 bash scripts/download_benchmarks.sh")
        return

    print(f"将评测 {len(available)} 个数据集: {', '.join(available)}")
    print(f"FastV 配置: R = {ratios}")
    print(f"每个数据集最多 {args.max_samples} 样本")

    # Load model
    print(f"\n加载模型: {args.model_path}")
    tokenizer, model, image_processor, _ = load_model(args.model_path, device)

    # ============================================================
    # 主评测循环
    # ============================================================
    all_results = {}

    for ds_name in available:
        info = DATASETS[ds_name]
        evaluator = info['evaluator']

        print(f"\n{'='*60}")
        print(f"  {ds_name.upper()} (风险: {info['risk']})")
        print(f"  原因: {info['reason']}")
        print(f"{'='*60}")

        # Load data
        data = evaluator.load_data(
            info['data_path'], info['image_dir'], args.max_samples
        )
        print(f"  样本数: {len(data)}")

        ds_results = {}

        for r in ratios:
            config_name = f"R{int(r*100)}" if r > 0 else "vanilla"
            print(f"\n  >>> {config_name} (R={r})")

            result = evaluate_dataset(
                evaluator, data, info['image_dir'],
                model, tokenizer, image_processor, device,
                fastv_r=r, fastv_k=args.fastv_k,
            )

            ds_results[config_name] = {
                'accuracy': result['accuracy'],
                'total': result['total'],
                'avg_time': result['avg_time'],
            }

            print(f"      准确率: {result['accuracy']:.2f}% "
                  f"({result['correct']:.0f}/{result['total']})")

        # Compute accuracy drops
        if 'vanilla' in ds_results:
            baseline = ds_results['vanilla']['accuracy']
            for cfg, res in ds_results.items():
                if cfg != 'vanilla':
                    drop = baseline - res['accuracy']
                    res['drop_from_vanilla'] = drop
                    marker = "⚠⚠⚠" if drop > 5 else ("⚠" if drop > 2 else "✓")
                    print(f"      {cfg} drop: {drop:+.2f}% {marker}")

        all_results[ds_name] = {
            'risk': info['risk'],
            'reason': info['reason'],
            'results': ds_results,
        }

    # ============================================================
    # 汇总报告
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"  FastV 泛化性测试 — 汇总报告")
    print(f"{'='*80}")

    header = f"  {'Dataset':<14}{'Risk':<6}"
    for r in ratios:
        cfg = f"R{int(r*100)}" if r > 0 else "vanilla"
        header += f"{cfg:<12}"
    if len(ratios) > 1:
        header += f"{'Drop(R75%)':<12}"
    print(header)
    print(f"  {'-'*(len(header)-2)}")

    critical_datasets = []

    for ds_name, ds_info in all_results.items():
        line = f"  {ds_name:<14}{ds_info['risk']:<6}"
        for r in ratios:
            cfg = f"R{int(r*100)}" if r > 0 else "vanilla"
            acc = ds_info['results'].get(cfg, {}).get('accuracy', 0)
            line += f"{acc:<12.1f}"

        # Highlight drops > 5%
        if len(ratios) > 1:
            drop = ds_info['results'].get('R75', {}).get('drop_from_vanilla', 0)
            marker = "⚠⚠⚠" if drop > 5 else ("⚠" if drop > 2 else "✓")
            line += f"{drop:+.1f}% {marker}"
            if drop > 5:
                critical_datasets.append(ds_name)

        print(line)

    print(f"{'='*80}")

    if critical_datasets:
        print(f"\n  🎯 FastV 在以下数据集上掉点 > 5%: {', '.join(critical_datasets)}")
        print(f"     → ReVoC 的突破口! 这些数据集上 ReVoC 可以保持准确率不掉。")
    else:
        print(f"\n  FastV 在所有数据集上掉点均 < 5%")
        print(f"  → 需要在对抗多轮场景上找差距")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, 'fastv_generalization_report.json')
    # Remove per_sample details for the summary file
    summary = {}
    for ds, info in all_results.items():
        summary[ds] = {
            'risk': info['risk'],
            'reason': info['reason'],
            'results': info['results'],
        }
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  报告已保存到: {report_path}")


if __name__ == '__main__':
    main()
