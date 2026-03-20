"""
从 HuggingFace 下载 DocVQA 验证集

DocVQA: 密集文档理解，FastV 最可能翻车的数据集
  - 文字分布在文档图片各处
  - 剪掉 75% token = 丢掉 75% 文字

用法:
    pip install datasets Pillow
    python scripts/download_docvqa.py
    python scripts/download_docvqa.py --max-samples 1000
"""

import os
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=2000)
    parser.add_argument('--output-dir', type=str, default='data/docvqa')
    args = parser.parse_args()

    print("下载 DocVQA 验证集 (from HuggingFace)...")

    from datasets import load_dataset

    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")

    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

    samples = []
    count = 0

    for i, item in enumerate(ds):
        if count >= args.max_samples:
            break

        image = item.get('image')
        if image is None:
            continue

        img_filename = f"doc_{i}.png"
        img_path = os.path.join(args.output_dir, 'images', img_filename)
        image.save(img_path)

        # DocVQA answers
        answers = item.get('answers', [])
        if isinstance(answers, str):
            answers = [answers]

        samples.append({
            'id': i,
            'image_id': f"doc_{i}",
            'image': img_filename,
            'question': item['question'],
            'answers': answers,
        })
        count += 1

        if count % 200 == 0:
            print(f"  已处理 {count}/{args.max_samples}...")

    # Save
    val_path = os.path.join(args.output_dir, 'val.json')
    with open(val_path, 'w') as f:
        json.dump({'data': samples}, f, indent=2, ensure_ascii=False)

    print(f"\n完成!")
    print(f"  样本数: {len(samples)}")
    print(f"  标注: {val_path}")
    print(f"  图片: {os.path.join(args.output_dir, 'images')}/")


if __name__ == '__main__':
    main()
