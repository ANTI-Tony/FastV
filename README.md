# FastV 复现项目

> 论文: "An Image is Worth 1/2 Tokens After Layer 2" (ECCV 2024 Oral)
> 原始仓库: https://github.com/pkunlp-icler/FastV

## 环境要求
- RunPod / A100 80GB
- CUDA 11.8+
- Python 3.10

## 快速开始（RunPod 上执行）

```bash
# 1. Clone 仓库
git clone https://github.com/ANTI-Tony/FastV.git
cd FastV

# 2. 一键安装环境
bash setup_runpod.sh

# 3. 下载模型 (LLaVA-1.5-7B)
bash scripts/download_model.sh

# 4. 运行 FastV Demo (验证安装)
python demo_fastv.py

# 5. 运行完整评测
bash scripts/run_eval.sh
```

## 项目结构
```
FastV/
├── README.md                    # 本文件
├── setup_runpod.sh              # RunPod 一键安装脚本
├── demo_fastv.py                # FastV 推理 Demo
├── fastv/                       # FastV 核心实现
│   ├── __init__.py
│   ├── fastv_llama.py           # 修改后的 LlamaModel (核心)
│   ├── fastv_config.py          # FastV 配置
│   └── attention_viz.py         # 注意力可视化
├── scripts/
│   ├── download_model.sh        # 模型下载
│   ├── download_data.sh         # 数据集下载
│   ├── run_eval.sh              # 运行全部评测
│   ├── eval_aokvqa.py           # A-OKVQA 评测
│   ├── eval_textvqa.py          # TextVQA 评测
│   └── benchmark_latency.py     # 延迟基准测试
├── configs/
│   └── fastv_configs.yaml       # 实验配置
└── results/                     # 输出结果目录
```

## FastV 核心参数

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| K | 在第几层剪枝 | 2 |
| R | 剪掉多少比例的 image tokens | 0.5 或 0.75 |

## 预期结果 (LLaVA-1.5-13B, A100 80GB)

| 配置 | 准确率 | 延迟 | 显存 |
|------|--------|------|------|
| Vanilla (无 FastV) | 81.9% | 0.203s | 33 GB |
| FastV K=2, R=75% | 80.9% | 0.124s | 27 GB |
