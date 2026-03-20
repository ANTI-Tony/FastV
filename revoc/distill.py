"""
Distillation training for the Cross-Attention Retriever.

Training objective: minimize KL divergence between:
  - Teacher: full model output with all 576 visual tokens
  - Student: model output with only ReVoC-retrieved ~140 tokens

Only the retriever's cross-attention layer (~4M params) is trained.
The VLM backbone is completely frozen.

Training data: any single-turn VQA dataset (e.g., TextVQA, GQA).
No multi-turn data needed — the retriever learns to match per-query retrieval.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from tqdm import tqdm

from .config import RevoCConfig
from .compressor import build_compressed_cache
from .retriever import CrossAttentionRetriever
from .model_adapter import VLMAdapter


class DistillationDataset(Dataset):
    """
    Wraps a VQA dataset for retriever distillation.
    Each sample = (image_path, question).
    """

    def __init__(self, data_path: str, image_dir: str, max_samples: int = -1):
        with open(data_path) as f:
            raw = json.load(f)

        if 'data' in raw:
            raw = raw['data']

        self.samples = []
        for item in raw:
            img_path = os.path.join(image_dir, f"{item['image_id']}.jpg")
            if os.path.exists(img_path):
                self.samples.append({
                    'image_path': img_path,
                    'question': item['question'],
                })

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_retriever(
    adapter: VLMAdapter,
    config: RevoCConfig,
    data_path: str,
    image_dir: str,
    output_path: str = "checkpoints/retriever.pt",
    max_samples: int = 2000,
    device: str = "cuda",
):
    """
    Train the cross-attention retriever via distillation.

    Process per sample:
      1. Encode image → 576 features
      2. Forward full model → teacher logits
      3. Build compressed cache
      4. Retrieve tokens via learnable cross-attention
      5. Forward with retrieved tokens → student logits
      6. Loss = KL(teacher || student)
      7. Backprop through retriever only

    Args:
        adapter: loaded VLM adapter
        config: RevoCConfig
        data_path: path to VQA JSON
        image_dir: path to images
        output_path: where to save trained retriever
        max_samples: training set size
        device: cuda/cpu
    """
    from PIL import Image as PILImage
    from llava.constants import IMAGE_TOKEN_INDEX
    from transformers import LlamaForCausalLM

    dataset = DistillationDataset(data_path, image_dir, max_samples)
    print(f"Distillation training: {len(dataset)} samples")

    # Initialize retriever
    retriever = CrossAttentionRetriever(config).to(device)
    optimizer = torch.optim.AdamW(retriever.parameters(), lr=config.distill_lr)

    # Freeze the VLM
    adapter.model.eval()
    for p in adapter.model.parameters():
        p.requires_grad = False

    embed_tokens = adapter.get_embed_tokens()

    total_loss = 0.0
    n_steps = 0

    for epoch in range(config.distill_epochs):
        epoch_loss = 0.0
        pbar = tqdm(range(len(dataset)), desc=f"Epoch {epoch + 1}/{config.distill_epochs}")

        for idx in pbar:
            sample = dataset[idx]
            try:
                image = PILImage.open(sample['image_path']).convert('RGB')
                question = sample['question']

                # 1) Prepare input and get full embeddings
                input_ids, image_tensor = adapter.prepare_input(image, question)
                with torch.no_grad():
                    full_embeds, image_start, num_img = adapter.build_multimodal_embeds(
                        input_ids, image_tensor
                    )

                if image_start < 0:
                    continue

                # 2) Teacher: full model forward → logits
                with torch.no_grad():
                    teacher_out = adapter.model(
                        inputs_embeds=full_embeds,
                        use_cache=False,
                        return_dict=True,
                    )
                    teacher_logits = teacher_out.logits[:, -1, :]  # last token

                # 3) Get attention for cache building
                _, attn_weights = adapter.forward_with_attention(
                    full_embeds, config.ranking_layer - 1
                )
                if attn_weights is None:
                    continue

                # 4) Build compressed cache
                image_features = full_embeds[0, image_start:image_start + num_img, :]
                cache = build_compressed_cache(
                    image_features, attn_weights, image_start, num_img, config
                )

                # 5) Retriever: select clusters (with gradient)
                query_tokens = adapter.tokenizer(
                    question, return_tensors="pt", add_special_tokens=False
                )
                query_ids = query_tokens.input_ids.to(device)
                query_embeds = embed_tokens(query_ids)  # (1, q_len, D)

                selected, scores = retriever(
                    query_embeds,
                    cache.cluster_centers.unsqueeze(0),
                )

                # 6) Recover tokens from selected clusters
                recovered_parts = [cache.global_tokens]
                for c_id in selected.tolist():
                    recovered = cache.residual_store.recover_cluster(
                        c_id, cache.cluster_centers[c_id],
                        target_device=str(device),
                    )
                    recovered_parts.append(recovered)
                retrieved = torch.cat(recovered_parts, dim=0)

                # 7) Build student embeddings
                safe_ids = input_ids.clone()
                image_mask = safe_ids[0] == IMAGE_TOKEN_INDEX
                safe_ids[0][image_mask] = 0
                input_embeds = embed_tokens(safe_ids)

                image_pos = image_mask.nonzero(as_tuple=True)[0]
                if len(image_pos) == 0:
                    continue

                img_start = image_pos[0].item()
                prefix = input_embeds[:, :img_start, :]
                suffix = input_embeds[:, img_start + 1:, :]
                student_embeds = torch.cat(
                    [prefix, retrieved.unsqueeze(0), suffix], dim=1
                )

                # 8) Student forward → logits
                student_out = adapter.model(
                    inputs_embeds=student_embeds,
                    use_cache=False,
                    return_dict=True,
                )
                student_logits = student_out.logits[:, -1, :]

                # 9) KL divergence loss
                loss = F.kl_div(
                    F.log_softmax(student_logits / 2.0, dim=-1),
                    F.softmax(teacher_logits / 2.0, dim=-1),
                    reduction='batchmean',
                ) * (2.0 ** 2)  # temperature scaling

                # 10) Backprop through retriever only
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retriever.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_steps += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            except Exception as e:
                continue

        avg_loss = epoch_loss / max(n_steps, 1)
        print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(retriever.state_dict(), output_path)
    print(f"Retriever saved to {output_path}")
    return retriever
