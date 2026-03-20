"""
Model adapters for multi-VLM support.

Abstracts model-specific details so ReVoC works across:
  - LLaVA-1.5 (7B/13B)
  - InternVL-2
  - Qwen-VL

Each adapter provides a uniform interface for:
  - Image encoding → visual features
  - Text embedding layer access
  - LLM layer access (for attention hooks)
  - Input preparation (conversation templates)
  - Generation bypass (inputs_embeds)
"""

import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
from PIL import Image


class VLMAdapter(ABC):
    """Abstract base class for VLM model adapters."""

    @abstractmethod
    def load(self, model_path: str, device: str = "cuda"):
        """Load model, tokenizer, image_processor."""
        ...

    @abstractmethod
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image → (1, N, D) visual features."""
        ...

    @abstractmethod
    def get_embed_tokens(self):
        """Return the text embedding layer."""
        ...

    @abstractmethod
    def get_llm_layers(self) -> list:
        """Return list of transformer decoder layers."""
        ...

    @abstractmethod
    def prepare_input(self, image: Image.Image, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input_ids and image_tensor for the model."""
        ...

    @abstractmethod
    def build_multimodal_embeds(
        self, input_ids: torch.Tensor, image_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Build full embeddings with image features inserted.
        Returns: (full_embeds, image_start, num_image_tokens)
        """
        ...

    @abstractmethod
    def build_multiturn_input(
        self, image: Image.Image, history: list, current_query: str,
    ) -> torch.Tensor:
        """
        Build input_ids for multi-turn conversation.
        history: [(query, response), ...]
        Returns: input_ids tensor
        """
        ...

    @abstractmethod
    def generate(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate text from inputs_embeds, bypassing model-specific limitations."""
        ...

    @abstractmethod
    def forward_with_attention(
        self, inputs_embeds: torch.Tensor, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass capturing attention at specified layer.
        Returns: (model_output, attn_weights_at_layer)
        """
        ...


class LLaVAAdapter(VLMAdapter):
    """Adapter for LLaVA-1.5 (7B and 13B)."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = "cuda"

    def load(self, model_path: str, device: str = "cuda"):
        from fastv.core import load_model
        self.device = device
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_model(model_path, device)
        return self

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        from llava.mm_utils import process_images
        image_tensor = process_images(
            [image], self.image_processor, None
        ).to(self.device, dtype=torch.float16)
        features = self.model.encode_images(image_tensor)  # (1, 576, D)
        return features

    def get_embed_tokens(self):
        return self.model.get_model().embed_tokens

    def get_llm_layers(self) -> list:
        return list(self.model.get_model().layers)

    def prepare_input(self, image: Image.Image, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        from fastv.core import prepare_input
        return prepare_input(
            self.tokenizer, self.image_processor, image, prompt, self.device
        )

    def build_multimodal_embeds(
        self, input_ids: torch.Tensor, image_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        from fastv.core import get_multimodal_embeds
        return get_multimodal_embeds(self.model, input_ids, image_tensor)

    def build_multiturn_input(
        self, image: Image.Image, history: list, current_query: str,
    ) -> torch.Tensor:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        conv = conv_templates["v1"].copy()

        # First turn includes image token
        first_q = history[0][0] if history else current_query
        if DEFAULT_IMAGE_TOKEN not in first_q:
            first_q = DEFAULT_IMAGE_TOKEN + "\n" + first_q

        if history:
            conv.append_message(conv.roles[0], first_q)
            conv.append_message(conv.roles[1], history[0][1])
            for q, a in history[1:]:
                conv.append_message(conv.roles[0], q)
                conv.append_message(conv.roles[1], a)
            conv.append_message(conv.roles[0], current_query)
        else:
            conv.append_message(conv.roles[0], first_q)

        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        return input_ids

    def generate(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
    ) -> str:
        from transformers import LlamaForCausalLM
        with torch.no_grad():
            output_ids = LlamaForCausalLM.generate(
                self.model,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def forward_with_attention(
        self, inputs_embeds: torch.Tensor, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_captured = {}
        llm = self.model.get_model()
        target_layer = llm.layers[layer_idx]

        def hook_fn(module, args, output):
            if len(output) > 1 and output[1] is not None:
                attn_captured['weights'] = output[1].detach()

        hook = target_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            out = self.model(
                inputs_embeds=inputs_embeds,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        hook.remove()

        attn_weights = attn_captured.get('weights', None)
        return out, attn_weights


class InternVLAdapter(VLMAdapter):
    """
    Adapter for InternVL-2.
    Stub implementation — fill in when running experiments on InternVL.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda"

    def load(self, model_path: str, device: str = "cuda"):
        self.device = device
        # InternVL-2 loading:
        # from transformers import AutoModel, AutoTokenizer
        # self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
        #     torch_dtype=torch.float16).to(device).eval()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        raise NotImplementedError(
            "InternVL adapter: install internvl dependencies and uncomment loading code"
        )

    def encode_image(self, image):
        # pixel_values = self.image_processor(image).unsqueeze(0).to(self.device)
        # return self.model.extract_feature(pixel_values)
        raise NotImplementedError

    def get_embed_tokens(self):
        # return self.model.language_model.get_input_embeddings()
        raise NotImplementedError

    def get_llm_layers(self):
        # return list(self.model.language_model.model.layers)
        raise NotImplementedError

    def prepare_input(self, image, prompt):
        raise NotImplementedError

    def build_multimodal_embeds(self, input_ids, image_tensor):
        raise NotImplementedError

    def build_multiturn_input(self, image, history, current_query):
        raise NotImplementedError

    def generate(self, inputs_embeds, attention_mask, max_new_tokens=256):
        raise NotImplementedError

    def forward_with_attention(self, inputs_embeds, layer_idx):
        raise NotImplementedError


class QwenVLAdapter(VLMAdapter):
    """
    Adapter for Qwen-VL / Qwen2-VL.
    Stub implementation — fill in when running experiments on Qwen-VL.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda"

    def load(self, model_path: str, device: str = "cuda"):
        self.device = device
        raise NotImplementedError(
            "QwenVL adapter: install qwen-vl dependencies and uncomment loading code"
        )

    def encode_image(self, image):
        raise NotImplementedError

    def get_embed_tokens(self):
        raise NotImplementedError

    def get_llm_layers(self):
        raise NotImplementedError

    def prepare_input(self, image, prompt):
        raise NotImplementedError

    def build_multimodal_embeds(self, input_ids, image_tensor):
        raise NotImplementedError

    def build_multiturn_input(self, image, history, current_query):
        raise NotImplementedError

    def generate(self, inputs_embeds, attention_mask, max_new_tokens=256):
        raise NotImplementedError

    def forward_with_attention(self, inputs_embeds, layer_idx):
        raise NotImplementedError


def get_adapter(model_type: str) -> VLMAdapter:
    """Factory function for model adapters."""
    adapters = {
        "llava": LLaVAAdapter,
        "internvl": InternVLAdapter,
        "qwen-vl": QwenVLAdapter,
    }
    if model_type not in adapters:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(adapters.keys())}")
    return adapters[model_type]()
