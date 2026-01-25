# -*- coding: utf-8 -*-
from __future__ import annotations

import torch


def load_llava_model(model_path: str, model_name: str = "llava_qwen", device_map: str = "auto"):
    """
    Assumes `llava/` is importable (e.g., repo-root added to PYTHONPATH).
    Returns (tokenizer, model, image_processor, context_len).
    """
    from llava.model.builder import load_pretrained_model  # type: ignore

    llava_model_args = {"multimodal": True, "overwrite_config": {"image_aspect_ratio": "pad"}}
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map=device_map, attn_implementation=None, **llava_model_args
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, image_processor, context_len


def get_conv_template(name: str = "qwen_2"):
    from llava.conversation import conv_templates  # type: ignore
    return conv_templates.get(name, conv_templates["qwen_1_5"]).copy()
