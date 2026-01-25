# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image
import torch

from prompt import build_judge_prompt
from llava_utils import load_llava_model, get_conv_template

JSON_RE = re.compile(r"\{.*\}", re.S)


@dataclass
class JudgeScores:
    r: float
    t: float
    a: float

    def to_dict(self) -> Dict[str, float]:
        return {"r": self.r, "t": self.t, "a": self.a}


class LLaVAJudgeEvaluator:
    """Scores (query image, candidate image) using three criteria r/t/a in [0,1]."""
    def __init__(self, model_path: str, model_name: str = "llava_qwen", conv_template: str = "qwen_2"):
        self.tokenizer, self.model, self.image_processor, self.context_len = load_llava_model(
            model_path=model_path, model_name=model_name, device_map="auto"
        )
        self.conv_template = conv_template

    def _decode_new_tokens(self, input_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
        in_len = input_ids.shape[1]
        gen = output_ids[:, in_len:] if output_ids.shape[1] >= in_len else output_ids
        txt = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
        if not txt:
            txt = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return txt

    def score_pair(
        self,
        question: str,
        query_image_path: str,
        candidate_image_path: str,
        reasoning_steps: Optional[List[str]] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 128,
    ) -> JudgeScores:
        conv = get_conv_template(self.conv_template)
        prompt_text = build_judge_prompt(question=question, reasoning_steps=reasoning_steps)

        user_msg = "<image><image>\n" + prompt_text
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        from llava.mm_utils import process_images, tokenizer_image_token  # type: ignore
        from llava.constants import DEFAULT_IMAGE_TOKEN  # type: ignore

        img1 = Image.open(query_image_path).convert("RGB")
        img2 = Image.open(candidate_image_path).convert("RGB")

        proc_out = process_images([img1, img2], self.image_processor, self.model.config)
        if isinstance(proc_out, tuple):
            image_tensor, image_sizes = proc_out
        else:
            image_tensor = proc_out
            image_sizes = [(img1.height, img1.width), (img2.height, img2.width)]

        base_device = self.model.get_input_embeddings().weight.device
        if isinstance(image_tensor, list):
            image_tensor = [im.to(base_device, dtype=torch.float16) for im in image_tensor]
        else:
            image_tensor = image_tensor.to(base_device, dtype=torch.float16)

        image_token_id = getattr(getattr(self.model, "config", {}), "image_token_index", None)
        if image_token_id is None:
            image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)

        input_ids = tokenizer_image_token(full_prompt, self.tokenizer, image_token_id, return_tensors="pt").unsqueeze(0).to(base_device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=base_device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-6) if temperature > 0 else 1.0,
                top_p=0.9,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        resp = self._decode_new_tokens(input_ids, output_ids)
        m = JSON_RE.search(resp)
        if not m:
            raise ValueError(f"Judge output not JSON: {resp[:200]}")
        obj = json.loads(m.group(0))

        r = max(0.0, min(1.0, float(obj.get("r", 0.0))))
        t = max(0.0, min(1.0, float(obj.get("t", 0.0))))
        a = max(0.0, min(1.0, float(obj.get("a", 0.0))))
        return JudgeScores(r=r, t=t, a=a)
