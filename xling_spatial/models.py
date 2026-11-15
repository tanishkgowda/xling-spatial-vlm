"""Model wrapper: load a VLM with eager attention, build inputs, score Yes/No with optional
AdaptVis scaling, and map image tokens to positions on the original image."""
from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence

import numpy as np
import torch
import transformers
from PIL import Image

from . import attention
from .prompts import answer_token_ids

SPECS = {
    # kind: (model class name, default dtype on a T4)
    "qwen2_5_vl": ("Qwen2_5_VLForConditionalGeneration", "float16"),
    "gemma3": ("Gemma3ForConditionalGeneration", "float32"),   # Gemma 3 overflows in fp16
    "llava": ("LlavaForConditionalGeneration", "float16"),
    "llava_onevision": ("LlavaOnevisionForConditionalGeneration", "float16"),
}


def detect_kind(model_id: str) -> str:
    m = model_id.lower()
    if "qwen2.5-vl" in m or "qwen2_5_vl" in m:
        return "qwen2_5_vl"
    if "gemma-3" in m or "gemma3" in m:
        return "gemma3"
    if "onevision" in m:
        return "llava_onevision"
    if "llava" in m:
        return "llava"
    raise ValueError(f"Cannot infer model kind from {model_id!r}; pass kind=...")


class VLM:
    def __init__(self, model_id: str, kind: Optional[str] = None, dtype: Optional[str] = None,
                 device_map="auto", min_pixels: int = 256 * 28 * 28, max_pixels: int = 512 * 28 * 28,
                 hf_token: Optional[str] = None):
        self.model_id = model_id
        self.kind = kind or detect_kind(model_id)
        cls_name, default_dtype = SPECS[self.kind]
        self.dtype = getattr(torch, dtype or default_dtype)

        proc_kwargs = {"token": hf_token}
        if self.kind == "qwen2_5_vl":
            # Caps image tokens (min..max) so eager attention fits on a T4 and runs stay comparable.
            proc_kwargs.update(min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor = transformers.AutoProcessor.from_pretrained(model_id, **proc_kwargs)
        self.tokenizer = self.processor.tokenizer

        cls = getattr(transformers, cls_name)
        self.model = cls.from_pretrained(model_id, torch_dtype=self.dtype, device_map=device_map,
                                         attn_implementation="eager", token=hf_token).eval()
        self.n_layers = attention.install(self.model)

        cfg = self.model.config
        self.image_token_id = getattr(cfg, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = getattr(cfg, "image_token_index")
        self._answer_cache = {}

    # ---------- inputs ----------
    def build_inputs(self, image: Image.Image, text: Optional[str] = None, raw_prompt: Optional[str] = None):
        if raw_prompt is not None:
            prompt = raw_prompt
        else:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(device)
        for k in ("pixel_values", "pixel_values_videos"):
            if k in inputs and inputs[k].is_floating_point():
                inputs[k] = inputs[k].to(self.dtype)
        return inputs

    def image_positions(self, inputs) -> torch.Tensor:
        pos = (inputs["input_ids"][0] == self.image_token_id).nonzero().squeeze(-1)
        if pos.numel() == 0:
            raise RuntimeError("No image tokens in input_ids: the processor did not expand the image placeholder.")
        return pos

    def answer_ids(self, langs: Sequence[str]):
        key = tuple(sorted(langs))
        if key not in self._answer_cache:
            yes, no, clash = answer_token_ids(self.tokenizer, key)
            if clash:
                warnings.warn(f"{self.model_id}: dropped yes/no first tokens shared by both answers: {clash}")
            self._answer_cache[key] = (yes, no)
        return self._answer_cache[key]

    # ---------- scoring ----------
    @torch.inference_mode()
    def score(self, inputs, answer_langs=("en",), weight: float = 1.0, layers=None, log: bool = False):
        pos = self.image_positions(inputs)
        attention.STATE.reset_call(pos, weight, layers, log)
        try:
            try:
                out = self.model(**inputs, use_cache=False, logits_to_keep=1)
            except TypeError:
                out = self.model(**inputs, use_cache=False)
        finally:
            attention.STATE.image_pos = None
        records = attention.STATE.records
        attention.STATE.records = {}

        logits = out.logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        yes_ids, no_ids = self.answer_ids(answer_langs)
        yes_lp = torch.logsumexp(lp[yes_ids], 0).item()
        no_lp = torch.logsumexp(lp[no_ids], 0).item()
        top = int(lp.argmax())
        return {
            "margin": yes_lp - no_lp,
            "yes_lp": yes_lp,
            "no_lp": no_lp,
            "conf": float(lp.max().exp()),          # AdaptVis confidence: max first-token probability
            "top_token": self.tokenizer.decode([top]),
            "yesno_mass": float(np.exp(yes_lp) + np.exp(no_lp)),  # low = model is not answering yes/no
            "n_tokens": int(inputs["input_ids"].shape[1]),
            "n_img_tokens": int(pos.numel()),
            "records": records,
        }

    # ---------- geometry ----------
    def token_centers(self, inputs, image_size, n_img: int) -> Optional[np.ndarray]:
        """(n_img, 2) array of (x, y) centres of each image token, in [0,1] coordinates of the image
        given to the processor (image_size = (W, H) of that image). None if unsupported."""
        W, H = image_size
        if self.kind == "qwen2_5_vl":
            t, gh, gw = [int(v) for v in inputs["image_grid_thw"][0]]
            m = getattr(self.processor.image_processor, "merge_size", 2)
            gh, gw = gh // m, gw // m
            x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
        elif self.kind in ("gemma3", "llava"):
            side = int(round(math.sqrt(n_img)))
            gh = gw = side
            x0, x1, y0, y1 = 0.0, 1.0, 0.0, 1.0
            ip = self.processor.image_processor
            if self.kind == "llava" and getattr(ip, "do_center_crop", False):
                # shortest edge -> crop size, then centre crop: part of the long side is cut off
                if W >= H:
                    x0 = (1 - H / W) / 2; x1 = 1 - x0
                else:
                    y0 = (1 - W / H) / 2; y1 = 1 - y0
        else:
            return None
        if gh * gw != n_img:
            warnings.warn(f"grid {gh}x{gw} != {n_img} image tokens; skipping box metrics")
            return None
        jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
        xs = x0 + (jj.ravel() + 0.5) / gw * (x1 - x0)
        ys = y0 + (ii.ravel() + 0.5) / gh * (y1 - y0)
        return np.stack([xs, ys], 1)


def in_boxes(centers: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
    m = np.zeros(len(centers), dtype=bool)
    for x0, y0, x1, y1 in boxes or []:
        m |= (centers[:, 0] >= x0) & (centers[:, 0] <= x1) & (centers[:, 1] >= y0) & (centers[:, 1] <= y1)
    return m
