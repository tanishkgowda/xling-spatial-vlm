"""AdaptVis-style attention scaling and attention logging for Hugging Face VLMs.

How AdaptVis works (shiqichen17/AdaptVis, model_zoo/llama/modeling_llama_add_attn.py):
during the prefill pass, in every decoder layer, the pre-softmax attention logits from
the LAST query position to the IMAGE key positions are multiplied by `weight`.
weight < 1 flattens attention over the image, weight > 1 sharpens it. Nothing else changes.

Here the same edit is made without copying model code: the model is loaded with
attn_implementation="eager", and the module-level `eager_attention_forward` of each
language-model family (llama, qwen2, qwen2_5_vl, gemma3) is swapped for a wrapper.
The wrapper only acts on the language-model self-attention modules registered in
STATE.targets. Vision-tower attention passes straight through to the original function.

The same wrapper optionally logs, per layer, the post-softmax attention of the last query
position: total mass on image tokens per head, and the head-averaged distribution over
image tokens (used for "does it look at the right objects?").
"""
from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set

import torch
from torch import nn

# Modeling modules whose `eager_attention_forward` we may patch.
_FAMILY_MODULES = [
    "transformers.models.llama.modeling_llama",          # LLaVA-1.5 (Vicuna / Llama LM)
    "transformers.models.qwen2.modeling_qwen2",          # LLaVA-OneVision (Qwen2 LM)
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    "transformers.models.gemma3.modeling_gemma3",
]


@dataclass
class _State:
    targets: Set[int] = field(default_factory=set)       # id() of LM self-attn modules
    image_pos: Optional[torch.Tensor] = None             # LongTensor of image token positions
    weight: float = 1.0                                  # AdaptVis coefficient
    layers: Optional[Set[int]] = None                    # None = all layers (AdaptVis default)
    log: bool = False
    records: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)
    _pos_cache: Dict[torch.device, torch.Tensor] = field(default_factory=dict)

    def positions_on(self, device: torch.device) -> torch.Tensor:
        if device not in self._pos_cache:
            self._pos_cache[device] = self.image_pos.to(device)
        return self._pos_cache[device]

    def reset_call(self, image_pos: torch.Tensor, weight: float, layers, log: bool):
        self.image_pos = image_pos
        self._pos_cache = {}
        self.weight = float(weight)
        self.layers = set(layers) if layers is not None else None
        self.log = log
        self.records = {}


STATE = _State()
_ORIGINALS: Dict[str, Callable] = {}


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _make_wrapper(original: Callable) -> Callable:
    sig = inspect.signature(original)

    def wrapper(module, query, key, value, attention_mask, *args, **kwargs):
        if id(module) not in STATE.targets or STATE.image_pos is None:
            return original(module, query, key, value, attention_mask, *args, **kwargs)

        bound = sig.bind(module, query, key, value, attention_mask, *args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        scaling = p.get("scaling")
        if scaling is None:
            scaling = module.head_dim ** -0.5
        dropout = p.get("dropout", 0.0) or 0.0
        softcap = p.get("softcap")
        extra = p.get("kwargs", {}) or {}
        if softcap is None:
            softcap = extra.get("softcap")

        n_rep = getattr(module, "num_key_value_groups", 1)
        key_states = _repeat_kv(key, n_rep)
        value_states = _repeat_kv(value, n_rep)

        attn = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if softcap is not None:
            attn = torch.tanh(attn / softcap) * softcap

        q_len, k_len = attn.shape[-2], attn.shape[-1]
        layer_idx = getattr(module, "layer_idx", -1)
        is_prefill = q_len > 1 and q_len == k_len
        layer_on = STATE.layers is None or layer_idx in STATE.layers
        pos = STATE.positions_on(attn.device) if is_prefill else None

        # AdaptVis edit: scale last-row logits on image columns, before the mask (as in the repo).
        if is_prefill and layer_on and STATE.weight != 1.0:
            last = attn[:, :, -1, :]
            last[..., pos] = last[..., pos] * STATE.weight

        if attention_mask is not None:
            attn = attn + attention_mask[:, :, :, :k_len]

        probs = nn.functional.softmax(attn, dim=-1, dtype=torch.float32)

        if STATE.log and is_prefill:
            row = probs[0, :, -1, :]                       # (heads, k_len), batch size is 1
            img = row[:, pos]                              # (heads, n_img)
            STATE.records[layer_idx] = {
                "img_mass_per_head": img.sum(-1).detach().cpu(),
                "img_dist": img.mean(0).detach().cpu(),    # head-averaged, unnormalised
                "first_token_mass": row[:, 0].mean().detach().cpu(),  # attention-sink check
            }

        probs = probs.to(query.dtype)
        probs = nn.functional.dropout(probs, p=dropout, training=module.training)
        out = torch.matmul(probs, value_states).transpose(1, 2).contiguous()
        return out, probs

    wrapper.__wrapped_original__ = original
    return wrapper


def install(model: nn.Module) -> int:
    """Patch eager attention and register the model's language-model self-attention modules.

    The model must be loaded with attn_implementation="eager". Returns the number of target layers.
    """
    impl = getattr(model.config, "_attn_implementation", None)
    text_impl = getattr(getattr(model.config, "text_config", None), "_attn_implementation", impl)
    if text_impl != "eager":
        raise RuntimeError(
            f"Load the model with attn_implementation='eager' (got {text_impl!r}); "
            "sdpa/flash attention never expose the logits we need to scale."
        )

    targets = []
    for name, mod in model.named_modules():
        if name.endswith("self_attn") and "language_model" in name and hasattr(mod, "layer_idx"):
            targets.append(mod)
    if not targets:
        raise RuntimeError("No language-model self-attention modules found; check the model class.")

    patched_any = False
    for mod_name in {type(m).__module__ for m in targets}:
        module = sys.modules.get(mod_name) or importlib.import_module(mod_name)
        if not hasattr(module, "eager_attention_forward"):
            raise RuntimeError(f"{mod_name} has no eager_attention_forward; unsupported transformers version.")
        current = module.eager_attention_forward
        if not hasattr(current, "__wrapped_original__"):
            _ORIGINALS[mod_name] = current
            module.eager_attention_forward = _make_wrapper(current)
        patched_any = True
    assert patched_any

    STATE.targets = {id(m) for m in targets}
    return len(targets)


def uninstall():
    for mod_name, orig in _ORIGINALS.items():
        sys.modules[mod_name].eager_attention_forward = orig
    _ORIGINALS.clear()
    STATE.targets = set()
