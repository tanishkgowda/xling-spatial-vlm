"""Cross-lingual spatial grounding diagnostics for VLMs (Capstone PW26_JR_05).

AdaptVis attention scaling (Chen et al., ICML 2025) re-implemented as an attention patch so the
same code runs on LLaVA-1.5, Qwen2.5-VL, Gemma 3 and LLaVA-OneVision, plus per-layer image-attention
logging and Yes/No logit-margin scoring.
"""
__version__ = "0.1.0"
