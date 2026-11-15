import sys, torch, transformers
sys.path.insert(0, "" + __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))) + "")
from xling_spatial import attention
torch.manual_seed(0)

def check(name, model, inputs, img_tok):
    model.eval()
    with torch.no_grad():
        ref = model(**inputs, use_cache=False).logits[0, -1].clone()
    n = attention.install(model)
    pos = (inputs["input_ids"][0] == img_tok).nonzero().squeeze(-1)
    assert pos.numel() > 0, "no image tokens"
    def run(w, layers=None, log=False):
        attention.STATE.reset_call(pos, w, layers, log)
        with torch.no_grad():
            out = model(**inputs, use_cache=False).logits[0, -1].clone()
        attention.STATE.image_pos = None
        return out, dict(attention.STATE.records)
    same, rec = run(1.0, log=True)
    d1 = (same - ref).abs().max().item()
    off, _ = run(2.0, layers=[])
    d_off = (off - ref).abs().max().item()
    sc, _ = run(2.0)
    d_sc = (sc - ref).abs().max().item()
    sc_one, _ = run(2.0, layers=[0])
    d_one = (sc_one - ref).abs().max().item()
    r0 = rec[0]
    masses = [float(rec[l]["img_mass_per_head"].mean()) for l in sorted(rec)]
    print(f"{name}: layers={n} img_tokens={pos.numel()} | w=1 diff {d1:.2e} | w=2 no-layers diff {d_off:.2e} | "
          f"w=2 diff {d_sc:.2e} | w=2 layer0 diff {d_one:.2e} | logged layers {sorted(rec)} "
          f"| dist len {len(r0['img_dist'])} | img_mass {[round(m,3) for m in masses]}")
    assert d1 < 1e-4 and d_off < 1e-4 and d_sc > 1e-4 and d_one > 1e-5
    assert len(rec) == n and len(r0["img_dist"]) == pos.numel()
    # mass consistency: img_dist sum == mean img mass
    assert abs(float(r0["img_dist"].sum()) - float(r0["img_mass_per_head"].mean())) < 1e-5
    attention.uninstall()

# ---- Qwen2.5-VL
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
cfg = Qwen2_5_VLConfig(
    vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                       patch_size=14, spatial_merge_size=2, temporal_patch_size=2, window_size=56, fullatt_block_indexes=[1]),
    text_config=dict(vocab_size=151700, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                     num_attention_heads=4, num_key_value_heads=2, rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}),
    vocab_size=151700, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
    num_key_value_heads=2, rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
    image_token_id=151655, video_token_id=151656, vision_start_token_id=151652, attn_implementation="eager")
cfg._attn_implementation = "eager"
m = Qwen2_5_VLForConditionalGeneration(cfg)
m.config._attn_implementation = "eager"
grid = torch.tensor([[1, 4, 6]])
pix = torch.randn(24, 3 * 2 * 14 * 14)
ids = torch.tensor([[1, 2, 3, 151652] + [151655] * 6 + [151653, 5, 6, 7, 8]])
check("qwen2_5_vl", m, dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pix, image_grid_thw=grid), 151655)

# ---- LLaVA (CLIP + Llama)
from transformers import LlavaConfig, LlavaForConditionalGeneration, CLIPVisionConfig, LlamaConfig
vc = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=56, patch_size=14)
tc = LlamaConfig(vocab_size=32100, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4)
cfg = LlavaConfig(vision_config=vc, text_config=tc, image_token_index=32000, attn_implementation="eager")
m = LlavaForConditionalGeneration(cfg); m.config._attn_implementation = "eager"
ids = torch.tensor([[1, 5, 6] + [32000] * 16 + [7, 8, 9]])
check("llava", m, dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=torch.randn(1, 3, 56, 56)), 32000)

# ---- Gemma 3
from transformers import Gemma3Config, Gemma3ForConditionalGeneration, SiglipVisionConfig, Gemma3TextConfig
vc = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=56, patch_size=14)
tc = Gemma3TextConfig(vocab_size=1100, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, sliding_window=8)
cfg = Gemma3Config(vision_config=vc, text_config=tc, mm_tokens_per_image=4, image_token_index=1050, boi_token_index=1049, eoi_token_index=1051)
m = Gemma3ForConditionalGeneration(cfg); m.model.multi_modal_projector.mm_input_projection_weight.data.normal_(); m.config._attn_implementation = "eager"; m.config.text_config._attn_implementation = "eager"
ids = torch.tensor([[2, 5, 6, 1049] + [1050] * 4 + [1051, 7, 8, 9]])
tt = (ids == 1050).long()
check("gemma3", m, dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=torch.randn(1, 3, 56, 56), token_type_ids=tt), 1050)
print("ALL OK")
