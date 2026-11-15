import sys, os, json, shutil, torch, numpy as np, pandas as pd
sys.path.insert(0, "" + __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))) + "")
from xling_spatial import attention, runner, analysis, data, models
from PIL import Image
torch.manual_seed(0); np.random.seed(0)

# 1) exact AdaptVis formula on a fake module
class FakeMod(torch.nn.Module):
    head_dim = 8; num_key_value_groups = 1; layer_idx = 0
def orig(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    w = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None: w = w + attention_mask
    p = torch.softmax(w.float(), -1).to(query.dtype)
    return torch.matmul(p, value).transpose(1, 2), p
wrap = attention._make_wrapper(orig)
mod = FakeMod(); attention.STATE.targets = {id(mod)}
q, k, v = torch.randn(1, 2, 7, 8), torch.randn(1, 2, 7, 8), torch.randn(1, 2, 7, 8)
mask = torch.triu(torch.full((7, 7), -1e9), 1)[None, None]
pos = torch.tensor([2, 3, 4])
attention.STATE.reset_call(pos, 1.5, None, True)
_, p = wrap(mod, q, k, v, mask, scaling=8 ** -0.5, dropout=0.0)
logits = (q @ k.transpose(2, 3)) * 8 ** -0.5
logits[:, :, -1, pos] *= 1.5          # AdaptVis: mask[-1, image span] -> attn_weights[:, :, mask] *= weight
exp = torch.softmax(logits + mask, -1)
assert torch.allclose(p, exp, atol=1e-6), (p - exp).abs().max()
assert torch.allclose(p[:, :, :-1], torch.softmax((q @ k.transpose(2, 3)) * 8 ** -0.5 + mask, -1)[:, :, :-1])
# non-target module passes through untouched
attention.STATE.targets = set()
_, p2 = wrap(mod, q, k, v, mask, scaling=8 ** -0.5)
assert torch.allclose(p2, torch.softmax((q @ k.transpose(2, 3)) * 8 ** -0.5 + mask, -1), atol=1e-6)
# decode step (q_len=1) not scaled
attention.STATE.targets = {id(mod)}
_, p3 = wrap(mod, q[:, :, -1:], k, v, None, scaling=8 ** -0.5)
assert torch.allclose(p3, torch.softmax((q[:, :, -1:] @ k.transpose(2, 3)) * 8 ** -0.5, -1), atol=1e-6)
print("formula OK")

# 2) geometry
class G: pass
g = G(); g.kind = "llava"; g.processor = G(); g.processor.image_processor = G(); g.processor.image_processor.do_center_crop = True
c = models.VLM.token_centers(g, {}, (640, 480), 576)
assert c.shape == (576, 2) and abs(c[:, 0].min() - ((1 - 480/640)/2 + (0.5/24)*(480/640))) < 1e-9 and abs(c[:, 1].max() - (1 - 0.5/24)) < 1e-9
g.kind = "qwen2_5_vl"; g.processor.image_processor.merge_size = 2
c = models.VLM.token_centers(g, {"image_grid_thw": torch.tensor([[1, 16, 24]])}, (640, 480), 96)
assert c.shape == (96, 2) and np.isclose(c[0, 0], 0.5/12) and np.isclose(c[0, 1], 0.5/8) and np.isclose(c[1, 0], 1.5/12)
m = models.in_boxes(c, [[0, 0, 0.5, 0.5]]); assert m.sum() == 6 * 4
print("geometry OK")

# 3) fake COCO + pilot + boxes
root = "/tmp/fakecoco"; shutil.rmtree(root, ignore_errors=True)
os.makedirs(f"{root}/train2017"); os.makedirs(f"{root}/annotations")
imgs, anns = [], []
for i in range(12):
    fn = f"{i:012d}.jpg"; Image.new("RGB", (640, 480), (i * 20, 50, 90)).save(f"{root}/train2017/{fn}")
    imgs.append({"id": i, "file_name": fn, "width": 640, "height": 480})
    anns.append({"image_id": i, "category_id": 1, "bbox": [10, 10, 200, 200]})
    anns.append({"image_id": i, "category_id": 2, "bbox": [300, 200, 200, 200]})
cats = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dining table"}, {"id": 3, "name": "table"}]
json.dump({"images": imgs, "annotations": anns, "categories": cats}, open(f"{root}/annotations/instances_train2017.json", "w"))
json.dump({"images": [], "annotations": [], "categories": cats}, open(f"{root}/annotations/instances_val2017.json", "w"))
assert data.find_coco_root("/tmp/fakecoco/..") .endswith("fakecoco")
print(data._parse_subj_obj("The cat is under the dining table.", data._coco_category_names(root)))
rows = []
for i in range(12):
    for v, lab, en in [("original", 1, "The cat is under the dining table."), ("relation_flip", 0, "The cat is above the dining table.")]:
        rows.append(dict(item_id=f"{i}_{v}", pair_id=str(i), variant=v, image=f"train2017/{i:012d}.jpg", subj="cat",
                         obj="dining table", relation="under", label=lab, text_en=en, text_kn="ಬೆಕ್ಕು ಮೇಜಿನ ಕೆಳಗೆ ಇದೆ."))
pd.DataFrame(rows).to_csv("/tmp/pilot.csv", index=False)
df = data.load_pilot("/tmp/pilot.csv"); assert data.languages(df) == ["en", "kn"]
sp = data.split_by_image(df); assert sp.groupby(df.image).nunique().max() == 1
boxes = data.load_coco_boxes(root, df.image.unique().tolist(), "/tmp/boxes.json")
assert np.allclose(boxes["train2017/000000000000.jpg"]["cat"][0], [10/640, 10/480, 210/640, 210/480])
boxes2 = data.load_coco_boxes(root, df.image.unique().tolist(), "/tmp/boxes.json"); assert boxes2 == boxes
print("data OK")

# 4) runner with a mock VLM (kn attends less to the objects and is less accurate)
class MockVLM:
    model_id = "mock"; kind = "qwen2_5_vl"
    def build_inputs(self, image, text=None, raw_prompt=None):
        return {"text": text or raw_prompt, "image_grid_thw": torch.tensor([[1, 16, 24]])}
    def score(self, inputs, answer_langs, weight=1.0, layers=None, log=False):
        kn = "ಬೆಕ್ಕು" in inputs["text"]; rng = np.random.default_rng(abs(hash((inputs["text"], weight))) % 2**32)
        recs = {}
        if log:
            for l in range(4):
                d = rng.random(96); d[:20] *= (1.0 if kn else 4.0) if l >= 2 else 1.0
                d = d / d.sum() * 0.3
                recs[l] = {"img_mass_per_head": torch.tensor([0.3, 0.3]), "img_dist": torch.tensor(d), "first_token_mass": torch.tensor(0.5)}
        margin = rng.normal(0.8 if not kn else -0.3, 1) * (1 if "under" in inputs["text"] or kn else -1)
        return {"margin": margin, "conf": float(rng.random()), "top_token": "Yes", "yesno_mass": 0.9,
                "n_tokens": 120, "n_img_tokens": 96, "records": recs}
    token_centers = models.VLM.token_centers
    processor = g.processor
mv = MockVLM()
out = "/tmp/run1"; shutil.rmtree(out, ignore_errors=True)
res = runner.run(mv, df, ["en", "kn"], root, out, weights=(1.0, 0.5, 2.0), boxes=boxes, save_maps=True, flush_every=5, limit=10)
res = runner.run(mv, df, ["en", "kn"], root, out, weights=(1.0, 0.5, 2.0), boxes=boxes, save_maps=True, flush_every=5)  # resume
assert len(res) == 24 * 2 * 3, len(res)
att = pd.read_csv(f"{out}/attention.csv", dtype={"item_id": str})
assert {"box_frac", "subj_frac", "area_frac", "entropy_norm"} <= set(att.columns) and att.box_frac.notna().all()
assert len(os.listdir(f"{out}/maps")) == 48
print(analysis.summarize(res).round(3).to_string())
best, grid = runner.tune_adaptvis(res[res.lang == "en"])
print("best", best)
print(analysis.paired_layer_diff(att, "box_frac", "kn").round(4).to_string())
print(analysis.diagnose(res, att, "kn"))
import matplotlib; matplotlib.use("Agg")
analysis.plot_layers(att).savefig("/tmp/layers.png"); analysis.confidence_by_lang(res).savefig("/tmp/conf.png"); analysis.accuracy_vs_weight(res).savefig("/tmp/acc.png")
print("pipeline OK")
