import sys, os, json, shutil, numpy as np, pandas as pd, torch, nbformat
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from xling_spatial import data, runner, analysis, models, templates
from xling_spatial.tokenizer_stats import tokenizer_table
display = print
WORK = "/tmp/dwork"; shutil.rmtree(WORK, ignore_errors=True); os.makedirs(WORK)
COCO_ROOT = "/tmp/fakecoco"
# fake VSR from fake coco (images 0..11 with cat & dining table boxes)
rels = ["under", "on", "left of", "behind", "in front of", "above"]
rows = []
for i in range(60):
    r = rels[i % len(rels)]
    rows.append(dict(item_id=f"vsr_{i}", pair_id=f"vsr_{i}", variant="original", image=f"train2017/{i%12:012d}.jpg",
                     subj="cat", obj="dining table", relation=r, label=i % 2, text_en=f"The cat is {r} the dining table."))
data.load_vsr = lambda split, root: pd.DataFrame([dict(r, label=1) for r in rows] if split == "test" else [])
class Tok:
    unk_token_id = None
    def encode(self, t, add_special_tokens=False): return list(range(len(t) // 2 + 1))
    def convert_ids_to_tokens(self, ids): return ["x"] * len(ids)
class MockVLM:
    model_id = "mock"; kind = "qwen2_5_vl"; tokenizer = Tok()
    class processor:
        class image_processor: merge_size = 2
    def __init__(self, *a, **k): pass
    def build_inputs(self, image, text=None, raw_prompt=None):
        return {"text": text, "image_grid_thw": torch.tensor([[1, 16, 24]])}
    def score(self, inputs, answer_langs, weight=1.0, layers=None, log=False):
        t = inputs["text"]; kn = any("ಀ" <= ch <= "೿" for ch in t)
        rng = np.random.default_rng(abs(hash((t, weight))) % 2**32)
        recs = {}
        if log:
            for l in range(8):
                d = rng.random(96); d[:20] *= 1.0 if kn else 3.0; d = d / d.sum() * 0.3
                recs[l] = {"img_mass_per_head": torch.tensor([0.3]), "img_dist": torch.tensor(d), "first_token_mass": torch.tensor(.5)}
        return {"margin": rng.normal(0, 1), "conf": float(rng.random()), "top_token": "Yes", "yesno_mass": .9,
                "n_tokens": 100, "n_img_tokens": 96, "records": recs}
    token_centers = models.VLM.token_centers
VLM = MockVLM
nb = nbformat.read(_os.path.join(ROOT, "notebooks", "00_quick_demo_1hr.ipynb"), 4)
code = [c.source for c in nb.cells if c.cell_type == "code"][1:]   # skip setup
g = dict(globals())
for i, src in enumerate(code):
    src = src.replace('templates.balanced_sample(covered, N_ITEMS)', 'templates.balanced_sample(covered, 30)')
    lines = src.rstrip().splitlines()
    try:
        exec(compile(src, f"cell{i}", "exec"), g)
    except Exception as e:
        print("FAIL cell", i, type(e).__name__, e); raise
print(sorted(os.listdir(f"{WORK}/demo")))
print(open(f"{WORK}/demo/findings.txt").read()[:400])
print(pd.read_csv(f"{WORK}/demo/accuracy.csv").round(3).to_string())
