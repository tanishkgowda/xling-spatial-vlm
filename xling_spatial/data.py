"""Data: VSR (English), the Hindi/Kannada pilot CSV, COCO object boxes, and image-level splits."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image

PILOT_COLUMNS = ["item_id", "pair_id", "variant", "image", "subj", "obj", "relation", "label", "text_en"]


# ---------------- COCO ----------------
def find_coco_root(search_root: str = "/kaggle/input") -> str:
    """Directory that contains train2017/ (and usually val2017/, annotations/)."""
    hits = sorted(glob.glob(os.path.join(search_root, "**", "train2017"), recursive=True))
    hits = [h for h in hits if os.path.isdir(h)]
    if not hits:
        raise FileNotFoundError(
            f"No train2017/ folder under {search_root}. Add the Kaggle dataset 'awsaf49/coco-2017-dataset' "
            "(Add Input -> Datasets) or set COCO_ROOT yourself.")
    return os.path.dirname(hits[0])


def image_path(coco_root: str, rel: str) -> str:
    p = os.path.join(coco_root, rel)
    if os.path.exists(p):
        return p
    name = os.path.basename(rel)
    for sub in ("train2017", "val2017"):
        q = os.path.join(coco_root, sub, name)
        if os.path.exists(q):
            return q
    raise FileNotFoundError(rel)


def load_image(coco_root: str, rel: str, preresize=None) -> Image.Image:
    img = Image.open(image_path(coco_root, rel)).convert("RGB")
    if preresize:
        img = img.resize(tuple(preresize))
    return img


def load_coco_boxes(coco_root: str, images: List[str], cache_path: Optional[str] = None) -> Dict[str, dict]:
    """{relative image path: {category name: [[x0, y0, x1, y1] in 0..1]}} for the given images only."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if all(os.path.basename(i) in cached for i in images):
            return {i: cached[os.path.basename(i)] for i in images}

    wanted = {os.path.basename(i) for i in images}
    ann_dir = os.path.join(coco_root, "annotations")
    out: Dict[str, dict] = {}
    for split in ("train2017", "val2017"):
        fp = os.path.join(ann_dir, f"instances_{split}.json")
        if not os.path.exists(fp):
            print(f"[boxes] missing {fp}; images from {split} get no boxes")
            continue
        print(f"[boxes] reading {fp} (takes ~1 min for train2017)")
        with open(fp) as f:
            coco = json.load(f)
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        imgs = {im["id"]: im for im in coco["images"] if im["file_name"] in wanted}
        for im in imgs.values():
            out.setdefault(im["file_name"], {})
        for a in coco["annotations"]:
            im = imgs.get(a["image_id"])
            if im is None:
                continue
            x, y, w, h = a["bbox"]
            W, H = im["width"], im["height"]
            out[im["file_name"]].setdefault(cats[a["category_id"]], []).append(
                [x / W, y / H, (x + w) / W, (y + h) / H])
        del coco
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(out, f)
    return {i: out.get(os.path.basename(i), {}) for i in images}


COCO_CATEGORIES = None


def _coco_category_names(coco_root: str) -> List[str]:
    global COCO_CATEGORIES
    if COCO_CATEGORIES is None:
        fp = os.path.join(coco_root, "annotations", "instances_val2017.json")
        with open(fp) as f:
            COCO_CATEGORIES = [c["name"] for c in json.load(f)["categories"]]
    return COCO_CATEGORIES


def _parse_subj_obj(caption: str, names: List[str]):
    cap = caption.lower()
    found = []
    for n in sorted(names, key=len, reverse=True):          # longest first: "dining table" before "table"
        for m in re.finditer(r"\b" + re.escape(n) + r"\b", cap):
            if not any(s <= m.start() < e for s, e, _ in found):
                found.append((m.start(), m.end(), n))
    found.sort()
    if len(found) >= 2:
        return found[0][2], found[1][2]
    return None, None


# ---------------- VSR ----------------
def load_vsr(split: str = "test", coco_root: Optional[str] = None, repo: str = "cambridgeltl/vsr_zeroshot") -> pd.DataFrame:
    """English VSR as rows in the pilot format. split: 'val' or 'test' (AdaptVis uses vsr_zeroshot)."""
    from huggingface_hub import hf_hub_download
    fp = hf_hub_download(repo, f"{split}.jsonl", repo_type="dataset")
    rows = []
    names = _coco_category_names(coco_root) if coco_root else None
    with open(fp) as f:
        for i, line in enumerate(f):
            a = json.loads(line)
            link = a.get("image_link", "")
            m = re.search(r"\.org/(.*)", link)
            rel = m.group(1) if m else a["image"]
            subj, obj = a.get("subj"), a.get("obj")
            if (subj is None or obj is None) and names:
                subj, obj = _parse_subj_obj(a["caption"], names)
            rows.append({"item_id": f"vsr_{split}_{i}", "pair_id": f"vsr_{split}_{i}", "variant": "original",
                         "image": rel, "subj": subj, "obj": obj, "relation": a.get("relation"),
                         "label": int(a["label"]), "text_en": a["caption"]})
    return pd.DataFrame(rows)


# ---------------- pilot ----------------
def load_pilot(csv_path: str) -> pd.DataFrame:
    """Pilot / benchmark CSV (e.g. exported from the review Google Sheet).

    Required columns: item_id, pair_id, variant, image, subj, obj, relation, label, text_en
    Optional columns: text_hi, text_kn (and any other text_<lang>)
      image  : COCO path such as train2017/000000123456.jpg
      label  : 1 = statement true for the image, 0 = false
      pair_id: the same for an original and its relation-flip / object-swap versions
    """
    df = pd.read_csv(csv_path, dtype={"item_id": str, "pair_id": str})
    missing = [c for c in PILOT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}")
    df["label"] = df["label"].astype(int)
    if df["item_id"].duplicated().any():
        raise ValueError("item_id must be unique")
    return df


def languages(df: pd.DataFrame) -> List[str]:
    return [c[5:] for c in df.columns if c.startswith("text_") and df[c].notna().any()]


def split_by_image(df: pd.DataFrame, val_frac: float = 0.2, salt: str = "xling") -> pd.Series:
    """'val' / 'test' per row, decided by a hash of the image, so every language version and every
    pair variant of one image lands in the same split (FR5)."""
    def which(img):
        h = int(hashlib.md5((salt + img).encode()).hexdigest(), 16) % 10_000
        return "val" if h < val_frac * 10_000 else "test"
    return df["image"].map(which)
