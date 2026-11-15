"""Run a model over items x languages x AdaptVis weights, with resumable CSV output.

Outputs in out_dir:
  results.csv    one row per (item, lang, weight): margin, prediction, confidence, ...
  attention.csv  one row per (item, lang, weight, layer) for logged weights: image-attention metrics
  maps/          optional .npz per (item, lang): head-averaged image attention per layer (for heatmaps)
"""
from __future__ import annotations

import os
import time
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .data import load_image
from .models import VLM, in_boxes
from .prompts import ADAPTVIS_VSR_RAW, answer_langs, build_text


def _attention_rows(records: Dict[int, dict], centers, subj_mask, obj_mask) -> list:
    rows = []
    for layer in sorted(records):
        r = records[layer]
        img_mass = float(r["img_mass_per_head"].mean())
        d = r["img_dist"].numpy().astype(np.float64)
        tot = d.sum()
        row = {"layer": layer, "img_mass": img_mass,
               "img_mass_head_max": float(r["img_mass_per_head"].max()),
               "first_token_mass": float(r["first_token_mass"])}
        if tot > 0:
            p = d / tot
            row["entropy_norm"] = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
            if centers is not None and subj_mask is not None:
                both = subj_mask | obj_mask
                row.update(subj_frac=float(p[subj_mask].sum()), obj_frac=float(p[obj_mask].sum()),
                           box_frac=float(p[both].sum()), area_frac=float(both.mean()))
        rows.append(row)
    return rows


def run(vlm: VLM, df: pd.DataFrame, langs: Sequence[str], coco_root: str, out_dir: str,
        weights: Sequence[float] = (1.0,), log_weights: Iterable[float] = (1.0,),
        prompt_mode: str = "en_instruction", raw_template: Optional[str] = None,
        preresize=None, layers=None, boxes: Optional[dict] = None, save_maps: bool = False,
        flush_every: int = 20, limit: Optional[int] = None):
    """raw_template: None for the model's chat template, or 'adaptvis_vsr' for the exact AdaptVis
    LLaVA-1.5 VSR prompt (English only). preresize: e.g. (2300, 4096) to copy AdaptVis's VSR resize."""
    os.makedirs(out_dir, exist_ok=True)
    res_fp, att_fp = os.path.join(out_dir, "results.csv"), os.path.join(out_dir, "attention.csv")
    done = set()
    if os.path.exists(res_fp):
        prev = pd.read_csv(res_fp, dtype={"item_id": str})
        done = set(zip(prev["item_id"], prev["lang"], prev["weight"].round(4)))
        print(f"[resume] {len(done)} results already in {res_fp}")
    if save_maps:
        os.makedirs(os.path.join(out_dir, "maps"), exist_ok=True)
    log_weights = {round(w, 4) for w in log_weights}

    jobs = [(r, lang) for r in df.itertuples(index=False) for lang in langs
            if isinstance(getattr(r, f"text_{lang}", None), str) and getattr(r, f"text_{lang}").strip()]
    if limit:
        jobs = jobs[:limit]
    res_buf, att_buf, t0, n_new = [], [], time.time(), 0

    def flush():
        nonlocal res_buf, att_buf
        if res_buf:
            pd.DataFrame(res_buf).to_csv(res_fp, mode="a", header=not os.path.exists(res_fp), index=False)
        if att_buf:
            pd.DataFrame(att_buf).to_csv(att_fp, mode="a", header=not os.path.exists(att_fp), index=False)
        res_buf, att_buf = [], []

    for k, (r, lang) in enumerate(jobs):
        todo = [w for w in weights if (r.item_id, lang, round(w, 4)) not in done]
        if not todo:
            continue
        text = getattr(r, f"text_{lang}")
        img = load_image(coco_root, r.image, preresize)
        if raw_template == "adaptvis_vsr":
            if lang != "en":
                raise ValueError("adaptvis_vsr raw prompt is English-only")
            inputs = vlm.build_inputs(img, raw_prompt=ADAPTVIS_VSR_RAW.format(text=text))
        else:
            inputs = vlm.build_inputs(img, text=build_text(text, lang, prompt_mode))
        a_langs = answer_langs(lang, prompt_mode)

        centers = subj_mask = obj_mask = None
        maps = {}
        for w in todo:
            log = round(w, 4) in log_weights
            s = vlm.score(inputs, a_langs, weight=w, layers=layers, log=log)
            pred = int(s["margin"] > 0)
            res_buf.append({"item_id": r.item_id, "pair_id": r.pair_id, "variant": r.variant, "lang": lang,
                            "weight": w, "label": int(r.label), "margin": s["margin"], "pred": pred,
                            "correct": int(pred == int(r.label)), "conf": s["conf"], "top_token": s["top_token"],
                            "yesno_mass": s["yesno_mass"], "n_tokens": s["n_tokens"],
                            "n_img_tokens": s["n_img_tokens"], "prompt_mode": prompt_mode, "model": vlm.model_id})
            if log and s["records"]:
                if centers is None:
                    centers = vlm.token_centers(inputs, img.size, s["n_img_tokens"])
                    if centers is not None and boxes is not None and isinstance(r.subj, str):
                        bx = boxes.get(r.image, {})
                        if r.subj in bx and isinstance(r.obj, str) and r.obj in bx:
                            subj_mask, obj_mask = in_boxes(centers, bx[r.subj]), in_boxes(centers, bx[r.obj])
                for row in _attention_rows(s["records"], centers, subj_mask, obj_mask):
                    att_buf.append({"item_id": r.item_id, "pair_id": r.pair_id, "lang": lang, "weight": w, **row})
                if save_maps:
                    maps[f"w{w}"] = np.stack([s["records"][l]["img_dist"].numpy() for l in sorted(s["records"])])
            done.add((r.item_id, lang, round(w, 4)))
        if save_maps and maps:
            extra = {"centers": centers} if centers is not None else {}
            if subj_mask is not None:
                extra.update(subj_mask=subj_mask, obj_mask=obj_mask)
            np.savez_compressed(os.path.join(out_dir, "maps", f"{r.item_id}__{lang}.npz"), **maps, **extra)
        n_new += 1
        if n_new % flush_every == 0:
            flush()
            el = time.time() - t0
            print(f"[{k + 1}/{len(jobs)}] {el / n_new:.2f}s per item-language, ~{el / n_new * (len(jobs) - k - 1) / 60:.0f} min left")
    flush()
    print(f"done: {out_dir}")
    return pd.read_csv(res_fp, dtype={"item_id": str})


# ---------------- AdaptVis from a weight sweep ----------------
def adaptvis_predictions(res: pd.DataFrame, w1: float, w2: float, threshold: float) -> pd.DataFrame:
    """AdaptVis rule from the repo: base-run confidence < threshold -> weight1, else weight2.
    Built offline from a sweep that contains weights 1.0, w1 and w2."""
    base = res[np.isclose(res.weight, 1.0)].set_index(["item_id", "lang"])
    a = res[np.isclose(res.weight, w1)].set_index(["item_id", "lang"])
    b = res[np.isclose(res.weight, w2)].set_index(["item_id", "lang"])
    out = base[["pair_id", "variant", "label", "conf"]].copy()
    use_a = out["conf"] < threshold
    out["weight_used"] = np.where(use_a, w1, w2)
    out["margin"] = np.where(use_a, a.loc[out.index, "margin"], b.loc[out.index, "margin"])
    out["pred"] = (out["margin"] > 0).astype(int)
    out["correct"] = (out["pred"] == out["label"]).astype(int)
    return out.reset_index()


def tune_adaptvis(res_val: pd.DataFrame, w1s=(0.5, 0.8), w2s=(1.2, 1.5, 2.0), thresholds=None):
    """Grid search on a validation split. Returns (best params dict, full grid DataFrame)."""
    have = set(np.round(res_val.weight.unique(), 4))
    w1s = [w for w in w1s if round(w, 4) in have]
    w2s = [w for w in w2s if round(w, 4) in have]
    if 1.0 not in have or not w1s or not w2s:
        raise ValueError(f"sweep needs weight 1.0 plus at least one weight1 and one weight2; have {sorted(have)}")
    base = res_val[np.isclose(res_val.weight, 1.0)]
    if thresholds is None:
        thresholds = sorted(set(np.round(np.quantile(base["conf"], np.linspace(0.05, 0.95, 19)), 3)))
    grid = []
    for w1 in w1s:
        for w2 in w2s:
            for t in thresholds:
                p = adaptvis_predictions(res_val, w1, w2, t)
                grid.append({"weight1": w1, "weight2": w2, "threshold": t, "acc": p["correct"].mean(),
                             "share_weight1": (p["weight_used"] == w1).mean()})
    grid = pd.DataFrame(grid).sort_values("acc", ascending=False)
    return grid.iloc[0][["weight1", "weight2", "threshold"]].to_dict(), grid
