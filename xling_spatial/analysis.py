"""Metrics, per-layer attention curves and EN-vs-Indic paired comparisons."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

LANG_COLORS = {"en": "#2a6fdb", "hi": "#e07b00", "kn": "#1a9e5c", "ta": "#9b3fb5"}


def summarize(res: pd.DataFrame, by: Sequence[str] = ("lang", "weight")) -> pd.DataFrame:
    """Accuracy, pair accuracy (all variants of a pair correct), yes-rate and mean margin."""
    by = list(by)
    g = res.groupby(by)
    out = g.agg(n=("correct", "size"), acc=("correct", "mean"), yes_rate=("pred", "mean"),
                true_rate=("label", "mean"), mean_margin=("margin", "mean"),
                mean_conf=("conf", "mean"), yesno_mass=("yesno_mass", "mean"))
    pair = res.groupby(by + ["pair_id"])["correct"].min().groupby(by).mean().rename("pair_acc")
    return out.join(pair).reset_index()


def layer_table(att: pd.DataFrame, metric: str, weight: float = 1.0, items: Optional[set] = None) -> pd.DataFrame:
    """Mean and 95% CI (normal approx) of `metric` per (lang, layer)."""
    a = att[np.isclose(att.weight, weight)].dropna(subset=[metric])
    if items is not None:
        a = a[a.item_id.isin(items)]
    g = a.groupby(["lang", "layer"])[metric]
    t = g.agg(["mean", "std", "count"]).reset_index()
    t["ci"] = 1.96 * t["std"] / np.sqrt(t["count"].clip(lower=1))
    return t


def plot_layers(att: pd.DataFrame, metrics=("img_mass", "box_frac", "entropy_norm"), weight: float = 1.0,
                items: Optional[set] = None, title: str = ""):
    import matplotlib.pyplot as plt
    metrics = [m for m in metrics if m in att.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.6), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        t = layer_table(att, m, weight, items)
        for lang, d in t.groupby("lang"):
            c = LANG_COLORS.get(lang)
            ax.plot(d.layer, d["mean"], label=lang, color=c, lw=2)
            ax.fill_between(d.layer, d["mean"] - d.ci, d["mean"] + d.ci, color=c, alpha=0.15, lw=0)
        if m == "box_frac" and "area_frac" in att.columns:
            chance = att[np.isclose(att.weight, weight)]["area_frac"].mean()
            ax.axhline(chance, ls="--", color="grey", lw=1, label="chance (box area)")
        ax.set_title(m); ax.set_xlabel("layer"); ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0][0].legend(frameon=False)
    fig.suptitle(title); fig.tight_layout()
    return fig


def flip_sets(res: pd.DataFrame, lang: str, ref: str = "en", weight: float = 1.0):
    """Items answered correctly in `ref` and wrongly in `lang` ('flip'), and correctly in both ('both_ok')."""
    r = res[np.isclose(res.weight, weight)].pivot_table(index="item_id", columns="lang", values="correct")
    r = r.dropna(subset=[ref, lang])
    flip = set(r.index[(r[ref] == 1) & (r[lang] == 0)])
    both = set(r.index[(r[ref] == 1) & (r[lang] == 1)])
    return flip, both


def paired_layer_diff(att: pd.DataFrame, metric: str, lang: str, ref: str = "en", weight: float = 1.0,
                      items: Optional[set] = None) -> pd.DataFrame:
    """Per layer: mean of (lang - ref) for the same item, with a Wilcoxon signed-rank p-value."""
    from scipy.stats import wilcoxon
    a = att[np.isclose(att.weight, weight)]
    if items is not None:
        a = a[a.item_id.isin(items)]
    p = a.pivot_table(index=["item_id", "layer"], columns="lang", values=metric)
    if p.empty or ref not in p.columns or lang not in p.columns:
        return pd.DataFrame(columns=["layer", "n", "mean_diff", "p", "p_holm"])
    p = p.dropna(subset=[ref, lang])
    p["diff"] = p[lang] - p[ref]
    rows = []
    for layer, d in p.groupby(level="layer"):
        x = d["diff"].values
        pv = wilcoxon(x).pvalue if len(x) >= 6 and np.any(x != 0) else np.nan
        rows.append({"layer": layer, "n": len(x), "mean_diff": x.mean(), "p": pv})
    out = pd.DataFrame(rows)
    # Holm correction across layers
    valid = out["p"].dropna().sort_values()
    m, running = len(valid), 0.0
    out["p_holm"] = np.nan
    for rank, (i, pv) in enumerate(valid.items()):
        running = max(running, min(1.0, (m - rank) * pv))
        out.loc[i, "p_holm"] = running
    return out


def diagnose(res: pd.DataFrame, att: pd.DataFrame, lang: str, ref: str = "en", metric: str = "box_frac") -> str:
    """Plain-language reading of the flip analysis, following the Review-1 decision table."""
    flip, both = flip_sets(res, lang, ref)
    if len(flip) < 6:
        return f"Only {len(flip)} {ref}-correct/{lang}-wrong items; collect more data before reading attention differences."
    d_flip = paired_layer_diff(att, metric, lang, ref, items=flip)
    d_both = paired_layer_diff(att, metric, lang, ref, items=both)
    sig = d_flip[(d_flip.get("p_holm", d_flip["p"]) < 0.05) & (d_flip.mean_diff < 0)]
    if len(sig):
        layers = ", ".join(map(str, sig.layer.tolist()))
        return (f"On {len(flip)} flipped items, {metric} is significantly lower in {lang} than {ref} at layers {layers} "
                f"-> consistent with 'looks at the wrong part of the image'. Compare with items correct in both "
                f"(mean diff {d_both.mean_diff.mean() if len(d_both) else float('nan'):+.3f}).")
    return (f"On {len(flip)} flipped items, {metric} does not differ significantly between {lang} and {ref} at any layer "
            f"-> attention to the objects looks similar, so the failure is likely later (relation not encoded / not used). "
            f"That needs activation patching, not AdaptVis.")


def confidence_by_lang(res: pd.DataFrame, weight: float = 1.0):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for lang, d in res[np.isclose(res.weight, weight)].groupby("lang"):
        ax.hist(d["conf"], bins=20, range=(0, 1), alpha=0.45, label=lang, color=LANG_COLORS.get(lang))
    ax.set_xlabel("first-token confidence (AdaptVis gate)"); ax.set_ylabel("items"); ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def accuracy_vs_weight(res: pd.DataFrame):
    import matplotlib.pyplot as plt
    s = summarize(res)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for lang, d in s.groupby("lang"):
        ax.plot(d.weight, d.acc, marker="o", label=lang, color=LANG_COLORS.get(lang))
    ax.axvline(1.0, color="grey", lw=1, ls="--")
    ax.set_xlabel("AdaptVis weight (1.0 = no change)"); ax.set_ylabel("accuracy"); ax.legend(frameon=False)
    ax.grid(alpha=0.3); fig.tight_layout()
    return fig


def overlay(npz_path: str, image, layers=(8, 16, 24), weight: float = 1.0, title: str = ""):
    """Head-averaged attention of the last token over image tokens, drawn on the image.
    `image` must be the same PIL image given to the model (after any pre-resize)."""
    import matplotlib.pyplot as plt
    z = np.load(npz_path)
    if "centers" not in z:
        raise ValueError("No token geometry saved for this model (e.g. LLaVA-OneVision); cannot overlay.")
    maps, c = z[f"w{weight}"], z["centers"]
    xs, ys = np.unique(np.round(c[:, 0], 6)), np.unique(np.round(c[:, 1], 6))
    gw, gh = len(xs), len(ys)
    W, H = image.size
    dx, dy = (xs[1] - xs[0]) / 2 if gw > 1 else 0.5, (ys[1] - ys[0]) / 2 if gh > 1 else 0.5
    extent = [(xs[0] - dx) * W, (xs[-1] + dx) * W, (ys[-1] + dy) * H, (ys[0] - dy) * H]
    layers = [l for l in layers if l < len(maps)]
    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 3.4), squeeze=False)
    for ax, l in zip(axes[0], layers):
        m = maps[l].reshape(gh, gw)
        ax.imshow(image); ax.imshow(m / (m.max() + 1e-12), extent=extent, cmap="inferno", alpha=0.55)
        if "subj_mask" in z:
            for key, col in (("subj_mask", "cyan"), ("obj_mask", "lime")):
                pts = c[z[key]]
                if len(pts):
                    ax.scatter(pts[:, 0] * W, pts[:, 1] * H, s=4, c=col, marker="s", alpha=0.6)
        ax.set_title(f"layer {l}"); ax.axis("off")
    fig.suptitle(title); fig.tight_layout()
    return fig
