"""Template Hindi / Kannada versions of VSR statements, for a quick DEMO only.

"subject  object+case  postposition  is" is built from small hand-written tables. The output is grammatical
for the covered nouns and relations, but it is NOT the native-validated benchmark: label any result from it
as "template translations, preliminary".

Kannada: {subj} {obj-GEN} {postposition} ಇದೆ      e.g. ಬೆಕ್ಕು ಮೇಜಿನ ಕೆಳಗೆ ಇದೆ  (cat table-GEN below is)
Hindi:   {subj} {obj-OBLIQUE} {postposition} है   e.g. बिल्ली मेज़ के नीचे है
"""
from __future__ import annotations

import pandas as pd

# COCO category -> (Hindi nominative, Hindi oblique, Kannada nominative, Kannada genitive)
NOUNS = {
    "person": ("व्यक्ति", "व्यक्ति", "ವ್ಯಕ್ತಿ", "ವ್ಯಕ್ತಿಯ"),
    "cat": ("बिल्ली", "बिल्ली", "ಬೆಕ್ಕು", "ಬೆಕ್ಕಿನ"),
    "dog": ("कुत्ता", "कुत्ते", "ನಾಯಿ", "ನಾಯಿಯ"),
    "chair": ("कुर्सी", "कुर्सी", "ಕುರ್ಚಿ", "ಕುರ್ಚಿಯ"),
    "couch": ("सोफ़ा", "सोफ़े", "ಸೋಫಾ", "ಸೋಫಾದ"),
    "bed": ("बिस्तर", "बिस्तर", "ಹಾಸಿಗೆ", "ಹಾಸಿಗೆಯ"),
    "dining table": ("मेज़", "मेज़", "ಮೇಜು", "ಮೇಜಿನ"),
    "car": ("कार", "कार", "ಕಾರು", "ಕಾರಿನ"),
    "bench": ("बेंच", "बेंच", "ಬೆಂಚು", "ಬೆಂಚಿನ"),
    "bicycle": ("साइकिल", "साइकिल", "ಸೈಕಲ್", "ಸೈಕಲ್‌ನ"),
    "umbrella": ("छाता", "छाते", "ಛತ್ರಿ", "ಛತ್ರಿಯ"),
    "cup": ("कप", "कप", "ಕಪ್", "ಕಪ್‌ನ"),
    "laptop": ("लैपटॉप", "लैपटॉप", "ಲ್ಯಾಪ್‌ಟಾಪ್", "ಲ್ಯಾಪ್‌ಟಾಪ್‌ನ"),
    "tv": ("टीवी", "टीवी", "ಟಿವಿ", "ಟಿವಿಯ"),
    "book": ("किताब", "किताब", "ಪುಸ್ತಕ", "ಪುಸ್ತಕದ"),
    "teddy bear": ("टेडी बियर", "टेडी बियर", "ಟೆಡ್ಡಿ ಬೇರ್", "ಟೆಡ್ಡಿ ಬೇರ್‌ನ"),
    "horse": ("घोड़ा", "घोड़े", "ಕುದುರೆ", "ಕುದುರೆಯ"),
    "truck": ("ट्रक", "ट्रक", "ಟ್ರಕ್", "ಟ್ರಕ್‌ನ"),
    "suitcase": ("सूटकेस", "सूटकेस", "ಸೂಟ್‌ಕೇಸ್", "ಸೂಟ್‌ಕೇಸ್‌ನ"),
    "potted plant": ("पौधा", "पौधे", "ಗಿಡ", "ಗಿಡದ"),
    "bowl": ("कटोरा", "कटोरे", "ಬಟ್ಟಲು", "ಬಟ್ಟಲಿನ"),
    "bus": ("बस", "बस", "ಬಸ್", "ಬಸ್‌ನ"),
    "motorcycle": ("मोटरसाइकिल", "मोटरसाइकिल", "ಮೋಟಾರ್‌ಸೈಕಲ್", "ಮೋಟಾರ್‌ಸೈಕಲ್‌ನ"),
    "cow": ("गाय", "गाय", "ಹಸು", "ಹಸುವಿನ"),
    "elephant": ("हाथी", "हाथी", "ಆನೆ", "ಆನೆಯ"),
    "bird": ("चिड़िया", "चिड़िया", "ಹಕ್ಕಿ", "ಹಕ್ಕಿಯ"),
    "sink": ("सिंक", "सिंक", "ಸಿಂಕ್", "ಸಿಂಕ್‌ನ"),
    "refrigerator": ("फ्रिज", "फ्रिज", "ಫ್ರಿಡ್ಜ್", "ಫ್ರಿಡ್ಜ್‌ನ"),
    "handbag": ("हैंडबैग", "हैंडबैग", "ಕೈಚೀಲ", "ಕೈಚೀಲದ"),
    "pizza": ("पिज़्ज़ा", "पिज़्ज़ा", "ಪಿಜ್ಜಾ", "ಪಿಜ್ಜಾದ"),
    "cake": ("केक", "केक", "ಕೇಕ್", "ಕೇಕ್‌ನ"),
    "giraffe": ("जिराफ़", "जिराफ़", "ಜಿರಾಫೆ", "ಜಿರಾಫೆಯ"),
    "clock": ("घड़ी", "घड़ी", "ಗಡಿಯಾರ", "ಗಡಿಯಾರದ"),
    "vase": ("फूलदान", "फूलदान", "ಹೂದಾನಿ", "ಹೂದಾನಿಯ"),
    "train": ("ट्रेन", "ट्रेन", "ರೈಲು", "ರೈಲಿನ"),
    "boat": ("नाव", "नाव", "ದೋಣಿ", "ದೋಣಿಯ"),
    "sheep": ("भेड़", "भेड़", "ಕುರಿ", "ಕುರಿಯ"),
    "oven": ("ओवन", "ओवन", "ಒಲೆ", "ಒಲೆಯ"),
    "toilet": ("टॉयलेट", "टॉयलेट", "ಟಾಯ್ಲೆಟ್", "ಟಾಯ್ಲೆಟ್‌ನ"),
    "keyboard": ("कीबोर्ड", "कीबोर्ड", "ಕೀಬೋರ್ಡ್", "ಕೀಬೋರ್ಡ್‌ನ"),
}

# VSR relation -> (Hindi postposition phrase, Kannada postposition, canonical relation)
RELATIONS = {
    "on": ("पर", "ಮೇಲೆ", "on"),
    "on top of": ("के ऊपर", "ಮೇಲೆ", "on"),
    "above": ("के ऊपर", "ಮೇಲೆ", "above"),
    "under": ("के नीचे", "ಕೆಳಗೆ", "under"),
    "beneath": ("के नीचे", "ಕೆಳಗೆ", "under"),
    "below": ("के नीचे", "ಕೆಳಗೆ", "under"),
    "left of": ("के बाईं ओर", "ಎಡಭಾಗದಲ್ಲಿ", "left"),
    "at the left side of": ("के बाईं ओर", "ಎಡಭಾಗದಲ್ಲಿ", "left"),
    "right of": ("के दाईं ओर", "ಬಲಭಾಗದಲ್ಲಿ", "right"),
    "at the right side of": ("के दाईं ओर", "ಬಲಭಾಗದಲ್ಲಿ", "right"),
    "in front of": ("के सामने", "ಮುಂದೆ", "front"),
    "behind": ("के पीछे", "ಹಿಂದೆ", "behind"),
    "at the back of": ("के पीछे", "ಹಿಂದೆ", "behind"),
    "inside": ("के अंदर", "ಒಳಗೆ", "inside"),
    "in": ("के अंदर", "ಒಳಗೆ", "inside"),
}


def hindi(subj: str, obj: str, rel: str) -> str:
    s, o, p = NOUNS[subj][0], NOUNS[obj][1], RELATIONS[rel][0]
    return f"{s} {o} {p} है।"


def kannada(subj: str, obj: str, rel: str) -> str:
    s, o, p = NOUNS[subj][2], NOUNS[obj][3], RELATIONS[rel][1]
    verb = "ಇದ್ದಾರೆ" if subj == "person" else "ಇದೆ"
    return f"{s} {o} {p} {verb}."


def add_template_translations(df: pd.DataFrame) -> pd.DataFrame:
    """Keep VSR rows whose subject, object and relation are covered; add text_hi / text_kn.
    English stays the original VSR caption."""
    ok = df.subj.isin(NOUNS) & df.obj.isin(NOUNS) & df.relation.isin(RELATIONS) & (df.subj != df.obj)
    out = df[ok].copy()
    out["text_hi"] = [hindi(r.subj, r.obj, r.relation) for r in out.itertuples()]
    out["text_kn"] = [kannada(r.subj, r.obj, r.relation) for r in out.itertuples()]
    out["relation_group"] = out.relation.map(lambda r: RELATIONS[r][2])
    return out.reset_index(drop=True)


def balanced_sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Up to n rows, half true / half false, spread over relation groups."""
    parts = []
    for lab in (1, 0):
        d = df[df.label == lab].sample(frac=1, random_state=seed).copy()
        d["_r"] = d.groupby("relation_group").cumcount()      # round-robin over relations
        parts.append(d.sort_values("_r", kind="stable").head(n // 2))
    return pd.concat(parts).drop(columns="_r").sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------- minimal pairs (Review 2: original / relation flip / object swap) ----------------
# canonical relation -> (English phrase, Hindi postposition, Kannada postposition)
CANON = {
    "on": ("on", "पर", "ಮೇಲೆ"),
    "above": ("above", "के ऊपर", "ಮೇಲೆ"),
    "under": ("under", "के नीचे", "ಕೆಳಗೆ"),
    "left": ("to the left of", "के बाईं ओर", "ಎಡಭಾಗದಲ್ಲಿ"),
    "right": ("to the right of", "के दाईं ओर", "ಬಲಭಾಗದಲ್ಲಿ"),
    "front": ("in front of", "के सामने", "ಮುಂದೆ"),
    "behind": ("behind", "के पीछे", "ಹಿಂದೆ"),
}
OPPOSITE = {"on": "under", "above": "under", "under": "above", "left": "right", "right": "left",
            "front": "behind", "behind": "front"}   # "inside" has no clean opposite: not used for pairs


def _texts(subj: str, obj: str, canon: str) -> dict:
    en, hi, kn = CANON[canon]
    verb = "ಇದ್ದಾರೆ" if subj == "person" else "ಇದೆ"
    return {"text_en": f"The {subj} is {en} the {obj}.",
            "text_hi": f"{NOUNS[subj][0]} {NOUNS[obj][1]} {hi} है।",
            "text_kn": f"{NOUNS[subj][2]} {NOUNS[obj][3]} {kn} {verb}."}


def build_minimal_pairs(covered: pd.DataFrame, n_pairs: int, seed: int = 0) -> pd.DataFrame:
    """From TRUE VSR statements, build triplets on the same image:
      original      (label 1)  The cat is under the table.   ಬೆಕ್ಕು ಮೇಜಿನ ಕೆಳಗೆ ಇದೆ
      relation_flip (label 0)  The cat is above the table.   ಬೆಕ್ಕು ಮೇಜಿನ ಮೇಲೆ ಇದೆ      (postposition changes)
      object_swap   (label 0)  The table is under the cat.   ಮೇಜು ಬೆಕ್ಕಿನ ಕೆಳಗೆ ಇದೆ     (the genitive suffix moves)
    All three are template-built in every language, so the wording style is identical across variants."""
    src = covered[(covered.label == 1) & covered.relation_group.isin(OPPOSITE)].sample(frac=1, random_state=seed).copy()
    src["_r"] = src.groupby("relation_group").cumcount()
    src = src.sort_values("_r", kind="stable").head(n_pairs)
    rows = []
    for k, r in enumerate(src.itertuples()):
        pid = f"p{k:03d}"
        base = {"pair_id": pid, "image": r.image, "vsr_caption": r.text_en}
        for variant, subj, obj, rel, lab in [("original", r.subj, r.obj, r.relation_group, 1),
                                             ("relation_flip", r.subj, r.obj, OPPOSITE[r.relation_group], 0),
                                             ("object_swap", r.obj, r.subj, r.relation_group, 0)]:
            rows.append({**base, "item_id": f"{pid}_{variant}", "variant": variant, "subj": subj, "obj": obj,
                         "relation": rel, "relation_group": r.relation_group, "label": lab, **_texts(subj, obj, rel)})
    return pd.DataFrame(rows)
