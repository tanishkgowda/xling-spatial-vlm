"""How each model's tokenizer splits English vs Hindi vs Kannada descriptions.

Report these next to any accuracy number: if a tokenizer shreds Kannada into byte pieces,
a low score says more about the tokenizer than about spatial grounding."""
from __future__ import annotations

import re
from typing import Dict, Sequence

import numpy as np
import pandas as pd

_BYTE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def tokenizer_table(tokenizers: Dict[str, object], df: pd.DataFrame, langs: Sequence[str]) -> pd.DataFrame:
    rows = []
    for name, tok in tokenizers.items():
        counts = {}
        for lang in langs:
            col = f"text_{lang}"
            texts = df[col].dropna().astype(str)
            if texts.empty:
                continue
            n_tok, n_word, n_char, n_byte = [], [], [], 0
            for t in texts:
                ids = tok.encode(t, add_special_tokens=False)
                pieces = tok.convert_ids_to_tokens(ids)
                n_byte += sum(bool(_BYTE.match(p)) for p in pieces)
                n_tok.append(len(ids)); n_word.append(len(t.split())); n_char.append(len(t))
            counts[lang] = pd.Series(n_tok, index=texts.index)
            rows.append({"tokenizer": name, "lang": lang, "n": len(texts),
                         "tokens_per_word": np.sum(n_tok) / max(np.sum(n_word), 1),
                         "chars_per_token": np.sum(n_char) / max(np.sum(n_tok), 1),
                         "byte_fallback_share": n_byte / max(np.sum(n_tok), 1),
                         "unk_share": np.mean([tok.unk_token_id in tok.encode(t, add_special_tokens=False)
                                               for t in texts]) if tok.unk_token_id is not None else 0.0})
        if "en" in counts:
            for r in rows:
                if r["tokenizer"] == name and r["lang"] in counts:
                    both = counts["en"].index.intersection(counts[r["lang"]].index)
                    r["parity_vs_en"] = float((counts[r["lang"]][both] / counts["en"][both]).mean())
    return pd.DataFrame(rows)
