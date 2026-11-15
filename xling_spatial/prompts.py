"""Prompt templates and answer words.

Two prompt modes:
  "en_instruction" (default): the instruction is always English and the model answers Yes/No.
                   Only the spatial description changes language, so any drop comes from
                   reading the description, not from the instruction or answer format.
  "native":        the instruction is in the same language as the description, and the
                   score uses native and English answer words.

The Hindi and Kannada instruction texts below are drafts. Have your native reviewers
check them before reporting any "native" results.
"""

INSTRUCTION = {
    "en": "Determine whether the description about the spatial relationship in the image is correct or not. "
          "Answer with yes or no.\nDescription: {text}",
    "hi": "चित्र में स्थानिक संबंध के बारे में दिया गया विवरण सही है या नहीं, यह बताइए। "
          "हाँ या नहीं में उत्तर दीजिए।\nविवरण: {text}",
    "kn": "ಚಿತ್ರದಲ್ಲಿನ ಸ್ಥಾನ ಸಂಬಂಧದ ಬಗ್ಗೆ ನೀಡಿರುವ ವಿವರಣೆ ಸರಿಯೇ ಅಥವಾ ತಪ್ಪೇ ಎಂದು ತಿಳಿಸಿ. "
          "ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಉತ್ತರಿಸಿ.\nವಿವರಣೆ: {text}",
}

# Exact prompt used by AdaptVis for VSR with LLaVA-1.5 (model_zoo/llava15.py, get_judge_scores_vsr_batched).
ADAPTVIS_VSR_RAW = ("User: <image>\n Determine whether the description about the spatial relationship "
                    "is correct or not. Answer with yes or no: {text} Assistant:")

YES_WORDS = {"en": ["Yes", "yes", " Yes", " yes", "YES"],
             "hi": ["हाँ", "हां", " हाँ", " हां"],
             "kn": ["ಹೌದು", " ಹೌದು"]}
NO_WORDS = {"en": ["No", "no", " No", " no", "NO"],
            "hi": ["नहीं", " नहीं"],
            "kn": ["ಇಲ್ಲ", " ಇಲ್ಲ"]}


def build_text(text: str, lang: str, mode: str = "en_instruction") -> str:
    if mode == "en_instruction":
        return INSTRUCTION["en"].format(text=text)
    if mode == "native":
        return INSTRUCTION[lang].format(text=text)
    raise ValueError(mode)


def answer_langs(lang: str, mode: str):
    return ["en"] if mode == "en_instruction" else sorted({"en", lang})


def answer_token_ids(tokenizer, langs):
    """First-token ids for yes/no words. A token that starts both a yes word and a no word
    (common for byte-fallback tokenizers on Indic scripts) is dropped from both sets."""
    yes, no = set(), set()
    for lang in langs:
        for w in YES_WORDS[lang]:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids:
                yes.add(ids[0])
        for w in NO_WORDS[lang]:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids:
                no.add(ids[0])
    clash = yes & no
    yes, no = sorted(yes - clash), sorted(no - clash)
    if not yes or not no:
        raise ValueError(f"No usable yes/no answer tokens for {langs} (clashing ids: {sorted(clash)})")
    return yes, no, sorted(clash)
