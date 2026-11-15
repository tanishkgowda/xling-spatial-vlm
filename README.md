# Where does the Kannada drop happen? AdaptVis-based diagnosis on Kaggle

Capstone PW26_JR_05 · Phase I, Stage 3 groundwork.

**Quick demo for a presentation:** `notebooks/00_quick_demo_1hr.ipynb` (Qwen2.5-VL-3B, EN/HI/KN minimal-pair triplets from VSR with
template translations, ~25 min on Kaggle). The full plan is steps 1–4 below.

**What's here**

```
xling_spatial/        library (upload to Kaggle as a dataset, or put it in the team repo)
  attention.py        AdaptVis attention scaling + per-layer attention logging (works on LLaVA-1.5, Qwen2.5-VL, Gemma 3, LLaVA-OV)
  models.py           load a VLM with eager attention, Yes/No logit-margin scoring, image-token → image-position mapping
  data.py             VSR loader, pilot CSV loader, COCO boxes, split-by-image (FR5)
  runner.py           resumable runs over items × languages × weights; AdaptVis rule and threshold tuning
  analysis.py         accuracy / pair accuracy / yes-bias, layer curves, EN-vs-KN paired tests, heatmap overlays
  tokenizer_stats.py  tokens per word, byte-fallback share, token parity vs English
  prompts.py          prompt templates and yes/no answer words (Hindi/Kannada instructions need native review)
notebooks/            00 = quick demo; 01–04 = one per step
tests/                CPU tests (tiny random models + mock pipeline): python tests/test_tiny.py
pilot_template.csv    format for the pilot / benchmark CSV (3 example rows; replace the image path)
```

## Step 0: one-time Kaggle setup

1. **Get the code into Kaggle**, either way works:
   - **GitHub:** in the notebook's setup cell set `REPO_URL = "https://github.com/<user>/<repo>.git"` (a private repo needs a token in the URL, so use public or a dataset).
   - **Kaggle dataset:** zip the `xling_spatial/` folder and upload it as a dataset named `xling-spatial-code`.
2. **COCO images and boxes:** in each notebook, *Add Input → Datasets → `awsaf49/coco-2017-dataset`* (has `train2017/`, `val2017/`, `annotations/`).
3. **Notebook settings:** Accelerator **GPU T4 x2**, Internet **On**.
4. **Gemma 3 only:** accept the licence on the Hugging Face model page, then add a Kaggle secret named `HF_TOKEN` (*Add-ons → Secrets*).
5. **Pilot CSV:** once the 50-item pilot is reviewed, export the sheet as CSV in the `pilot_template.csv` format and upload it as a dataset (file name must start with `pilot`).

Notebook 01 pins transformers 4.39.1 (the repo's version); notebooks 02–04 pin 4.57.1. Never run them in the same session.

## Step 1: Reproduce AdaptVis (notebook 01, ~1.5–2 h GPU)

Runs the **original repo** on English VSR with LLaVA-1.5-7B: `base`, `scaling_vis` (w=0.5), `adapt_vis` (0.5 / 1.2 / threshold 0.64).
Patches applied automatically: fp16 loading (fp32 does not fit a T4), the unused `clip` import, and the attention-dump path.

*Done when* `adapt_vis` and `scaling_vis` beat `base`. **Save Version** so `output/sampled_idx_VSR.npy` and `step1_adaptvis_repo_summary.csv` are kept.

## Step 2: Port check (notebook 02, ~2–3 h)

- **Part A:** the same LLaVA-1.5 runs through *our* patch, with the repo's exact prompt and image resize. Attach notebook 01's output to use the identical test split.
  *Done when* the accuracies land close to notebook 01's (within ~1–2 points; scoring differs slightly, see "Design choices" below).
- **Part B:** Qwen2.5-VL-3B on English VSR: weight sweep, AdaptVis tuned on VSR val and tested on VSR test, plus per-layer attention curves.
  *Done when* tuned AdaptVis ≥ base and `box_frac` is above chance in middle layers. If scaling doesn't help Qwen, record that as a finding.
  Then switch to 7B (split across both T4s automatically).

## Step 3: Pilot, EN vs HI vs KN (notebook 03, ~1 h per model on 50 pairs)

Per model and language: accuracy, pair accuracy, yes-rate and `yesno_mass`. Per layer, at the answer position: `img_mass`, `box_frac` and `entropy_norm`.
Then the flip analysis (items right in English and wrong in Kannada, compared within each item, Wilcoxon with Holm correction across layers),
heatmap overlays for flipped items, and AdaptVis with English-tuned parameters.
`diagnose()` prints which branch of the Review-1 decision table the data points to.

Add Gemma 3 and LLaVA-OneVision in the `MODELS` list when ready (LLaVA-OV gets accuracy and `img_mass` only; its multi-crop layout has no single grid).

## Step 4: LLaVA-1.5 on Kannada, reported with tokenization (notebook 04, ~30 min)

Tokenizer table (tokens per word, byte-fallback share, parity vs English) next to LLaVA-1.5's accuracy, yes-rate and `yesno_mass` per language.
If Kannada is near chance with byte-level tokens, report it as "cannot read the script" and keep LLaVA-1.5 out of the morphology analysis.

## Design choices you should know about

- **Same edit as AdaptVis.** In every LM layer during prefill, the pre-softmax logits from the **last** position to **image** tokens are multiplied by `weight`.
  Tested to match the repo's formula exactly, to be identical to normal eager attention at weight 1, and to leave the vision towers untouched.
- **Scoring = Yes/No logit margin** (FR6) from a single forward pass, not a text search in the generated answer. The prediction is `margin > 0`.
- **AdaptVis confidence** = max first-token probability on the unscaled pass, as in the repo. AdaptVis results are assembled offline from a weight sweep, so thresholds can be retuned per language without rerunning the model.
- **Prompt mode.** The default `en_instruction` keeps the instruction and answer in English and changes only the description's language, which isolates the spatial sentence.
  `native` translates the instruction too; the Hindi/Kannada instruction texts in `prompts.py` are drafts for your reviewers.
- **Image size.** Qwen2.5-VL is capped at 256–512 image tokens (`min_pixels`/`max_pixels` in `VLM(...)`) so eager attention fits on a T4. Keep it fixed across languages.
- **Splits.** Pilot numbers are descriptive. Tune per-language AdaptVis only on the 500–1,000-pair set, using `data.split_by_image()`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA OOM loading LLaVA-1.5 in notebook 01 | Restart the session (nothing else on GPU 0); as a last resort use a P100 session |
| numpy import errors in notebook 01 | `!pip install -q "numpy<2"`, then restart and run all |
| `NaN` margins with Qwen | `VLM(..., dtype="float32")` (fp16 overflow); 3B fits across both T4s |
| Gemma 3 gives `inf`/`NaN` | It must run fp32 (the default here); needs both T4s |
| Session dies mid-run | Rerun the same cell: `runner.run` resumes from `results.csv` |
| `yesno_mass` very low for a language | The model isn't answering yes/no; check `top_token` before reading accuracy |
