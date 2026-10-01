# Spec Doc 1 — Model Training (Single Kaggle Notebook)

**Project:** Image–Text Match/Mismatch Detection using CLIP Embeddings
**Deliverable:** one notebook `image_text_mismatch.ipynb` → one downloadable folder `model_outputs/`
**Rule:** the web app is built only after `model_outputs/` is downloaded. The backend never trains.

---

## 1. Problem and Scope

| Item | Decision |
|---|---|
| Task | Decide if an image and a piece of text **match** or **mismatch** in meaning |
| Out of scope | Hate, fake, misleading, offensive, or factual-truth detection |
| Output | `MATCH` / `MISMATCH`, calibrated match probability, CLIP similarity, attention heatmap |
| Backbone | Frozen CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) — never fine-tuned |
| Trained part | Small cross-attention head (~2M parameters) |

**Three input modes (one model, one mode flag):**

| Mode | Image | Text | Training data |
|---|---|---|---|
| `A` | Plain photo | Typed caption | COCO Captions |
| `B-cap` | Photo with text, text **masked out** | Typed caption | TextCaps |
| `B-ocr` | Photo with text, text **masked out** | Text read from the image (OCR) | TextCaps + TextVQA OCR |

- Embedded text is masked before CLIP sees the image. Otherwise CLIP reads the words directly and similarity becomes trivially high (typographic shortcut).

---

## 2. Kaggle Environment

1. **Accelerator:** GPU (T4 or P100). Use one GPU only.
2. **Internet:** ON (Settings → Internet). Requires a phone-verified Kaggle account.
3. **Directories:**

| Path | Purpose |
|---|---|
| `/tmp/data/` | Raw downloaded zips and JSON files |
| `/tmp/cache/` | Token caches (memmaps), pooled embeddings, pair tables |
| `/kaggle/working/model_outputs/` | Final artifacts only |
| `/kaggle/working/model_outputs.zip` | Downloadable zip |

4. Run `df -h /tmp /kaggle/working` in the first cell. If `/tmp` has under ~45 GB free, set `CACHE_DIR` and `DATA_DIR` to another disk shown by `df -h`.
5. Keep `/kaggle/working` for outputs only, since it is size-capped.
6. **Run mode:** the final run uses **Save Version → Save & Run All (Commit)**, so no cell may wait for user input.
7. **Session budget:** the full run is expected to take about 2–4 h, which fits within one Kaggle GPU session (12 h limit).

---

## 3. Datasets (Official Sources Only — No Mirrors)

| File | Official URL | Size (approx.) |
|---|---|---|
| COCO train2014 images | `http://images.cocodataset.org/zips/train2014.zip` | 13 GB |
| COCO val2014 images | `http://images.cocodataset.org/zips/val2014.zip` | 6 GB |
| Karpathy split (`dataset_coco.json`) | `https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip` | 36 MB |
| TextCaps train annotations | `https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_train.json` | small |
| TextCaps val annotations | `https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_val.json` | small |
| TextVQA/TextCaps images | `https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip` | 7 GB |
| Official OCR (train) | `https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_train.json` | small |
| Official OCR (val) | `https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_val.json` | small |
| CLIP weights | `https://huggingface.co/openai/clip-vit-base-patch32` (OpenAI's own account) | 600 MB |

- **Why COCO 2014 and not 2017:** the Karpathy split is defined on 2014 image files.
- **Why TextVQA images:** TextCaps uses exactly the TextVQA image set, so no Open Images downloader is needed.
- **If a URL fails:** get the current link from the official pages (cocodataset.org/#download, textvqa.org/textcaps, textvqa.org/dataset). Never switch to a Kaggle or Hugging Face re-upload.

---

## 4. Splits (Always by Image, Never by Pair)

| Dataset | Train | Val | Test |
|---|---|---|---|
| COCO (Karpathy) | `train` + `restval` = 113,287 images | `val` = 5,000 | `test` = 5,000 |
| TextCaps | 90% of official train (~19.8k images) | 10% of official train (~2.2k images) | Official val (3,166 images) |

1. **TextCaps test:** official TextCaps test captions are hidden, so the official **val** becomes our test.
2. **Our TextCaps val** is carved from official train with a fixed seed (`SEED=42`), grouped by `image_id`.
3. **All captions of an image stay in the image's split.**
4. **Negatives are built inside each split only.** A test image is never paired with a train caption.
5. **Leakage assert:** the notebook asserts that the image-ID sets of train, val, and test are disjoint, per dataset.

---

## 5. Notebook Structure (Cell Sections in Order)

| # | Section | Output |
|---|---|---|
| 0 | Title, problem statement, scope | Markdown |
| 1 | Environment check (`nvidia-smi`, `df -h`, versions), installs (`easyocr`), seeds | Printed env |
| 2 | Config dictionary (all paths and hyperparameters in one place) | `CFG` |
| 3 | `%%writefile mismatch_core.py` — shared code (model, masking, cleaning, Predictor) | `mismatch_core.py` |
| 4 | Download official files (skip if already present) | `/tmp/data/*` |
| 5 | Schema checks on every JSON (print keys, counts, one sample) | Printed schema |
| 6 | Build image and caption index tables | `images.parquet`, `captions.parquet` |
| 7 | Dataset exploration (counts, caption lengths, sample grid) | Figures |
| 8 | Caption cleaning | Cleaned captions |
| 9 | OCR loading and masking check (draw boxes on 5 images) | Figures |
| 10 | CLIP feature extraction in chunks → memmaps | Token caches |
| 11 | Pooled text embeddings for all captions | Text embedding cache |
| 12 | Pair construction (easy / hard / swapped / text-swap) | Pair tables |
| 13 | Baseline 1: cosine threshold | Val and test scores |
| 14 | Baselines 2–3: Logistic Regression, MLP | Val and test scores |
| 15 | Main model: cross-attention head training | `head.pt` |
| 16 | Calibration and thresholds (val only) | `calibration.json`, `thresholds.json` |
| 17 | Final test evaluation (run once) | `metrics.json`, figures |
| 18 | Error analysis | Figures, tables |
| 19 | `predict()` demo + CPU latency + EasyOCR gap check | Printed results |
| 20 | Golden predictions for backend tests | `golden/` |
| 21 | Save artifacts, zip, print manifest | `model_outputs.zip` |

**Why section 3 matters:** the model class, masking, cleaning, and `Predictor` are written to `mismatch_core.py` and then imported by the notebook. The same file is copied into `model_outputs/`, so the backend uses byte-identical code. This removes any architecture mismatch when loading weights.

---

## 6. Data Loading and Memory Strategy

**Goal:** RAM stays flat during the run. Nothing large is fully loaded into the kernel.

1. **No unzipping.** Images are read straight from the zip files with `zipfile.ZipFile`, opened once per DataLoader worker (in `worker_init_fn`), never shared across processes.
2. **Lazy `Dataset`.** It stores only `(zip_name, member_path)` and indices. Images are decoded inside `__getitem__`.
3. **`DataLoader`:** `batch_size=256`, `num_workers=2`, `pin_memory=True`, `persistent_workers=True`.
4. **Chunked CLIP extraction.** Encode images in chunks of 2,048. Each chunk is written into a disk-backed `np.memmap`, then run `del`, `gc.collect()`, `torch.cuda.empty_cache()`.
5. **Cache sizes (fp16, on disk, not RAM):**

| Cache | Shape | Size |
|---|---|---|
| COCO image tokens | 123,287 × 50 × 768 | ~9.5 GB |
| TextCaps image tokens (masked) | ~25k × 50 × 768 | ~1.9 GB |
| Pooled image embeddings (both) | ~148k × 512 | ~0.15 GB |
| Pooled text embeddings (all captions + OCR strings) | ~0.8M × 512 | ~0.8 GB |

6. **Text tokens are NOT cached.** 700k+ captions × 77 × 512 would exceed 40 GB. The frozen CLIP text encoder runs per batch under `torch.no_grad()` and `torch.autocast("cuda")`.
7. **Training reads memmap slices** by index, so only the current batch of image tokens is in memory.
8. **Delete raw zips** after both caches are complete and verified (row counts plus NaN check).
9. **Memory logging.** Print RAM (`psutil`) and GPU memory every 200 steps. Values must stay roughly constant.
10. **Resume support.** Each cache writes a `done.json` marker. Reruns skip finished stages.

---

## 7. Preprocessing

### 7.1 Caption cleaning
1. Strip, collapse whitespace, normalize unicode (NFKC).
2. Drop captions with fewer than 3 words.
3. Truncate to CLIP's 77-token limit (tokenizer `truncation=True`).
4. Lowercasing is not needed, since the CLIP tokenizer lowercases.

### 7.2 OCR text for training (`B-ocr`)
1. Source: official TextVQA Rosetta OCR files (tokens plus bounding boxes).
2. OCR string = OCR tokens joined by spaces, in the file's order.
3. Images with no OCR tokens are excluded from `B-ocr` pairs (they remain in `B-cap`).
4. **Schema check first.** The expected fields are `data[].image_id`, `ocr_tokens`, and `ocr_info[].bounding_box`, with normalized coordinates. If the fields differ, adapt the loader. If boxes do not align in the visual check (section 9 of the notebook), fall back to running EasyOCR on the TextCaps images.

### 7.3 Noisy-OCR augmentation (train only)
- On `B-ocr` text, with probability 0.3 per sample, randomly drop, substitute, or change the case of characters, at a rate of ~5% of characters. This keeps the model robust to real EasyOCR mistakes at inference.

### 7.4 Text masking (`mask_text_regions` in `mismatch_core.py`)
1. Convert boxes to pixel coordinates and dilate by 3 px.
2. Fill the regions with `cv2.inpaint(..., cv2.INPAINT_TELEA)`.
3. The same function is used in training (official OCR boxes) and at inference (EasyOCR boxes).

### 7.5 Image preprocessing
- Use `CLIPProcessor` defaults: RGB, resize to 224, center crop 224, CLIP mean and std.
- Apply `ImageOps.exif_transpose` before processing.

---

## 8. Pair Construction

### 8.1 Negative types

| Type | Modes | How it is built |
|---|---|---|
| `easy` | A, B-cap | Caption from a random other image in the same split |
| `hard` | A, B-cap | Caption from another image that CLIP rates highly similar, after false-negative filtering |
| `swapped` | A, B-cap | The image's own caption with one word replaced |
| `text_swap` | B-ocr | OCR string from another image (random or nearest by CLIP text embedding, 50/50) |

### 8.2 Hard-negative mining
1. Compute image-to-caption similarity with pooled CLIP embeddings, in GPU chunks of 1,024 images, keeping `torch.topk(k=20)`.
2. **False-negative filter:** reject a candidate if its text-to-text cosine with **any** true caption of the image is ≥ 0.90, or if it is an identical string.
3. Pick uniformly at random among the top-10 survivors, not always the nearest, since the nearest is most likely a false negative.

### 8.3 Swapped negatives (rule-based, deterministic with seed)
1. Swap groups, where the swap stays within a group so the caption stays plausible:
   - **Colors:** red, blue, green, yellow, black, white, orange, pink, purple, brown, gray/grey
   - **Numbers:** one … ten, where the swap changes the value
   - **People:** man↔woman, boy↔girl, men↔women
   - **Animals:** dog, cat, horse, sheep, cow, elephant, bear, zebra, giraffe, bird
   - **Vehicles:** car, bus, truck, train, motorcycle, bicycle, airplane, boat
   - **Food:** pizza, sandwich, cake, donut, banana, apple, broccoli, carrot
2. Match whole words only, and keep simple plurals (dogs → cats).
3. If no word in the chosen caption is swappable, try the image's other captions. If none work, use a `hard` negative and record type `hard`.
4. Save the swap vocabulary inside `config.json` for the report.

### 8.4 Evaluation pair tables (val and test, built once, saved)

| Mode | Per image | Balance |
|---|---|---|
| A | 3 positives (3 different true captions) + 1 easy + 1 hard + 1 swapped | 50/50, equal share per negative type |
| B-cap | Same as A | 50/50 |
| B-ocr | 1 positive (own OCR text) + 1 text_swap | 50/50 |

- COCO test is about 30,000 pairs, and TextCaps test is about 19,000 `B-cap` pairs plus `B-ocr` pairs.
- The test pair table is saved as `test_pairs.csv` in `model_outputs/` for reproducibility.

### 8.5 Training pairs
- Positives come from all true captions.
- Offline negatives (easy, hard, swapped, text_swap) are resampled each epoch with seed `SEED + epoch`.
- Online in-batch hard negatives are added during training (section 10).

---

## 9. Feature Extraction (Frozen CLIP)

| Feature | Source (Hugging Face `transformers`) | Shape |
|---|---|---|
| Image tokens | `CLIPVisionModel(...).last_hidden_state` | 50 × 768 (1 CLS + 49 patches) |
| Pooled image embedding | `CLIPModel.get_image_features` | 512, L2-normalized |
| Text tokens | `CLIPTextModel(...).last_hidden_state` + attention mask | L × 512 (L ≤ 77) |
| Pooled text embedding | `CLIPModel.get_text_features` | 512, L2-normalized |
| CLIP cosine | dot product of the two pooled embeddings | scalar |

- CLIP runs in `eval()` mode with `requires_grad=False` everywhere.
- Mode B images are masked **before** extraction.

---

## 10. Model Architecture (Cross-Attention Head)

```
Image tokens (50×768) ──Linear→256 + LayerNorm──┐
                                                 │ keys / values
Text tokens (L×512) ──Linear→256 + LayerNorm     │
      + mode embedding (A | B-cap | B-ocr)       │
                 │                               │
     ┌───────────▼───────────────────────────────▼──┐
     │  × 2 layers (pre-LayerNorm, residuals):       │
     │   1. text self-attention (4 heads)            │
     │   2. cross-attention: Q = text, K/V = image   │
     │   3. feed-forward 256 → 1024 → 256 (GELU)     │
     └───────────────────────┬──────────────────────┘
                             │
          masked mean-pool over text tokens (256)
                             │
          concat CLIP cosine (1) → 257
                             │
                   Linear(257 → 1) = match logit (ITM)

Contrastive branch (ITC), used for training and negative mining:
  image: mean of projected image tokens → Linear(256→128) → L2 norm
  text : masked mean of projected text tokens → Linear(256→128) → L2 norm
  learnable temperature, initialized at 0.07
```

- **Dropout:** 0.1 in attention and FFN.
- **Padding:** text padding is excluded with `key_padding_mask` and the masked mean-pool.
- **Heatmap:** last-layer cross-attention weights, averaged over heads and valid text tokens, over the 49 patch tokens, reshaped to 7 × 7.

---

## 11. Training

1. **Batch:** B = 256 positive pairs, mixing COCO and TextCaps at a 2:1 ratio. Within TextCaps, `B-cap` and `B-ocr` are split 1:1.
2. **ITC loss:** symmetric InfoNCE over the B × B similarity matrix. Other captions of the same image are excluded from the negatives.
3. **Online hard negatives (BLIP-style):** for each image, sample 1 negative caption from the batch with probability proportional to softmax(ITC similarity), excluding its own captions. For each caption, sample 1 negative image the same way.
4. **ITM loss:** BCE over B positives, 2B online hard negatives, and B offline negatives from section 8.5, with `pos_weight = 3` to balance.
5. **Total loss:** `L = L_ITM + 0.5 · L_ITC`.
6. **Optimizer:** AdamW, lr 2e-4, weight decay 0.01, 500 warmup steps, then cosine decay.
7. **Epochs:** max 6. One epoch equals one pass over COCO train images.
8. **Early stopping:** monitor val ROC-AUC (all modes pooled) with patience 2, and keep the best checkpoint.
9. **Precision:** `torch.autocast("cuda", dtype=torch.float16)` plus `GradScaler`.
10. **Checkpoints:** save to `/tmp/cache/ckpt/` each epoch, and resume if present.
11. **Seeds:** `random`, `numpy`, and `torch` all set to 42, with DataLoader generator seeded too.
12. **Logging:** loss curves (train ITM/ITC, val ROC-AUC) saved as figures.

---

## 12. Baselines (Ablation)

| # | Model | Input features | Trained on |
|---|---|---|---|
| 1 | CLIP cosine threshold | Cosine similarity only | Threshold picked on val |
| 2 | Logistic Regression | `[cos, |i−t|, i⊙t]` (1025-d) | Train pairs (PyTorch, same loader) |
| 3 | MLP | Same 1025-d features → 512 → 128 → 1 (ReLU, dropout 0.2) | Train pairs |
| 4 | **Cross-attention head (main)** | Image and text tokens + cosine | Section 11 |

- Baselines 2 and 3 are implemented in PyTorch, so they use the same streaming loader and never hold the full feature matrix in RAM.
- All four use the same val tuning procedure and the same test tables.

---

## 13. Calibration and Thresholds (Val Only)

1. **Calibration per mode (A, B-cap, B-ocr):**
   - Split val by image into halves V1 and V2.
   - Fit both Platt scaling (a, b) and isotonic regression on V1.
   - Pick the one with lower ECE (15 bins) on V2, then refit it on the full val.
2. **Storage without pickle:** save to `calibration.json`, with Platt as `{a, b}` and isotonic as `{x: [...], y: [...]}`, applied with `np.interp`. This avoids scikit-learn version issues in the backend.
3. **Threshold per mode:** the calibrated-probability threshold that maximizes F1 on val, saved to `thresholds.json`.
4. **Reliability diagrams** are plotted before and after calibration, per mode.

---

## 14. Evaluation (Test Touched Once)

1. **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, ECE, and a confusion matrix.
2. **Breakdown:** per model (4) × per mode (3) × per negative type. The per-type metric uses positives vs. that type's negatives only.
3. **Figures:** ROC and PR curves, similarity histograms (match vs. each negative type), confusion matrices, and reliability diagrams.
4. **Main results table:** rows are the 4 models, and columns are modes × negative types (ROC-AUC and F1).
5. **Circularity note (write it in the report):** `hard` negatives are mined with CLIP, which disadvantages the cosine baseline by construction. `swapped` negatives are model-independent and are the fairest comparison.
6. **EasyOCR gap check:** run `B-ocr` on 500 TextCaps test images using EasyOCR instead of the official OCR, and report the metric drop.
7. **Error analysis:** show the top 20 false positives and false negatives of the main model, with image, text, probability, and negative type. Count errors by swap group (color, number, people, animals, vehicles, food).

---

## 15. `predict()` Contract (Defined in `mismatch_core.py`)

```python
predictor = Predictor(artifacts_dir="model_outputs", device="cpu")
result = predictor.predict(image: PIL.Image.Image, captions: list[str])
```

**Logic:**
1. Run EasyOCR. The image has text if at least one box has confidence ≥ `OCR_MIN_CONF` (0.5) and the joined alphanumeric length is ≥ `OCR_MIN_CHARS` (3).
2. If text is found, mask it. User captions use mode `B-cap`, and an extra row uses the OCR text with mode `B-ocr`.
3. If no text is found, user captions use mode `A`.
4. If there are no captions and no text, raise `NoTextInputError`.
5. For each text, compute CLIP cosine, head logit, calibrated probability, per-mode threshold, verdict, and a 7 × 7 heatmap.

**Return (dict):**
```json
{
  "text_detected": true,
  "ocr_text": "coca cola",
  "results": [
    {"source": "user", "text": "a bottle of coke on a table", "mode": "B-cap",
     "similarity": 0.31, "match_probability": 0.93, "threshold": 0.47,
     "verdict": "MATCH", "heatmap": [[0.01, "... 7x7 ..."]]},
    {"source": "ocr", "text": "coca cola", "mode": "B-ocr",
     "similarity": 0.27, "match_probability": 0.81, "threshold": 0.52,
     "verdict": "MATCH", "heatmap": [[0.02, "... 7x7 ..."]]}
  ]
}
```
- Results are sorted by `match_probability`, descending.
- Measure CPU latency in the notebook with `device="cpu"`, as the median of 20 runs, and store it in `metrics.json`.

---

## 16. Artifacts and Download

```
model_outputs/
├── head.pt                     # state_dict of the best checkpoint
├── mismatch_core.py            # model class, masking, cleaning, Predictor
├── calibration.json            # per-mode Platt or isotonic parameters
├── thresholds.json             # per-mode decision thresholds
├── config.json                 # hyperparameters, CLIP id, OCR settings, seeds, swap vocab,
│                               # library versions (torch, transformers, easyocr, opencv)
├── metrics.json                # all test metrics, per model/mode/type, CPU latency
├── requirements_inference.txt  # pinned versions from the notebook environment
├── test_pairs.csv              # frozen test tables
├── figures/                    # ROC, PR, reliability, histograms, confusion, loss curves
└── golden/                     # 6 sample images + golden_predictions.json
```

1. **Golden set:** 2 COCO test images and 4 TextCaps test images (2 with captions, 2 OCR-only), plus their exact `predict()` outputs. The backend must reproduce these within ±0.01 probability.
2. **Zip:** `shutil.make_archive("/kaggle/working/model_outputs", "zip", "/kaggle/working/model_outputs")`.
3. **Manifest:** print every file with its size and SHA-256 hash.
4. **Download:** after the committed run finishes, open the notebook's **Output** tab and download `model_outputs.zip`.
5. **On your machine:** unzip into `backend/model_outputs/`.
6. CLIP weights are **not** saved. The backend loads `openai/clip-vit-base-patch32` from Hugging Face by the name in `config.json`.

---

## 17. Acceptance Checklist

- [ ] Every data file downloaded from the official URLs in section 3
- [ ] Split disjointness asserts pass for both datasets
- [ ] RAM stays flat during extraction and training (log shown)
- [ ] Masking visually verified on sample images
- [ ] All 4 models evaluated on identical test tables
- [ ] Threshold and calibration fitted on val only; test evaluated exactly once
- [ ] Per-mode and per-negative-type metrics present in `metrics.json`
- [ ] `predict()` works on CPU for all 4 input scenarios, including the no-text error
- [ ] Golden predictions saved
- [ ] Full notebook passes **Save & Run All** without manual steps
- [ ] `model_outputs.zip` downloaded and unzipped locally

---

## 18. Known Limitations (Write in Report)

1. The system measures semantic alignment, not truth.
2. CLIP ViT-B/32 is weak on counting and spatial relations, and the swap results will show this.
3. Center cropping to 224 can cut off image edges, including text.
4. Inference OCR (EasyOCR) differs from the training OCR (Rosetta). The gap is measured in section 14.6.
5. The system handles English only.
