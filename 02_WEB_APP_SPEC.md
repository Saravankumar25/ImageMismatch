# Spec Doc 2 — Web Application

**Project:** Image–Text Match/Mismatch Detection
**Precondition:** `model_outputs/` from Spec Doc 1 is downloaded and unzipped into `backend/model_outputs/`.
**Rule:** the app only runs inference. It never trains or retrains.

---

## 1. Overview and Scope

| In scope | Out of scope |
|---|---|
| One-page app: upload image, optional captions, Analyze | Login, accounts, users |
| Automatic text detection (OCR) in the image | Database, history, CRUD |
| MATCH / MISMATCH per caption with scores and heatmap | Storing uploaded images |
| Clear error when no text is available | Training or feedback loops |
| Docker deployment | Mobile apps |

---

## 2. User Flow

1. The user opens the page and sees the upload box, one caption box, and the Analyze button.
2. The user drops or selects an image, and a preview appears.
3. **Optional:** the user types a caption. "+ Add another caption" adds more boxes, up to 5.
4. The user clicks **Analyze**, and a loading state appears.
5. The results appear, with one card per text, ranked best match first.
6. The user can change the captions or image and analyze again, or click **Reset**.

---

## 3. Input Rules

| Input | Rule | Where checked |
|---|---|---|
| Image | Required; JPEG, PNG, or WebP | Frontend and backend |
| Image size | ≤ 10 MB; ≤ 40 megapixels | Frontend and backend |
| Captions | 0 to 5 | Frontend and backend |
| Caption length | 1 to 300 characters after trimming | Frontend and backend |
| Blank caption boxes | Ignored (not an error) | Frontend and backend |
| Duplicate captions | Removed (case-insensitive) | Backend |

- Captions are optional because an image containing text can be analyzed on its own.

---

## 4. Decision Logic (Backend)

```
image received
   │
   ▼
run OCR (EasyOCR)
   │
   ├── text found in image?
   │      YES → mask text regions
   │            ├─ each user caption  → mode B-cap
   │            └─ OCR text           → mode B-ocr  (extra row, "Text found in image")
   │
   │      NO  → user captions present?
   │            ├─ YES → each caption → mode A
   │            └─ NO  → error NO_TEXT_INPUT
   ▼
per text: CLIP → cross-attention head → calibration → per-mode threshold
   ▼
ranked results (JSON)
```

- "Text found" means at least one OCR box with confidence ≥ `OCR_MIN_CONF` and joined alphanumeric length ≥ `OCR_MIN_CHARS`. Both values come from `config.json`, so the app matches the notebook exactly.

---

## 5. System Architecture

```
Browser
   │  HTTP (same origin)
   ▼
┌───────────────────────────────┐
│ Nginx container (frontend)    │
│  - serves built React files   │
│  - proxies /api/* → backend   │
└──────────────┬────────────────┘
               │ http://backend:8000
               ▼
┌───────────────────────────────┐
│ FastAPI container (backend)   │
│  startup (once):              │
│   - load CLIP ViT-B/32        │
│   - load EasyOCR reader       │
│   - load head.pt              │
│   - load calibration.json     │
│   - load thresholds.json      │
│  per request: predict()       │
│  stateless, no disk writes    │
└───────────────────────────────┘
```

- Nginx proxies `/api`, so the browser uses one origin and **no CORS setup** is needed in production.
- All model logic comes from `model_outputs/mismatch_core.py`, the same file the notebook used. The backend does not re-implement the model.

---

## 6. Repository Structure

```
image-text-match/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app, routes, startup
│   │   ├── config.py        # env vars and limits
│   │   ├── schemas.py       # Pydantic response models
│   │   ├── service.py       # validation + calls Predictor
│   │   └── errors.py        # error codes and handlers
│   ├── model_outputs/       # unzipped from Kaggle (do not edit)
│   ├── tests/
│   │   ├── test_api.py
│   │   └── test_golden.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── components/
│   │   │   ├── ImageDropzone.jsx
│   │   │   ├── CaptionList.jsx
│   │   │   ├── AnalyzeButton.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   ├── HeatmapOverlay.jsx
│   │   │   └── ErrorBanner.jsx
│   │   └── styles.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

- The frontend is React with Vite in **plain JavaScript (.jsx)**, with no TypeScript and no UI library. Styling is plain CSS.

---

## 7. Backend Specification

### 7.1 Startup
1. Read `MODEL_DIR` (default `./model_outputs`).
2. Verify that the required files exist: `head.pt`, `mismatch_core.py`, `calibration.json`, `thresholds.json`, `config.json`. If any is missing, fail fast with a clear log message.
3. Import `Predictor` from `mismatch_core.py` and create it once with `device="cpu"`, then store it in `app.state`.
4. Run one warm-up prediction on a golden image.
5. Call `torch.set_num_threads(N)`, where N equals the CPU cores available to the container.

### 7.2 Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/predict` | Analyze an image with 0–5 captions |
| `GET` | `/api/health` | Liveness and model-loaded status |
| `GET` | `/api/model-info` | Model name, version, and headline test metrics from `metrics.json` |

### 7.3 `POST /api/predict`

**Request:** `multipart/form-data`
- `image`: a file (required)
- `captions`: a repeated text field (0–5), e.g. `captions=a dog&captions=a cat`

**Success response (200):**
```json
{
  "request_id": "b7f1c2e0",
  "text_detected": true,
  "ocr_text": "coca cola",
  "results": [
    {
      "rank": 1,
      "source": "user",
      "text": "a bottle of coke on a table",
      "mode": "B-cap",
      "verdict": "MATCH",
      "match_probability": 0.93,
      "similarity": 0.31,
      "threshold": 0.47,
      "heatmap": [[0.01, 0.02, 0.03, 0.02, 0.01, 0.01, 0.00], "... 7 rows ..."]
    },
    {
      "rank": 2,
      "source": "ocr",
      "text": "coca cola",
      "mode": "B-ocr",
      "verdict": "MATCH",
      "match_probability": 0.81,
      "similarity": 0.27,
      "threshold": 0.52,
      "heatmap": ["... 7x7 ..."]
    }
  ],
  "warnings": [],
  "latency_ms": 1840,
  "model_version": "head-v1"
}
```

- **`similarity`:** raw CLIP cosine, typically about 0.1–0.4. Show it as a number, never as a percentage.
- **`match_probability`:** the calibrated probability, shown as a percentage.
- **`warnings`:** e.g. `CAPTION_TRUNCATED` when a caption exceeds CLIP's 77 tokens.

### 7.4 Error responses

**Format:**
```json
{ "error": { "code": "NO_TEXT_INPUT", "message": "No text was provided and no text was found in the image. Add a caption or upload an image that contains text." } }
```

| HTTP | Code | When |
|---|---|---|
| 400 | `MISSING_IMAGE` | No image field |
| 400 | `TOO_MANY_CAPTIONS` | More than 5 non-blank captions |
| 400 | `CAPTION_TOO_LONG` | A caption over 300 characters |
| 413 | `FILE_TOO_LARGE` | Over 10 MB or over 40 megapixels |
| 415 | `UNSUPPORTED_FILE_TYPE` | Not decodable as JPEG, PNG, or WebP |
| 422 | `NO_TEXT_INPUT` | No captions and no text found in the image |
| 500 | `INFERENCE_FAILED` | Unexpected model error (logged with `request_id`) |
| 503 | `MODEL_NOT_READY` | Request arrives before startup completes |

### 7.5 Processing steps per request
1. Check the size limit while reading the upload, streaming into memory with a hard cap.
2. Decode with Pillow, verifying the actual format, not just the file extension.
3. Set `Image.MAX_IMAGE_PIXELS` to guard against decompression bombs.
4. Apply `ImageOps.exif_transpose`, then convert to RGB.
5. Clean the captions: trim, drop blanks, dedupe, and validate count and length.
6. Call `predictor.predict(image, captions)`. Map `NoTextInputError` to 422.
7. Add rank, request ID, warnings, and latency.
8. Discard the image from memory. **Nothing is written to disk.**

### 7.6 Performance targets (CPU, 4 cores)

| Metric | Target |
|---|---|
| Startup (model load) | ≤ 60 s |
| p50 latency, 1 caption | ≤ 2 s |
| p95 latency, 5 captions + OCR | ≤ 4 s |
| Memory (RSS) | ≤ 3 GB |

- Run uvicorn with **1 worker**, since the models are about 1+ GB. Inference runs in FastAPI's threadpool (sync route).

### 7.7 Backend dependencies
- Use the versions from `model_outputs/requirements_inference.txt`: `torch` (CPU build), `transformers`, `easyocr`, `opencv-python-headless`, `numpy`, `Pillow`.
- Add `fastapi`, `uvicorn[standard]`, `python-multipart`, and `pytest` + `httpx` for tests.

---

## 8. Frontend Specification

### 8.1 Layout (single page, centered, max width ~900 px)

```
┌──────────────────────────────────────────────┐
│  Image–Text Match Checker                    │
│  Check whether an image and text agree.      │
├──────────────────────────────────────────────┤
│  ┌────────────────────────────────────────┐  │
│  │   Drag & drop an image, or click       │  │
│  │   JPEG / PNG / WebP · max 10 MB        │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  Captions (optional if the image has text)   │
│  [ A dog running on the beach        ] [×]   │
│  + Add another caption                       │
│                                              │
│  [ Analyze ]   [ Reset ]                     │
├──────────────────────────────────────────────┤
│  Results                                     │
│  ┌ #1 ─────────────────────────── MATCH ─┐   │
│  │ "A dog running on the beach"          │   │
│  │ Match probability  ███████████░ 91%   │   │
│  │ Similarity 0.31 · Threshold 47%       │   │
│  │ [ Show heatmap ]                      │   │
│  └───────────────────────────────────────┘   │
│  ┌ Text found in image ──────── MISMATCH ┐   │
│  │ "SALE 50% OFF"                        │   │
│  │ ...                                   │   │
│  └───────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

### 8.2 Components

| Component | Responsibility |
|---|---|
| `ImageDropzone` | Drag-and-drop and click-to-select. Client-side type and size check, preview via `URL.createObjectURL`, revoke on change |
| `CaptionList` | 1–5 inputs, add and remove buttons, character counter (300), hint "Optional if the image contains text" |
| `AnalyzeButton` | Disabled until an image is selected or while loading, and shows a spinner when loading |
| `ResultCard` | Verdict badge, caption text, probability bar, similarity, threshold, source label ("Your caption" / "Text found in image") |
| `HeatmapOverlay` | Draws the 7 × 7 grid over the preview on a `<canvas>` (bilinear upscale, semi-transparent) |
| `ErrorBanner` | Shows the backend error `message`, dismissible |
| `api.js` | Builds the `FormData` and calls `POST /api/predict` with a 30 s timeout via `AbortController` |

### 8.3 UI states

| State | What the user sees |
|---|---|
| `idle` | Empty dropzone, Analyze disabled |
| `ready` | Image preview, Analyze enabled |
| `loading` | Spinner, inputs disabled, text "Analyzing…" |
| `success` | Ranked result cards |
| `error` | Error banner, inputs re-enabled, previous results cleared |

### 8.4 Display rules
1. **Badges:** MATCH is green and MISMATCH is red. Each badge also shows its text label, so color is never the only signal.
2. Show the probability as a percentage with 0 decimals and the similarity with 2 decimals.
3. Show the threshold as a marker on the probability bar.
4. The OCR row is always labelled "Text found in image" and shows the extracted text.
5. If `text_detected` is true and the user typed captions, show a small note: "Text detected in the image was removed before comparing, so the model judges the scene, not the printed words."
6. Add a footer link "About the model" that opens a modal with `/api/model-info` data (headline metrics and limitations).

### 8.5 Frontend error messages

| Code | Message shown |
|---|---|
| `NO_TEXT_INPUT` | No text was provided and no text was found in the image. Add a caption or upload an image that contains text. |
| `UNSUPPORTED_FILE_TYPE` | Please upload a JPEG, PNG, or WebP image. |
| `FILE_TOO_LARGE` | Image is too large. Maximum is 10 MB. |
| `TOO_MANY_CAPTIONS` | You can add up to 5 captions. |
| `CAPTION_TOO_LONG` | Each caption can be at most 300 characters. |
| Network or timeout | Could not reach the server. Please try again. |
| Other | Something went wrong. Please try again. |

### 8.6 Quality requirements
- Responsive down to 360 px width, with the heatmap canvas scaling to the preview.
- Keyboard accessible: the dropzone is focusable and opens on Enter or Space.
- Visible focus outlines, `aria-live="polite"` on the results region, and alt text on the preview.

---

## 9. Configuration (Environment Variables)

| Variable | Default | Used by |
|---|---|---|
| `MODEL_DIR` | `./model_outputs` | Backend |
| `MAX_UPLOAD_MB` | `10` | Backend |
| `MAX_CAPTIONS` | `5` | Backend |
| `MAX_CAPTION_CHARS` | `300` | Backend |
| `TORCH_THREADS` | number of CPU cores | Backend |
| `HF_HOME` | `/models/hf` | Backend (CLIP cache inside the image) |
| `EASYOCR_MODULE_PATH` | `/models/easyocr` | Backend (OCR models inside the image) |
| `VITE_API_BASE` | `/api` | Frontend build |

---

## 10. Deployment

1. **Backend Dockerfile:** Python slim image. Install the CPU-only torch wheel and dependencies, then copy `app/` and `model_outputs/`. **At build time, pre-download** CLIP (`openai/clip-vit-base-patch32`) and the EasyOCR English models, so the container starts without internet.
2. **Frontend Dockerfile:** a multi-stage build that runs `npm ci && npm run build` on a Node image, then copies `dist/` into Nginx.
3. **`nginx.conf`:** serve `dist/`, proxy `/api/` to `http://backend:8000/api/`, set `client_max_body_size 11m`, and use `index.html` as the fallback.
4. **`docker-compose.yml`:** two services, `frontend` (port 80 exposed) and `backend` (internal only). The backend healthcheck is `GET /api/health`, and the frontend `depends_on` the backend being healthy.
5. **Run:** `docker compose up --build`, then open `http://localhost`.
6. **Hosting:** any VM with 4 vCPU and 8 GB RAM, e.g. a college server or cloud VM. No GPU is needed.
7. **Local development without Docker:** run `uvicorn app.main:app --reload` in `backend/` and `npm run dev` in `frontend/`, with a Vite dev proxy from `/api` to `http://localhost:8000`.

---

## 11. Testing

### 11.1 Golden tests (most important)
- `test_golden.py` loads each image in `model_outputs/golden/` and calls `/api/predict` with the saved captions.
- It asserts that verdicts are identical to `golden_predictions.json` and that probabilities are within **±0.01**.
- This proves the web app reproduces the notebook exactly.

### 11.2 API tests (`test_api.py`)

| # | Scenario | Expected |
|---|---|---|
| 1 | Plain photo + matching caption | 200, mode A, MATCH |
| 2 | Plain photo + unrelated caption | 200, mode A, MISMATCH |
| 3 | Text image, no captions | 200, one `ocr` row, mode B-ocr |
| 4 | Text image + caption | 200, `user` row (B-cap) + `ocr` row (B-ocr) |
| 5 | Plain photo, no captions | 422 `NO_TEXT_INPUT` |
| 6 | Plain photo + 3 captions | 200, 3 results ranked by probability |
| 7 | Blank captions only on a plain photo | 422 `NO_TEXT_INPUT` |
| 8 | 6 captions | 400 `TOO_MANY_CAPTIONS` |
| 9 | 301-character caption | 400 `CAPTION_TOO_LONG` |
| 10 | 11 MB file | 413 `FILE_TOO_LARGE` |
| 11 | `.txt` renamed to `.jpg` | 415 `UNSUPPORTED_FILE_TYPE` |
| 12 | `GET /api/health` | 200, `model_loaded: true` |

### 11.3 Manual UI checklist
- [ ] Drag-and-drop and click-to-select both work
- [ ] Add and remove caption boxes, and a 6th box is not allowed
- [ ] Analyze stays disabled with no image
- [ ] All error messages from section 8.5 appear correctly
- [ ] Heatmap aligns with the preview at desktop and 360 px widths
- [ ] Reset clears the image, captions, and results
- [ ] Keyboard-only use works end to end

---

## 12. Acceptance Criteria

1. All golden tests and 12 API tests pass.
2. `docker compose up --build` starts the full app with no manual steps.
3. The app works offline after the build, with no model downloads at runtime.
4. No uploaded image is written to disk (verified by checking the container filesystem after requests).
5. The latency targets in section 7.6 are met on a 4-vCPU machine.
6. The UI matches section 8 on desktop and mobile widths.

---

## 13. Build Order

1. Unzip `model_outputs.zip` into `backend/model_outputs/`.
2. Write a small script that loads `Predictor` and runs the golden set. **Confirm it matches before writing any API code.**
3. Build the FastAPI app (sections 7.1–7.5), then pass the golden and API tests.
4. Build the frontend components (section 8) against the local backend.
5. Add the heatmap overlay and the "About the model" modal.
6. Write the Dockerfiles, `nginx.conf`, and `docker-compose.yml`, then run the full stack.
7. Run the manual UI checklist and latency measurement, and record the results in `README.md`.
