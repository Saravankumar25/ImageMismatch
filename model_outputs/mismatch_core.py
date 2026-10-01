"""mismatch_core.py -- shared code for the training notebook AND the inference backend.

Contents
  * caption cleaning            (clean_text, keep_caption)
  * image preparation / masking (prepare_image, mask_text_regions, normalize_pixels)
  * frozen-CLIP helpers         (load_clip, clip_image, clip_text)
  * model                       (CrossAttentionHead)
  * calibration                 (apply_calibration)
  * inference                   (Predictor, NoTextInputError)

The notebook writes this file, imports it, and copies it byte-for-byte into model_outputs/,
so training and serving use exactly the same architecture and preprocessing.
Only numpy / torch / transformers / opencv / pillow (+ easyocr at inference) are needed.
"""
from __future__ import annotations

import io
import json
import math
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

__version__ = "1.0.0"

MODES = {"A": 0, "B-cap": 1, "B-ocr": 2}
MODE_NAMES = ["A", "B-cap", "B-ocr"]

# CLIPImageProcessor defaults for openai/clip-vit-base-patch32
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_IMAGE_CFG = {"work_side": 336, "image_size": 224, "dilate_px": 3, "inpaint_radius": 3}


class NoTextInputError(ValueError):
    """Raised when there is neither a user caption nor text detected in the image."""


# --------------------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def clean_text(s) -> str:
    """NFKC-normalise, strip, collapse whitespace. (CLIP's tokenizer lower-cases itself.)"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    return _WS.sub(" ", s).strip()


def n_words(s: str) -> int:
    return len(s.split()) if s else 0


def keep_caption(s: str, min_words: int = 3) -> bool:
    return n_words(s) >= min_words


def alnum_len(s: str) -> int:
    return len(re.sub(r"[^0-9A-Za-z]", "", s or ""))


# --------------------------------------------------------------------------------------
# Image preparation and text masking
# --------------------------------------------------------------------------------------
def load_image_bytes(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _resize_short(img: Image.Image, side: int) -> Image.Image:
    """Short side -> `side`, long side truncated exactly like transformers' get_resize_output_image_size."""
    w, h = img.size
    if w <= h:
        nw, nh = side, int(side * h / w)
    else:
        nw, nh = int(side * w / h), side
    if (nw, nh) == (w, h):
        return img
    return img.resize((max(1, nw), max(1, nh)), Image.BICUBIC)


def mask_text_regions(arr: np.ndarray, boxes, dilate_px: int = 3, inpaint_radius: int = 3) -> np.ndarray:
    """Inpaint axis-aligned text boxes.

    arr   : HxWx3 uint8 RGB
    boxes : iterable of [x0, y0, x1, y1] in NORMALISED [0,1] image coordinates
    Boxes are converted to pixels, dilated by `dilate_px`, and filled with cv2 TELEA inpainting.
    """
    if boxes is None or len(boxes) == 0:
        return arr
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = [float(v) for v in b[:4]]
        px0 = int(math.floor(min(x0, x1) * w)) - dilate_px
        py0 = int(math.floor(min(y0, y1) * h)) - dilate_px
        px1 = int(math.ceil(max(x0, x1) * w)) + dilate_px
        py1 = int(math.ceil(max(y0, y1) * h)) + dilate_px
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(w, px1), min(h, py1)
        if px1 > px0 and py1 > py0:
            mask[py0:py1, px0:px1] = 255
    if not mask.any():
        return arr
    return cv2.inpaint(np.ascontiguousarray(arr), mask, inpaint_radius, cv2.INPAINT_TELEA)


def prepare_image(img: Image.Image, boxes=None, image_cfg: dict | None = None) -> np.ndarray:
    """PIL image -> 224x224x3 uint8 (masked if boxes given).

    1. exif_transpose + RGB
    2. down-scale so the short side is at most `work_side` (speeds up inpainting; boxes are normalised)
    3. mask text regions (optional)
    4. CLIPProcessor geometry: resize short side to 224 (bicubic), centre-crop 224
    Normalisation (mean/std) happens on the tensor side in `normalize_pixels`.
    """
    c = dict(DEFAULT_IMAGE_CFG)
    if image_cfg:
        c.update({k: v for k, v in image_cfg.items() if k in c})
    img = ImageOps.exif_transpose(img).convert("RGB")
    if min(img.size) > c["work_side"]:
        img = _resize_short(img, c["work_side"])
    if boxes is not None and len(boxes) > 0:
        img = Image.fromarray(mask_text_regions(np.asarray(img), boxes, c["dilate_px"], c["inpaint_radius"]))
    size = c["image_size"]
    img = _resize_short(img, size)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2             # same integer centre-crop as CLIPProcessor
    img = img.crop((left, top, left + size, top + size))
    return np.array(img, dtype=np.uint8, copy=True)


def normalize_pixels(u8: torch.Tensor) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,H,W) float32 normalised with CLIP mean/std (on u8's device)."""
    x = u8.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# --------------------------------------------------------------------------------------
# Frozen CLIP helpers (work with transformers 4.x and 5.x)
# --------------------------------------------------------------------------------------
def load_clip(clip_id: str, device="cpu"):
    from transformers import AutoTokenizer, CLIPModel

    model = CLIPModel.from_pretrained(clip_id)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(clip_id)
    return model, tok


def clip_dims(clip) -> dict:
    vc, tc = clip.config.vision_config, clip.config.text_config
    grid = vc.image_size // vc.patch_size
    return {
        "img_dim": vc.hidden_size,
        "txt_dim": tc.hidden_size,
        "proj_dim": clip.config.projection_dim,
        "n_img_tokens": grid * grid + 1,
    }


@torch.no_grad()
def clip_image(clip, pixel_values: torch.Tensor):
    """-> (tokens (N,1+P,Dv) = vision last_hidden_state, pooled (N,proj) L2-normalised)."""
    out = clip.vision_model(pixel_values=pixel_values)
    pooled = clip.visual_projection(out.pooler_output)
    return out.last_hidden_state, F.normalize(pooled.float(), dim=-1)


@torch.no_grad()
def clip_text(clip, tokenizer, texts, device, max_length: int = 77):
    """-> (tokens (N,L,Dt), attention mask (N,L) bool, pooled (N,proj) L2-normalised)."""
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    out = clip.text_model(input_ids=ids, attention_mask=am)
    pooled = clip.text_projection(out.pooler_output)
    return out.last_hidden_state, am.bool(), F.normalize(pooled.float(), dim=-1)


# --------------------------------------------------------------------------------------
# Cross-attention head
# --------------------------------------------------------------------------------------
class FusionLayer(nn.Module):
    """Pre-LN block: text self-attn -> cross-attn (Q=text, K/V=image) -> FFN, all residual."""

    def __init__(self, d_model=256, n_heads=4, d_ffn=1024, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ffn, d_model), nn.Dropout(dropout)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, img, pad_mask, need_weights=False):
        h = self.ln1(x)
        a, _ = self.self_attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.drop(a)
        h = self.ln2(x)
        a, w = self.cross_attn(h, img, img, need_weights=need_weights, average_attn_weights=False)
        x = x + self.drop(a)
        x = x + self.ffn(self.ln3(x))
        return x, w


class CrossAttentionHead(nn.Module):
    """Small trainable head on top of frozen CLIP tokens.

    ITM : text tokens (+mode embedding) attend to image tokens -> masked mean-pool -> [pooled, clip_cos] -> logit
    ITC : mean image token / masked-mean text token -> 128-d, L2-normalised, learnable temperature
    """

    def __init__(self, img_dim=768, txt_dim=512, d_model=256, n_layers=2, n_heads=4, d_ffn=1024,
                 dropout=0.1, n_modes=3, itc_dim=128, temp_init=0.07, n_img_tokens=50, **_ignored):
        super().__init__()
        self.n_img_tokens = n_img_tokens
        self.img_proj = nn.Sequential(nn.Linear(img_dim, d_model), nn.LayerNorm(d_model))
        self.txt_proj = nn.Sequential(nn.Linear(txt_dim, d_model), nn.LayerNorm(d_model))
        self.mode_emb = nn.Embedding(n_modes, d_model)
        nn.init.normal_(self.mode_emb.weight, std=0.02)
        self.layers = nn.ModuleList([FusionLayer(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        self.itm = nn.Linear(d_model + 1, 1)
        self.itc_img = nn.Linear(d_model, itc_dim)
        self.itc_txt = nn.Linear(d_model, itc_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp_init)))

    # --- projections -----------------------------------------------------------------
    def proj_image(self, img_tokens):
        return self.img_proj(img_tokens.to(self.img_proj[0].weight.dtype))

    def proj_text(self, txt_tokens):
        return self.txt_proj(txt_tokens.to(self.txt_proj[0].weight.dtype))

    @staticmethod
    def masked_mean(x, mask):
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1).clamp_min(1.0)

    # --- contrastive branch ------------------------------------------------------------
    def itc_embed(self, img_p=None, txt_p=None, txt_mask=None):
        zi = zt = None
        if img_p is not None:
            zi = F.normalize(self.itc_img(img_p.mean(1)).float(), dim=-1)
        if txt_p is not None:
            zt = F.normalize(self.itc_txt(self.masked_mean(txt_p, txt_mask)).float(), dim=-1)
        return zi, zt

    def scale(self):
        return self.logit_scale.clamp(0, math.log(100.0)).exp()

    # --- matching branch ---------------------------------------------------------------
    def fuse(self, img_p, txt_p, txt_mask, mode_ids, cos, return_attn=False):
        x = txt_p + self.mode_emb(mode_ids).unsqueeze(1).to(txt_p.dtype)
        pad = ~txt_mask
        w = None
        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) - 1
            x, w = layer(x, img_p, pad, need_weights=(return_attn and last))
        pooled = self.masked_mean(self.ln_out(x), txt_mask)
        feat = torch.cat([pooled, cos.to(pooled.dtype).unsqueeze(-1)], dim=-1)
        logit = self.itm(feat).squeeze(-1)
        if not return_attn:
            return logit, None
        return logit, self.heatmap(w, txt_mask)

    def heatmap(self, w, txt_mask):
        """w: (N, heads, L, T_img) last-layer cross-attn -> (N, g, g) over patch tokens, sums to 1."""
        w = w.float().mean(1)                                   # avg heads -> (N, L, T)
        m = txt_mask.float().unsqueeze(-1)
        w = (w * m).sum(1) / m.sum(1).clamp_min(1.0)           # avg valid text tokens -> (N, T)
        w = w[:, 1:]                                            # drop CLS
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
        g = int(round(math.sqrt(w.shape[-1])))
        return w.view(-1, g, g)

    def forward(self, img_tokens, txt_tokens, txt_mask, mode_ids, cos, return_attn=False):
        img_p = self.proj_image(img_tokens)
        txt_p = self.proj_text(txt_tokens)
        return self.fuse(img_p, txt_p, txt_mask, mode_ids, cos, return_attn=return_attn)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# Calibration (pure numpy; no pickles)
# --------------------------------------------------------------------------------------
def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def apply_calibration(score, entry: dict):
    """entry = {"method": "platt", "a":..,"b":..}  or  {"method": "isotonic", "x":[...], "y":[...]}"""
    s = np.asarray(score, dtype=np.float64)
    if entry["method"] == "platt":
        return sigmoid(entry["a"] * s + entry["b"])
    if entry["method"] == "isotonic":
        return np.interp(s, np.asarray(entry["x"], dtype=np.float64), np.asarray(entry["y"], dtype=np.float64))
    if entry["method"] == "sigmoid":
        return sigmoid(s)
    raise ValueError(f"unknown calibration method {entry['method']}")


# --------------------------------------------------------------------------------------
# Predictor (the backend contract)
# --------------------------------------------------------------------------------------
class Predictor:
    """predictor = Predictor(artifacts_dir="model_outputs", device="cpu")
    result = predictor.predict(image: PIL.Image.Image, captions: list[str])
    """

    def __init__(self, artifacts_dir="model_outputs", device="cpu", ocr_reader=None, clip_model=None, tokenizer=None):
        d = Path(artifacts_dir)
        self.dir = d
        self.cfg = json.loads((d / "config.json").read_text())
        self.calibration = json.loads((d / "calibration.json").read_text())
        self.thresholds = json.loads((d / "thresholds.json").read_text())
        self.device = torch.device(device)
        if clip_model is None:
            clip_model, tokenizer = load_clip(self.cfg["clip_id"], self.device)
        self.clip, self.tok = clip_model, tokenizer
        self.head = CrossAttentionHead(**self.cfg["head_kwargs"])
        sd = torch.load(d / "head.pt", map_location="cpu", weights_only=True)
        self.head.load_state_dict(sd)
        self.head.to(self.device).eval()
        self.ocr_cfg = self.cfg["ocr"]
        self.image_cfg = self.cfg["image"]
        self.max_tokens = int(self.cfg.get("max_text_tokens", 77))
        self._reader = ocr_reader

    # --- OCR ---------------------------------------------------------------------------
    @property
    def reader(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(list(self.ocr_cfg.get("langs", ["en"])),
                                          gpu=(self.device.type == "cuda"), verbose=False)
        return self._reader

    def run_ocr(self, image: Image.Image) -> dict:
        img = ImageOps.exif_transpose(image).convert("RGB")
        W, H = img.size
        max_side = int(self.ocr_cfg.get("max_side", 1024))
        if max(W, H) > max_side:
            r = max_side / float(max(W, H))
            img = img.resize((max(1, int(round(W * r))), max(1, int(round(H * r)))), Image.BICUBIC)
        w, h = img.size
        dets = self.reader.readtext(np.asarray(img))
        min_conf = float(self.ocr_cfg.get("min_conf", 0.5))
        boxes, words = [], []
        for pts, txt, conf in dets:
            if float(conf) < min_conf:
                continue
            pts = np.asarray(pts, dtype=np.float64)
            x0, y0 = pts[:, 0].min() / w, pts[:, 1].min() / h
            x1, y1 = pts[:, 0].max() / w, pts[:, 1].max() / h
            boxes.append([float(np.clip(v, 0, 1)) for v in (x0, y0, x1, y1)])
            words.append(str(txt))
        text = clean_text(" ".join(words))
        has_text = alnum_len(text) >= int(self.ocr_cfg.get("min_chars", 3))
        return {"has_text": bool(has_text), "text": text if has_text else "", "boxes": boxes if has_text else []}

    # --- scoring -----------------------------------------------------------------------
    @torch.no_grad()
    def score(self, image: Image.Image, boxes, texts, modes):
        u8 = prepare_image(image, boxes if boxes else None, self.image_cfg)
        pix = normalize_pixels(torch.from_numpy(u8)[None].to(self.device))
        itok, ipool = clip_image(self.clip, pix)
        ttok, tmask, tpool = clip_text(self.clip, self.tok, texts, self.device, self.max_tokens)
        cos = (tpool @ ipool[0]).float()
        n = len(texts)
        mode_ids = torch.tensor([MODES[m] for m in modes], device=self.device)
        logits, heat = self.head(itok.float().expand(n, -1, -1), ttok.float(), tmask, mode_ids, cos, return_attn=True)
        return cos.cpu().numpy(), logits.float().cpu().numpy(), heat.cpu().numpy()

    def predict(self, image: Image.Image, captions=None, ocr_result: dict | None = None) -> dict:
        caps = [clean_text(c) for c in (captions or [])]
        caps = [c for c in caps if c]
        ocr = ocr_result if ocr_result is not None else self.run_ocr(image)
        if not caps and not ocr["has_text"]:
            raise NoTextInputError("No caption was given and no text was detected in the image.")
        texts, modes, sources = [], [], []
        user_mode = "B-cap" if ocr["has_text"] else "A"
        for c in caps:
            texts.append(c); modes.append(user_mode); sources.append("user")
        if ocr["has_text"]:
            texts.append(ocr["text"]); modes.append("B-ocr"); sources.append("ocr")
        cos, logits, heat = self.score(image, ocr["boxes"] if ocr["has_text"] else None, texts, modes)
        results = []
        for i, (t, m, s) in enumerate(zip(texts, modes, sources)):
            p = float(apply_calibration(logits[i], self.calibration[m]))
            thr = float(self.thresholds[m])
            results.append({
                "source": s, "text": t, "mode": m,
                "similarity": round(float(cos[i]), 6),
                "match_probability": round(p, 6),
                "threshold": round(thr, 6),
                "verdict": "MATCH" if p >= thr else "MISMATCH",
                "heatmap": np.round(heat[i], 6).tolist(),
            })
        results.sort(key=lambda r: r["match_probability"], reverse=True)
        return {"text_detected": bool(ocr["has_text"]), "ocr_text": ocr["text"], "results": results}
