"""Model loading, input validation and the call into mismatch_core.Predictor.

All model logic lives in model_outputs/mismatch_core.py (the exact file the notebook used);
this module only validates inputs and shapes the response.
"""
from __future__ import annotations

import importlib.util
import io
import json
import logging
import sys
import threading
import time
import uuid
import warnings
from pathlib import Path
from types import ModuleType

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import ALLOWED_FORMATS, REQUIRED_MODEL_FILES, Settings
from .errors import AppError

log = logging.getLogger("app.service")

WARMUP_IMAGE = "coco1_coco_220446.jpg"
WARMUP_CAPTION = "A standing toilet hole filled with nasty filthy."

LIMITATIONS = [
    "The model measures semantic alignment between image and text, not factual truth.",
    "CLIP ViT-B/32 is weak at counting, colours and spatial relations, so swapped details "
    "(e.g. 'two dogs' vs 'three dogs') are often missed.",
    "Images are centre-cropped to 224x224, which can cut off image edges, including text.",
    "Text in the image is read with EasyOCR at inference, while training used Rosetta OCR; "
    "OCR errors lower accuracy on text-heavy images.",
    "English only.",
]


def verify_model_dir(model_dir: Path) -> None:
    """Fail fast with a clear message if any required artifact is missing."""
    if not model_dir.is_dir():
        raise RuntimeError(f"MODEL_DIR {model_dir} does not exist")
    missing = [f for f in REQUIRED_MODEL_FILES if not (model_dir / f).is_file()]
    if missing:
        raise RuntimeError(f"MODEL_DIR {model_dir} is missing required file(s): {', '.join(missing)}")


def import_core(model_dir: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("mismatch_core", model_dir / "mismatch_core.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mismatch_core"] = module
    spec.loader.exec_module(module)
    return module


def _short(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.core: ModuleType | None = None
        self.predictor = None
        self.ready = threading.Event()
        self._lock = threading.Lock()  # one inference at a time: predictable latency and memory
        self.model_info = self._build_model_info()

    # --- startup ---------------------------------------------------------------------
    def load(self) -> None:
        import torch

        s = self.settings
        t0 = time.perf_counter()
        torch.set_num_threads(s.torch_threads)
        self.core = import_core(s.model_dir)
        self.predictor = self.core.Predictor(artifacts_dir=str(s.model_dir), device="cpu")
        log.info("Predictor created in %.1fs (torch threads=%d)", time.perf_counter() - t0, s.torch_threads)

        # EasyOCR's CRAFT detector dominates latency. Channels-last memory layout (plus denormal
        # flushing, set per inference thread in analyze) makes it ~30% faster on CPU with identical
        # boxes and text on the golden set; mismatch_core.py itself is untouched.
        torch.set_flush_denormal(True)
        reader = self.predictor.reader
        detector = reader.detector.eval()
        try:  # fold each BatchNorm into its conv: one less full-resolution pass per layer, same outputs
            from torch.fx.experimental.optimization import fuse

            detector = fuse(detector, inplace=False)
        except Exception:
            log.warning("Conv-BN fusion of the OCR detector failed; using it unfused", exc_info=True)
        reader.detector = detector.to(memory_format=torch.channels_last)

        t1 = time.perf_counter()
        warm = s.model_dir / "golden" / WARMUP_IMAGE
        if warm.is_file():
            with Image.open(warm) as img:
                self.predictor.predict(img.convert("RGB"), [WARMUP_CAPTION])  # also initialises EasyOCR
        else:
            log.warning("Warm-up image %s not found; running warm-up on a blank image", warm)
            self.predictor.predict(Image.new("RGB", (224, 224), "white"), ["a white square"])
        log.info("Warm-up prediction took %.1fs; model ready (total %.1fs)",
                 time.perf_counter() - t1, time.perf_counter() - t0)
        self.ready.set()

    def _build_model_info(self) -> dict:
        d = self.settings.model_dir
        cfg = json.loads((d / "config.json").read_text())
        thresholds = json.loads((d / "thresholds.json").read_text())
        metrics_path = d / "metrics.json"
        m = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}

        keys = ("n", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "ece")
        main = m.get("test", {}).get("main", {})
        cosine = m.get("test", {}).get("cosine", {})
        test_metrics = {mode: {k: v["overall"].get(k) for k in keys} for mode, v in main.items()}
        baseline = {mode: v["overall"].get("roc_auc") for mode, v in cosine.items()}
        gap = m.get("easyocr_gap", {})
        return {
            "name": "Image–Text Match Checker",
            "model_version": self.settings.model_version,
            "architecture": "Frozen CLIP ViT-B/32 + trainable cross-attention head, calibrated per mode",
            "clip_id": cfg.get("clip_id"),
            "head_parameters": cfg.get("head_parameters"),
            "trained_epochs": cfg.get("trained_epochs"),
            "best_val_roc_auc": cfg.get("best_val_roc_auc"),
            "mismatch_core_version": cfg.get("mismatch_core_version"),
            "modes": {
                "A": "Your caption, image without text",
                "B-cap": "Your caption, image with its text masked out",
                "B-ocr": "Text found in the image vs. the masked image",
            },
            "thresholds": thresholds,
            "calibration_methods": m.get("calibration_methods", {}).get("main", {}),
            "test_metrics": test_metrics,
            "baseline_cosine_roc_auc": baseline,
            "by_negative_type": m.get("summary", {}).get("main", {}),
            "easyocr_gap": {k: {x: v.get(x) for x in ("f1", "roc_auc", "accuracy")} for k, v in gap.items()
                            if isinstance(v, dict)},
            "counts": m.get("counts", {}),
            "limitations": LIMITATIONS,
        }

    # --- validation ------------------------------------------------------------------
    def decode_image(self, data: bytes) -> Image.Image:
        s = self.settings
        if len(data) > s.max_upload_bytes:
            raise AppError("FILE_TOO_LARGE", f"Image is too large. Maximum is {s.max_upload_mb} MB.")
        Image.MAX_IMAGE_PIXELS = s.max_pixels  # Pillow refuses anything over 2x this outright
        too_many_px = AppError("FILE_TOO_LARGE", f"Image is too large. Maximum is {s.max_megapixels} megapixels.")
        try:
            with warnings.catch_warnings():  # the explicit pixel check below replaces Pillow's warning
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                img = Image.open(io.BytesIO(data))  # reads the header only
        except Image.DecompressionBombError:
            raise too_many_px
        except (UnidentifiedImageError, OSError, ValueError):
            raise AppError("UNSUPPORTED_FILE_TYPE")
        if img.format not in ALLOWED_FORMATS:  # actual format, not the file extension
            raise AppError("UNSUPPORTED_FILE_TYPE")
        w, h = img.size
        if w * h > s.max_pixels:
            raise too_many_px
        try:
            img.load()
            return ImageOps.exif_transpose(img).convert("RGB")
        except Image.DecompressionBombError:
            raise too_many_px
        except Exception:
            raise AppError("UNSUPPORTED_FILE_TYPE")

    def clean_captions(self, raw: list[str]) -> tuple[list[str], list[dict]]:
        s = self.settings
        cleaned = [self.core.clean_text(c) for c in raw]
        cleaned = [c for c in cleaned if c]  # blank boxes are ignored
        if len(cleaned) > s.max_captions:
            raise AppError("TOO_MANY_CAPTIONS", f"You can add up to {s.max_captions} captions.")
        if any(len(c) > s.max_caption_chars for c in cleaned):
            raise AppError("CAPTION_TOO_LONG", f"Each caption can be at most {s.max_caption_chars} characters.")
        seen, unique = set(), []
        for c in cleaned:
            key = c.casefold()
            if key not in seen:
                seen.add(key)
                unique.append(c)
        warnings = []
        if len(unique) < len(cleaned):
            n = len(cleaned) - len(unique)
            warnings.append({"code": "DUPLICATE_CAPTIONS_REMOVED",
                             "message": f"{n} duplicate caption{'s were' if n > 1 else ' was'} removed."})
        return unique, warnings

    def _truncation_warnings(self, results: list[dict]) -> list[dict]:
        tok, limit = self.predictor.tok, self.predictor.max_tokens
        out = []
        for r in results:
            if len(tok(r["text"], truncation=False)["input_ids"]) > limit:
                what = "Text found in the image" if r["source"] == "ocr" else f'Caption "{_short(r["text"])}"'
                out.append({"code": "CAPTION_TRUNCATED" if r["source"] == "user" else "OCR_TEXT_TRUNCATED",
                            "message": f"{what} is longer than CLIP's {limit}-token limit; "
                                       f"only the first {limit} tokens were used."})
        return out

    # --- request ---------------------------------------------------------------------
    def analyze(self, image_bytes: bytes | None, raw_captions: list[str]) -> dict:
        if not self.ready.is_set():
            raise AppError("MODEL_NOT_READY")
        t0 = time.perf_counter()
        request_id = uuid.uuid4().hex[:8]
        if not image_bytes:
            raise AppError("MISSING_IMAGE")
        image = self.decode_image(image_bytes)
        captions, warnings = self.clean_captions(raw_captions)

        import torch

        try:
            with self._lock:
                torch.set_num_threads(self.settings.torch_threads)  # per-thread in OpenMP builds
                torch.set_flush_denormal(True)  # CPU flag is per thread
                out = self.predictor.predict(image, captions)
        except self.core.NoTextInputError:
            raise AppError("NO_TEXT_INPUT")
        except Exception:
            log.exception("Inference failed (request_id=%s)", request_id)
            raise AppError("INFERENCE_FAILED")
        finally:
            image.close()
            del image

        results = [{"rank": i + 1, **r} for i, r in enumerate(out["results"])]
        warnings += self._truncation_warnings(results)
        latency_ms = int(round((time.perf_counter() - t0) * 1000))
        log.info("predict request_id=%s captions=%d text_detected=%s latency_ms=%d",
                 request_id, len(captions), out["text_detected"], latency_ms)
        return {
            "request_id": request_id,
            "text_detected": out["text_detected"],
            "ocr_text": out["ocr_text"],
            "results": results,
            "warnings": warnings,
            "latency_ms": latency_ms,
            "model_version": self.settings.model_version,
        }
