"""Pre-download every third-party model the backend needs, at Docker image build time.

The container must start without internet (spec sections 10.1 and 12.3), so this script
fills the caches that the runtime reads from:

  * CLIP (model + tokenizer)   -> $HF_HOME            (transformers / huggingface_hub cache)
  * EasyOCR English detector
    and recogniser              -> $EASYOCR_MODULE_PATH/model

The CLIP id and OCR languages are read from model_outputs/config.json so they always match
what mismatch_core.Predictor loads. Loading through the same from_pretrained / Reader calls
the app uses guarantees that every file the runtime asks for (including the "does not exist"
markers transformers checks for optional files) is present in the cache.

Usage:  python scripts/download_models.py [MODEL_DIR]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_DIR", "model_outputs")).resolve()
    cfg = json.loads((model_dir / "config.json").read_text())
    clip_id = cfg.get("clip_id", "openai/clip-vit-base-patch32")
    langs = list(cfg.get("ocr", {}).get("langs", ["en"]))

    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "0") not in ("", "0"):
            print(f"error: {var} is set; downloads would be blocked", file=sys.stderr)
            return 1

    hf_home = os.environ.get("HF_HOME", "(default)")
    ocr_home = os.environ.get("EASYOCR_MODULE_PATH", "(default)")

    t0 = time.perf_counter()
    from transformers import AutoTokenizer, CLIPModel

    # The repo's main revision only ships pytorch_model.bin; without use_safetensors=False,
    # transformers also fetches a converted model.safetensors from a Hub PR ref (+578 MB) that
    # an offline runtime can never resolve, so it would be dead weight in the image.
    CLIPModel.from_pretrained(clip_id, use_safetensors=False)
    AutoTokenizer.from_pretrained(clip_id)
    print(f"CLIP {clip_id} cached in HF_HOME={hf_home} ({time.perf_counter() - t0:.1f}s)")

    t1 = time.perf_counter()
    import easyocr

    easyocr.Reader(langs, gpu=False, download_enabled=True, verbose=False)
    print(f"EasyOCR {langs} models cached in EASYOCR_MODULE_PATH={ocr_home} ({time.perf_counter() - t1:.1f}s)")

    # Download logs from the hf-xet transfer backend are not needed at runtime.
    if "HF_HOME" in os.environ:
        shutil.rmtree(Path(os.environ["HF_HOME"]) / "xet", ignore_errors=True)

    # Prove the runtime path works with downloads disabled, exactly as the container runs.
    check = (
        "from transformers import AutoTokenizer, CLIPModel; import easyocr;"
        f"CLIPModel.from_pretrained({clip_id!r}); AutoTokenizer.from_pretrained({clip_id!r});"
        f"easyocr.Reader({langs!r}, gpu=False, download_enabled=False, verbose=False)"
    )
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run([sys.executable, "-c", check], env=env, check=True, stdout=subprocess.DEVNULL)
    print("Offline load check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
