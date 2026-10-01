"""Measure /api/predict latency against a running backend (spec section 7.6).

Usage (from backend/, server already running):  python scripts/latency.py [--url URL] [--runs N]

Scenarios (images from model_outputs/golden; EasyOCR runs on every request):
  a) 1 caption, plain photo (no text)             target p50 <= 2 s
  b) 5 captions, text images (OCR + masking)      target p95 <= 4 s
  c) 0 captions, text images (OCR text only)
Reports p50/p95/mean of client wall time and the server-reported latency_ms.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import httpx

GOLDEN = Path(__file__).resolve().parent.parent / "model_outputs" / "golden"
FIVE = [
    "A green note on a coffee pot says that it has a defect.",
    "A blue book called Be Here Now sits with some other books on a shelf.",
    "Lego blog website with the name of the blog across the top banner.",
    "Picture from the back side of BMW 325 Ci with number plate from Virginia.",
    "A large pizza covered in pepperoni, olives, peppers and mushrooms.",
]
SCENARIOS = [
    ("a) 1 caption, plain photo", ["coco1_coco_220446.jpg", "coco2_coco_451683.jpg"], FIVE[-1:]),
    ("b) 5 captions, text image", sorted(p.name for p in GOLDEN.glob("textcaps_*.jpg")), FIVE),
    ("c) 0 captions, OCR only", sorted(p.name for p in GOLDEN.glob("textcaps_ocronly*.jpg")), []),
]


def pct(xs: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    s = sorted(xs)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]


def post(client: httpx.Client, url: str, image: str, captions: list[str]) -> tuple[float, int]:
    data = (GOLDEN / image).read_bytes()
    t0 = time.perf_counter()
    r = client.post(f"{url}/api/predict", files={"image": (image, data, "image/jpeg")},
                    data={"captions": captions})
    wall = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return wall, r.json()["latency_ms"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--runs", type=int, default=20, help="measured requests per scenario")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    with httpx.Client(timeout=300) as client:
        if client.get(f"{url}/api/health").status_code != 200:
            print("backend is not ready (GET /api/health != 200)", file=sys.stderr)
            return 1
        print(f"{'scenario':28s} {'n':>3s} | {'wall p50':>8s} {'p95':>7s} {'mean':>7s} | "
              f"{'server p50':>10s} {'p95':>7s} {'mean':>7s}  (ms)")
        for name, images, captions in SCENARIOS:
            post(client, url, images[0], captions)  # warm-up, not counted
            wall, server = [], []
            for i in range(args.runs):  # round-robin over the scenario's images
                w, s = post(client, url, images[i % len(images)], captions)
                wall.append(w)
                server.append(s)
            print(f"{name:28s} {len(wall):3d} | {pct(wall, 50):8.0f} {pct(wall, 95):7.0f} "
                  f"{statistics.mean(wall):7.0f} | {pct(server, 50):10.0f} {pct(server, 95):7.0f} "
                  f"{statistics.mean(server):7.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
