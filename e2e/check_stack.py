"""End-to-end check of the running Docker stack, through Nginx exactly as a browser sees it.

Usage:  python e2e/check_stack.py [BASE_URL]      (default http://localhost)
Needs httpx and Pillow. Exits non-zero on any failure.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost").rstrip("/")
ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "backend" / "model_outputs" / "golden"
PLAIN = (GOLDEN / "coco1_coco_220446.jpg").read_bytes()
PIZZA = (GOLDEN / "coco2_coco_451683.jpg").read_bytes()
TEXT_IMG = (GOLDEN / "textcaps_ocronly2_tc_2659571273d85bbd.jpg").read_bytes()

client = httpx.Client(base_url=BASE, timeout=60)
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail and not ok else ''}")
    if not ok:
        failures.append(name)


def predict(image: bytes | None, captions=(), filename="photo.jpg", ctype="image/jpeg"):
    files = [("image", (filename, image, ctype))] if image is not None else [("x", (None, b""))]
    t0 = time.perf_counter()
    r = client.post("/api/predict", files=files, data={"captions": list(captions)} if captions else {})
    return r, (time.perf_counter() - t0) * 1000


def code(r) -> str | None:
    try:
        return r.json()["error"]["code"]
    except Exception:
        return None


# --- static frontend -------------------------------------------------------------------
r = client.get("/")
check("GET / serves the React app", r.status_code == 200 and '<div id="root">' in r.text)
asset = next((p for p in r.text.split('"') if p.startswith("/assets/") and p.endswith(".js")), None)
ra = client.get(asset) if asset else None
check("JS bundle served with immutable caching", bool(ra) and ra.status_code == 200
      and "immutable" in ra.headers.get("cache-control", ""))
check("SPA fallback for unknown routes", client.get("/some/route").status_code == 200)
check("example image served", client.get("/examples/pizza.jpg").headers.get("content-type") == "image/jpeg")

# --- health / model info ---------------------------------------------------------------
h = client.get("/api/health")
check("12. GET /api/health -> 200 model_loaded", h.status_code == 200 and h.json().get("model_loaded") is True, h.text)
mi = client.get("/api/model-info").json()
check("GET /api/model-info has metrics + limitations", set(mi.get("test_metrics", {})) == {"A", "B-cap", "B-ocr"}
      and bool(mi.get("limitations")))

# --- golden set through the full stack -------------------------------------------------
spec = json.loads((GOLDEN / "golden_predictions.json").read_text())
tol = spec.get("tolerance", 0.01)
for case in spec["cases"]:
    r, _ = predict((GOLDEN / case["file"]).read_bytes(), case["captions"])
    name = f"golden {case['file']} ({len(case['captions'])} captions)"
    if case.get("expect_error"):
        check(name, r.status_code == 422 and code(r) == "NO_TEXT_INPUT", r.text[:200])
        continue
    body, exp = r.json(), case["expected"]
    ok = r.status_code == 200 and body["ocr_text"] == exp["ocr_text"] and len(body["results"]) == len(exp["results"])
    worst = 0.0
    for got, want in zip(body.get("results", []), exp["results"]):
        worst = max(worst, abs(got["match_probability"] - want["match_probability"]))
        ok &= got["verdict"] == want["verdict"] and got["text"] == want["text"] and got["mode"] == want["mode"]
    check(f"{name} max|dp|={worst:.6f}", ok and worst <= tol, r.text[:200])

# --- spec section 11.2 scenarios ---------------------------------------------------------
lat_one = []
r, ms = predict(PLAIN, ["A standing toilet hole filled with nasty filthy."]); lat_one.append(ms)
res = r.json()["results"]
check("1. plain photo + matching caption -> A MATCH", r.status_code == 200 and res[0]["mode"] == "A" and res[0]["verdict"] == "MATCH")
r, ms = predict(PLAIN, ["A herd of zebras grazing on a grassy savanna under a blue sky."]); lat_one.append(ms)
res = r.json()["results"]
check("2. plain photo + unrelated caption -> A MISMATCH", r.status_code == 200 and res[0]["mode"] == "A" and res[0]["verdict"] == "MISMATCH")
r, _ = predict(TEXT_IMG)
res = r.json()["results"]
check("3. text image, no captions -> one ocr row B-ocr", r.status_code == 200 and len(res) == 1 and res[0]["source"] == "ocr" and res[0]["mode"] == "B-ocr")
r, _ = predict(TEXT_IMG, ["The back of a silver car with a Virginia licence plate."])
modes = {x["source"]: x["mode"] for x in r.json()["results"]}
check("4. text image + caption -> user B-cap + ocr B-ocr", r.status_code == 200 and modes == {"user": "B-cap", "ocr": "B-ocr"})
r, _ = predict(PLAIN)
check("5. plain photo, no captions -> 422 NO_TEXT_INPUT", r.status_code == 422 and code(r) == "NO_TEXT_INPUT")
r, _ = predict(PIZZA, ["A large pizza covered in pepperoni.", "A man riding a surfboard.", "A cat on a laptop."])
res = r.json()["results"]
probs = [x["match_probability"] for x in res]
check("6. plain photo + 3 captions -> ranked by probability", r.status_code == 200 and [x["rank"] for x in res] == [1, 2, 3] and probs == sorted(probs, reverse=True))
r, _ = predict(PLAIN, ["  ", ""])
check("7. blank captions only -> 422 NO_TEXT_INPUT", r.status_code == 422 and code(r) == "NO_TEXT_INPUT")
r, _ = predict(PLAIN, [f"caption {i}" for i in range(6)])
check("8. 6 captions -> 400 TOO_MANY_CAPTIONS", r.status_code == 400 and code(r) == "TOO_MANY_CAPTIONS")
r, _ = predict(PLAIN, ["a" * 301])
check("9. 301-char caption -> 400 CAPTION_TOO_LONG", r.status_code == 400 and code(r) == "CAPTION_TOO_LONG")
r, _ = predict(b"\xff\xd8\xff" + b"0" * (11 * 1024 * 1024))
check("10. 11 MB file -> 413", r.status_code == 413)
r, _ = predict(b"just text\n" * 20, filename="notes.jpg")
check("11. .txt renamed .jpg -> 415 UNSUPPORTED_FILE_TYPE", r.status_code == 415 and code(r) == "UNSUPPORTED_FILE_TYPE")

# --- extras ------------------------------------------------------------------------------
buf = io.BytesIO()
Image.open(io.BytesIO(PLAIN)).save(buf, "PNG")
r, _ = predict(buf.getvalue(), ["a toilet"], filename="x.png", ctype="image/png")
check("PNG upload accepted", r.status_code == 200)
r, _ = predict(None, ["a dog"])
check("missing image -> 400 MISSING_IMAGE", r.status_code == 400 and code(r) == "MISSING_IMAGE")

# --- latency -----------------------------------------------------------------------------
N = 20
for _ in range(N - len(lat_one)):
    lat_one.append(predict(PLAIN, ["A standing toilet hole filled with nasty filthy."])[1])
lat_ocr = [predict(TEXT_IMG, [f"caption number {i} about a car" for i in range(5)])[1] for _ in range(N)]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


print(f"\nlatency  1 caption:          p50 {pct(lat_one, .5):5.0f} ms  p95 {pct(lat_one, .95):5.0f} ms  (n={N})")
print(f"latency  5 captions + OCR:   p50 {pct(lat_ocr, .5):5.0f} ms  p95 {pct(lat_ocr, .95):5.0f} ms  (n={N})")
check("latency: p50 1 caption <= 2 s", pct(lat_one, .5) <= 2000)
check("latency: p95 5 captions + OCR <= 4 s", pct(lat_ocr, .95) <= 4000)

print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
