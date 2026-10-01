"""Build-order step 2: load Predictor directly and compare against golden_predictions.json.

Usage (from backend/):  python scripts/golden_check.py [MODEL_DIR]
Exits non-zero if any verdict differs or any probability is off by more than the tolerance.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

from PIL import Image

model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "model_outputs").resolve()
spec = importlib.util.spec_from_file_location("mismatch_core", model_dir / "mismatch_core.py")
core = importlib.util.module_from_spec(spec)
sys.modules["mismatch_core"] = core
spec.loader.exec_module(core)

golden = json.loads((model_dir / "golden" / "golden_predictions.json").read_text())
tol = float(golden.get("tolerance", 0.01))

t0 = time.perf_counter()
predictor = core.Predictor(artifacts_dir=str(model_dir), device="cpu")
print(f"Predictor loaded in {time.perf_counter() - t0:.1f}s")

failures = 0
for case in golden["cases"]:
    img = Image.open(model_dir / "golden" / case["file"])
    label = f"{case['file']} captions={len(case['captions'])}"
    t0 = time.perf_counter()
    if case.get("expect_error"):
        try:
            predictor.predict(img, case["captions"])
            print(f"FAIL {label}: expected {case['expect_error']}")
            failures += 1
        except core.NoTextInputError:
            print(f"ok   {label}: NoTextInputError")
        continue
    out = predictor.predict(img, case["captions"])
    exp = case["expected"]
    problems = []
    if out["text_detected"] != exp["text_detected"]:
        problems.append(f"text_detected {out['text_detected']} != {exp['text_detected']}")
    if out["ocr_text"] != exp["ocr_text"]:
        problems.append(f"ocr_text {out['ocr_text']!r} != {exp['ocr_text']!r}")
    got = {(r["source"], r["text"]): r for r in out["results"]}
    for e in exp["results"]:
        r = got.get((e["source"], e["text"]))
        if r is None:
            problems.append(f"missing result {e['source']}:{e['text'][:40]!r}")
            continue
        d = abs(r["match_probability"] - e["match_probability"])
        flag = "" if d <= tol and r["verdict"] == e["verdict"] else "  <-- MISMATCH"
        if flag:
            problems.append(f"{e['text'][:40]!r}")
        print(f"     {e['mode']:5s} p={r['match_probability']:.6f} exp={e['match_probability']:.6f} "
              f"d={d:.6f} {r['verdict']}/{e['verdict']}{flag}")
    status = "ok  " if not problems else "FAIL"
    failures += bool(problems)
    print(f"{status} {label} ({(time.perf_counter() - t0) * 1000:.0f} ms) {'; '.join(problems)}")

print("ALL GOLDEN CASES MATCH" if not failures else f"{failures} golden case(s) FAILED")
sys.exit(1 if failures else 0)
