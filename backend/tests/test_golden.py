"""Golden tests (spec section 11.1): the API must reproduce the notebook's predict() outputs."""
import json

import pytest

from conftest import GOLDEN, post_predict

SPEC = json.loads((GOLDEN / "golden_predictions.json").read_text())
TOL = float(SPEC.get("tolerance", 0.01))
CASES = SPEC["cases"]


@pytest.mark.parametrize("case", CASES, ids=[f"{c['file']}-{len(c['captions'])}cap" for c in CASES])
def test_golden(client, case):
    r = post_predict(client, (GOLDEN / case["file"]).read_bytes(), case["captions"])
    if case.get("expect_error"):
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "NO_TEXT_INPUT"
        return

    assert r.status_code == 200, r.text
    body, exp = r.json(), case["expected"]
    assert body["text_detected"] == exp["text_detected"]
    assert body["ocr_text"] == exp["ocr_text"]
    assert len(body["results"]) == len(exp["results"])
    for got, want in zip(body["results"], exp["results"]):  # same ranking order
        assert (got["source"], got["text"], got["mode"]) == (want["source"], want["text"], want["mode"])
        assert got["verdict"] == want["verdict"]
        assert got["match_probability"] == pytest.approx(want["match_probability"], abs=TOL)
        assert got["similarity"] == pytest.approx(want["similarity"], abs=TOL)
        assert got["threshold"] == pytest.approx(want["threshold"], abs=1e-6)
