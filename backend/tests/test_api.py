"""API tests (spec section 11.2) plus a few extra edge cases."""
import io

import pytest
from PIL import Image

from conftest import GOLDEN, post_predict

PLAIN = (GOLDEN / "coco1_coco_220446.jpg").read_bytes()          # toilet photo, no text
PIZZA = (GOLDEN / "coco2_coco_451683.jpg").read_bytes()          # pizza photo, no text
TEXT_IMG = (GOLDEN / "textcaps_ocronly2_tc_2659571273d85bbd.jpg").read_bytes()  # licence plate

MATCHING = "A standing toilet hole filled with nasty filthy."
UNRELATED = "A herd of zebras grazing on a grassy savanna under a blue sky."


def err(resp):
    return resp.json()["error"]["code"]


# --- 1-12 from the spec ------------------------------------------------------------------
def test_01_plain_photo_matching_caption(client):
    r = post_predict(client, PLAIN, [MATCHING])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text_detected"] is False
    (res,) = body["results"]
    assert (res["mode"], res["verdict"], res["source"], res["rank"]) == ("A", "MATCH", "user", 1)
    assert len(res["heatmap"]) == 7 and all(len(row) == 7 for row in res["heatmap"])


def test_02_plain_photo_unrelated_caption(client):
    r = post_predict(client, PLAIN, [UNRELATED])
    assert r.status_code == 200, r.text
    (res,) = r.json()["results"]
    assert (res["mode"], res["verdict"]) == ("A", "MISMATCH")


def test_03_text_image_no_captions(client):
    r = post_predict(client, TEXT_IMG)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text_detected"] is True and body["ocr_text"]
    (res,) = body["results"]
    assert (res["source"], res["mode"], res["text"]) == ("ocr", "B-ocr", body["ocr_text"])


def test_04_text_image_with_caption(client):
    r = post_predict(client, TEXT_IMG, ["The back of a silver car with a Virginia licence plate."])
    assert r.status_code == 200, r.text
    rows = {res["source"]: res for res in r.json()["results"]}
    assert set(rows) == {"user", "ocr"}
    assert rows["user"]["mode"] == "B-cap" and rows["ocr"]["mode"] == "B-ocr"


def test_05_plain_photo_no_captions(client):
    r = post_predict(client, PLAIN)
    assert r.status_code == 422 and err(r) == "NO_TEXT_INPUT"
    assert "No text was provided" in r.json()["error"]["message"]


def test_06_plain_photo_three_captions_ranked(client):
    caps = ["A large pizza covered in pepperoni, olives and mushrooms.",
            "A man riding a surfboard on a large wave.",
            "A cat sleeping on a laptop keyboard."]
    r = post_predict(client, PIZZA, caps)
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) == 3
    assert [x["rank"] for x in results] == [1, 2, 3]
    probs = [x["match_probability"] for x in results]
    assert probs == sorted(probs, reverse=True)
    assert results[0]["text"] == caps[0] and results[0]["verdict"] == "MATCH"


def test_07_blank_captions_only(client):
    r = post_predict(client, PLAIN, ["   ", "", "\t"])
    assert r.status_code == 422 and err(r) == "NO_TEXT_INPUT"


def test_08_six_captions(client):
    r = post_predict(client, PLAIN, [f"caption number {i}" for i in range(6)])
    assert r.status_code == 400 and err(r) == "TOO_MANY_CAPTIONS"


def test_09_caption_too_long(client):
    r = post_predict(client, PLAIN, ["a" * 301])
    assert r.status_code == 400 and err(r) == "CAPTION_TOO_LONG"


def test_10_file_too_large(client):
    r = post_predict(client, b"\xff\xd8\xff" + b"0" * (11 * 1024 * 1024))
    assert r.status_code == 413 and err(r) == "FILE_TOO_LARGE"


def test_11_text_file_renamed_to_jpg(client):
    r = post_predict(client, b"this is just a text file\n" * 10, filename="notes.jpg")
    assert r.status_code == 415 and err(r) == "UNSUPPORTED_FILE_TYPE"


def test_12_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


# --- extra edge cases ----------------------------------------------------------------------
def test_missing_image(client):
    r = post_predict(client, None, [MATCHING])
    assert r.status_code == 400 and err(r) == "MISSING_IMAGE"


def test_exactly_300_chars_allowed(client):
    caption = "a " * 149 + "ab"
    assert len(caption) == 300
    r = post_predict(client, PLAIN, [caption])
    assert r.status_code == 200, r.text


def test_duplicates_removed_case_insensitive(client):
    r = post_predict(client, PLAIN, [MATCHING, MATCHING.upper(), "  " + MATCHING + "  "])
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["results"]) == 1
    assert any(w["code"] == "DUPLICATE_CAPTIONS_REMOVED" for w in body["warnings"])


def test_six_boxes_with_one_blank_is_ok(client):
    r = post_predict(client, PLAIN, ["", "a dog", "a cat", "a bus", "a toilet", "a pizza"])
    assert r.status_code == 200 and len(r.json()["results"]) == 5


def test_long_caption_truncation_warning(client):
    r = post_predict(client, PLAIN, [" ".join(["a"] * 100)])  # 199 chars, ~100 tokens > 77
    assert r.status_code == 200, r.text
    assert any(w["code"] == "CAPTION_TRUNCATED" for w in r.json()["warnings"])


def test_png_and_webp_accepted(client):
    img = Image.open(io.BytesIO(PLAIN))
    for fmt, ctype in (("PNG", "image/png"), ("WEBP", "image/webp")):
        buf = io.BytesIO()
        img.save(buf, fmt)
        r = post_predict(client, buf.getvalue(), [MATCHING], filename=f"x.{fmt.lower()}", content_type=ctype)
        assert r.status_code == 200, (fmt, r.text)


def test_gif_rejected(client):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, "GIF")
    r = post_predict(client, buf.getvalue(), ["a red square"], filename="x.gif", content_type="image/gif")
    assert r.status_code == 415 and err(r) == "UNSUPPORTED_FILE_TYPE"


def test_too_many_megapixels(client):
    buf = io.BytesIO()
    Image.new("L", (7000, 6000), 255).save(buf, "PNG")  # 42 MP, tiny file
    r = post_predict(client, buf.getvalue(), ["a white wall"], filename="big.png", content_type="image/png")
    assert r.status_code == 413 and err(r) == "FILE_TOO_LARGE"


def test_response_shape(client):
    body = post_predict(client, PLAIN, [MATCHING]).json()
    assert set(body) == {"request_id", "text_detected", "ocr_text", "results", "warnings",
                         "latency_ms", "model_version"}
    res = body["results"][0]
    assert set(res) == {"rank", "source", "text", "mode", "verdict", "match_probability", "similarity",
                        "threshold", "heatmap"}
    assert abs(sum(sum(row) for row in res["heatmap"]) - 1.0) < 1e-3


def test_model_info(client):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "head-v1"
    assert set(body["test_metrics"]) == {"A", "B-cap", "B-ocr"}
    assert body["limitations"]


@pytest.mark.parametrize("path", ["/api/nope", "/api/predict"])
def test_errors_use_standard_shape(client, path):
    r = client.get(path)
    assert r.status_code in (404, 405)
    assert "code" in r.json()["error"]
