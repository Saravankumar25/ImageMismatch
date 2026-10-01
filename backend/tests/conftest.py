import os
import time
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
os.environ.setdefault("MODEL_DIR", str(BACKEND / "model_outputs"))
MODEL_DIR = Path(os.environ["MODEL_DIR"])
GOLDEN = MODEL_DIR / "golden"


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        deadline = time.time() + 300
        while c.get("/api/health").status_code != 200:
            if time.time() > deadline:
                pytest.fail("model did not load within 300 s")
            time.sleep(0.5)
        yield c


def post_predict(client, image_bytes, captions=(), filename="image.jpg", content_type="image/jpeg"):
    files = [("image", (filename, image_bytes, content_type))] if image_bytes is not None else []
    data = {"captions": list(captions)} if captions else {}
    if not files:
        # httpx only sends multipart when files are present; add a dummy text part.
        files = [("unused", (None, b""))]
    return client.post("/api/predict", files=files, data=data)
