"""FastAPI app: routes and startup (spec section 7)."""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import settings
from .errors import AppError, install_error_handlers
from .multipart import parse_multipart, read_body_capped
from .schemas import ErrorResponse, HealthResponse, PredictResponse
from .service import ModelService, verify_model_dir

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


def _load_in_background(service: ModelService) -> None:
    try:
        service.load()
    except Exception:
        log.critical("Model failed to load; shutting down", exc_info=True)
        os._exit(1)  # fail fast: a backend without a model is useless


@asynccontextmanager
async def lifespan(app: FastAPI):
    verify_model_dir(settings.model_dir)  # raises -> uvicorn exits with a clear message
    log.info("MODEL_DIR=%s, loading models in the background", settings.model_dir)
    service = ModelService(settings)
    app.state.service = service
    threading.Thread(target=_load_in_background, args=(service,), name="model-loader", daemon=True).start()
    yield


app = FastAPI(title="Image–Text Match API", version="1.0.0", lifespan=lifespan,
              docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
install_error_handlers(app)

_ERRORS = {c: {"model": ErrorResponse} for c in (400, 413, 415, 422, 500, 503)}
_PREDICT_BODY = {
    "requestBody": {
        "required": True,
        "content": {"multipart/form-data": {"schema": {
            "type": "object",
            "required": ["image"],
            "properties": {
                "image": {"type": "string", "format": "binary", "description": "JPEG, PNG or WebP, max 10 MB"},
                "captions": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            },
        }}},
    }
}


@app.post("/api/predict", response_model=PredictResponse, responses=_ERRORS, openapi_extra=_PREDICT_BODY)
async def predict(request: Request):
    service: ModelService = request.app.state.service
    if not service.ready.is_set():
        raise AppError("MODEL_NOT_READY")
    body = await read_body_capped(request, settings.max_body_bytes)
    parts = parse_multipart(body, request.headers.get("content-type", ""), settings.max_form_fields)
    del body
    image = next((bytes(p.data) for p in parts if p.name == "image"), None)
    captions = [p.data.decode("utf-8", "replace") for p in parts if p.name == "captions"]
    del parts
    # Inference is CPU-bound; run it in the threadpool so the event loop stays responsive.
    return await run_in_threadpool(service.analyze, image, captions)


@app.get("/api/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health(request: Request):
    loaded = request.app.state.service.ready.is_set()
    body = {"status": "ok" if loaded else "loading", "model_loaded": loaded, "model_version": settings.model_version}
    return JSONResponse(body, status_code=200 if loaded else 503)


@app.get("/api/model-info")
def model_info(request: Request):
    return request.app.state.service.model_info
