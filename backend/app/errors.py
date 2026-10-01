"""Error codes (spec section 7.4) and the handlers that render them as {"error": {code, message}}."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("app.errors")

# code -> (HTTP status, user-facing message)
ERRORS: dict[str, tuple[int, str]] = {
    "MISSING_IMAGE": (400, "Please upload an image."),
    "TOO_MANY_CAPTIONS": (400, "You can add up to 5 captions."),
    "CAPTION_TOO_LONG": (400, "Each caption can be at most 300 characters."),
    "BAD_REQUEST": (400, "The request could not be understood."),
    "FILE_TOO_LARGE": (413, "Image is too large. Maximum is 10 MB."),
    "UNSUPPORTED_FILE_TYPE": (415, "Please upload a JPEG, PNG, or WebP image."),
    "NO_TEXT_INPUT": (
        422,
        "No text was provided and no text was found in the image. "
        "Add a caption or upload an image that contains text.",
    ),
    "INFERENCE_FAILED": (500, "Something went wrong. Please try again."),
    "MODEL_NOT_READY": (503, "The model is still loading. Please try again in a moment."),
}


class AppError(Exception):
    def __init__(self, code: str, message: str | None = None):
        status, default_message = ERRORS[code]
        super().__init__(message or default_message)
        self.code = code
        self.status = status
        self.message = message or default_message


def error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return JSONResponse(error_body("BAD_REQUEST", ERRORS["BAD_REQUEST"][1]), status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(error_body(code, str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(error_body("INFERENCE_FAILED", ERRORS["INFERENCE_FAILED"][1]), status_code=500)
