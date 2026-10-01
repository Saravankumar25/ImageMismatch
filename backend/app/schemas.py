"""Pydantic response models (spec sections 7.2-7.4)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class WarningItem(BaseModel):
    code: str
    message: str


class ResultItem(BaseModel):
    rank: int
    source: Literal["user", "ocr"]
    text: str
    mode: Literal["A", "B-cap", "B-ocr"]
    verdict: Literal["MATCH", "MISMATCH"]
    match_probability: float
    similarity: float
    threshold: float
    heatmap: list[list[float]]


class PredictResponse(BaseModel):
    request_id: str
    text_detected: bool
    ocr_text: str
    results: list[ResultItem]
    warnings: list[WarningItem]
    latency_ms: int
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "loading"]
    model_loaded: bool
    model_version: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
