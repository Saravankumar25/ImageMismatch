"""Capped, fully in-memory multipart/form-data reading.

Starlette's default form parser spools uploads larger than 1 MB to a temporary file on disk.
The spec requires that nothing is written to disk, so the body is streamed into memory with a
hard size cap and parsed here with python-multipart's low-level callback parser.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.requests import Request

from .errors import AppError


@dataclass
class Part:
    name: str
    filename: str | None
    content_type: str
    data: bytearray = field(default_factory=bytearray)


async def read_body_capped(request: Request, max_bytes: int) -> bytes:
    """Read the request body into memory, aborting as soon as it exceeds `max_bytes`."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise AppError("FILE_TOO_LARGE")
    buf = bytearray()
    async for chunk in request.stream():
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise AppError("FILE_TOO_LARGE")
    return bytes(buf)


def parse_multipart(body: bytes, content_type: str, max_fields: int) -> list[Part]:
    ctype, params = parse_options_header(content_type or "")
    boundary = params.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise AppError("BAD_REQUEST", "Expected a multipart/form-data request.")

    parts: list[Part] = []
    state: dict = {"field": bytearray(), "value": bytearray(), "headers": {}, "part": None}

    def on_part_begin():
        if len(parts) >= max_fields:
            raise AppError("BAD_REQUEST", "Too many form fields.")
        state["headers"] = {}
        state["part"] = None

    def on_header_field(data, start, end):
        state["field"].extend(data[start:end])

    def on_header_value(data, start, end):
        state["value"].extend(data[start:end])

    def on_header_end():
        state["headers"][bytes(state["field"]).lower()] = bytes(state["value"])
        state["field"].clear()
        state["value"].clear()

    def on_headers_finished():
        _, disp = parse_options_header(state["headers"].get(b"content-disposition", b""))
        name = disp.get(b"name", b"").decode("utf-8", "replace")
        filename = disp.get(b"filename")
        ptype = state["headers"].get(b"content-type", b"").decode("latin-1")
        state["part"] = Part(name, filename.decode("utf-8", "replace") if filename is not None else None, ptype)

    def on_part_data(data, start, end):
        state["part"].data.extend(data[start:end])

    def on_part_end():
        if state["part"] is not None:
            parts.append(state["part"])

    parser = MultipartParser(
        boundary,
        callbacks={
            "on_part_begin": on_part_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
        },
    )
    try:
        parser.write(body)
        parser.finalize()
    except AppError:
        raise
    except Exception as exc:  # malformed framing
        raise AppError("BAD_REQUEST", "Malformed multipart body.") from exc
    return parts
