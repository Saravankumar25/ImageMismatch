"""Environment-driven settings and limits (spec section 9)."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_MODEL_FILES = ("head.pt", "mismatch_core.py", "calibration.json", "thresholds.json", "config.json")
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def available_cpus() -> int:
    """CPU cores available to this process, honouring cgroup (container) quotas when present."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # Windows / macOS
        n = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    model_dir: Path = field(default_factory=lambda: Path(os.environ.get("MODEL_DIR", "./model_outputs")).resolve())
    max_upload_mb: int = field(default_factory=lambda: _int_env("MAX_UPLOAD_MB", 10))
    max_captions: int = field(default_factory=lambda: _int_env("MAX_CAPTIONS", 5))
    max_caption_chars: int = field(default_factory=lambda: _int_env("MAX_CAPTION_CHARS", 300))
    max_megapixels: int = field(default_factory=lambda: _int_env("MAX_MEGAPIXELS", 40))
    # Capped at 4: EasyOCR gains nothing past 4 threads (~9.2 s at 4, 6 or 8 on an 18-core host), and
    # 18 OpenMP threads on a shared Docker VM stalled a single request for minutes.
    torch_threads: int = field(default_factory=lambda: _int_env("TORCH_THREADS", min(available_cpus(), 4)))
    model_version: str = field(default_factory=lambda: os.environ.get("MODEL_VERSION", "head-v1"))
    # Safety cap on the number of multipart fields, so a request cannot carry thousands of parts.
    max_form_fields: int = 32

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_body_bytes(self) -> int:
        # Image cap + room for captions and multipart framing.
        return self.max_upload_bytes + 64 * 1024

    @property
    def max_pixels(self) -> int:
        return self.max_megapixels * 1_000_000


settings = Settings()
