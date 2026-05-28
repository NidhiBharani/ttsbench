"""Enums and the per-synthesis result record (PRD section 9).

Timing fields that do not apply to a given execution mode are stored as ``None``
and rendered as ``not_applicable`` so "measured zero" stays distinct from "this
metric does not exist for this adapter".
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

NOT_APPLICABLE = "not_applicable"


class ExecutionMode(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class RuntimeBackend(str, Enum):
    CPU = "cpu"
    MPS = "mps"
    MLX = "mlx"
    COREML = "coreml"
    ONNX = "onnx"
    CUDA = "cuda"
    PROVIDER_HOSTED = "provider_hosted"
    UNKNOWN = "unknown"


class StreamingMode(str, Enum):
    NONE = "none"
    WEBSOCKET = "websocket"
    HTTP_CHUNKED = "http_chunked"
    LOCAL_INCREMENTAL = "local_incremental"


class AudioFormat(str, Enum):
    PCM_S16LE = "pcm_s16le"
    WAV = "wav"
    MP3 = "mp3"
    OPUS = "opus"


def render_timing(value: float | int | None) -> str:
    """Display a timing/metric field: its value, or ``not_applicable`` when ``None``."""
    return NOT_APPLICABLE if value is None else str(value)


class SynthesisRecord(BaseModel):
    """One synthesis result, serialized as a single JSONL record."""

    run_id: str
    provider: str
    execution_mode: ExecutionMode
    model: str
    voice: str
    runtime_backend: RuntimeBackend
    device: str | None = None

    text: str
    dataset: str | None = None
    dataset_item_id: str | None = None
    benchmark_suite: str

    model_load_ms: float | None = None
    cold_start: bool

    # Timing checkpoints in time.perf_counter_ns(). Cloud-style points are None
    # for local batch adapters rather than fabricated.
    t0_request_sent_ns: int | None = None
    t1_first_chunk_ns: int | None = None
    t2_first_audio_ns: int | None = None
    t3_first_playable_ns: int | None = None
    t4_synthesis_complete_ns: int | None = None

    audio_duration_ms: float | None = None
    wall_time_ms: float | None = None
    ttfb_ms: float | None = None
    ttfa_ms: float | None = None
    ttfp_ms: float | None = None
    realtime_factor: float | None = None

    sample_rate: int
    format: AudioFormat
    streaming: bool
    streaming_mode: StreamingMode

    region: str = "local"
    repeat_index: int
    audio_path: str | None = None
    characters_sent: int
    estimated_cost_usd: float = 0.0
    provider_params: dict[str, Any] = Field(default_factory=dict)
