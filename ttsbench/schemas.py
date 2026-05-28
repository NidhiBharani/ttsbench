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


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DatasetItem(BaseModel):
    """One dataset line: input text plus the spoken forms it must contain."""

    id: str
    input: str
    category: str
    severity: Severity
    # All patterns are required (every one must match) for the item to pass.
    expected_spoken_contains: list[str] = Field(default_factory=list)
    # Code-switched items pass/fail is advisory only, not a hard signal.
    advisory: bool = False


class Dataset(BaseModel):
    name: str
    items: list[DatasetItem]


class PatternMatch(BaseModel):
    pattern: str
    matched: bool
    # Phoneme error rate of this pattern against the produced phonemes (None if
    # phoneme evaluation did not run). Lower is better; matched = per <= tolerance.
    per: float | None = None


class PronunciationResult(BaseModel):
    """Pronunciation result for one (item, repeat).

    Pass/fail is driven by phoneme distance (sound-based, robust to digit/word
    spelling). The ASR word transcript is kept as advisory human-readable context.
    """

    run_id: str
    dataset: str
    item_id: str
    repeat_index: int
    input: str
    category: str
    severity: Severity
    advisory: bool
    audio_path: str | None
    pattern_matches: list[PatternMatch]
    passed: bool
    produced_phonemes: str | None = None  # space-joined recognized phonemes
    # Advisory ASR output (does not affect pass/fail); raw kept for high severity.
    transcript: str | None = None
    raw_transcript: str | None = None


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
