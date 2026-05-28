"""CSV report: one row per synthesis. Phase 3.

Timing/metric columns that do not apply to an adapter (e.g. ttfb for a local
batch model) are written as ``not_applicable`` so a reader can tell "unsupported"
apart from a real measured value.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from enum import Enum
from pathlib import Path

from ttsbench.schemas import NOT_APPLICABLE, SynthesisRecord

# Columns where ``None`` means "this metric does not exist for this adapter".
_METRIC_COLUMNS = frozenset(
    {
        "model_load_ms",
        "t0_request_sent_ns",
        "t1_first_chunk_ns",
        "t2_first_audio_ns",
        "t3_first_playable_ns",
        "t4_synthesis_complete_ns",
        "audio_duration_ms",
        "wall_time_ms",
        "ttfb_ms",
        "ttfa_ms",
        "ttfp_ms",
        "realtime_factor",
    }
)

_COLUMNS: tuple[str, ...] = (
    "run_id",
    "provider",
    "execution_mode",
    "model",
    "voice",
    "runtime_backend",
    "device",
    "benchmark_suite",
    "dataset",
    "dataset_item_id",
    "text",
    "repeat_index",
    "cold_start",
    "model_load_ms",
    "t0_request_sent_ns",
    "t1_first_chunk_ns",
    "t2_first_audio_ns",
    "t3_first_playable_ns",
    "t4_synthesis_complete_ns",
    "audio_duration_ms",
    "wall_time_ms",
    "ttfb_ms",
    "ttfa_ms",
    "ttfp_ms",
    "realtime_factor",
    "sample_rate",
    "format",
    "streaming",
    "streaming_mode",
    "region",
    "characters_sent",
    "estimated_cost_usd",
    "audio_path",
    "provider_params",
)


def _cell(column: str, value: object) -> str:
    if isinstance(value, dict):
        return json.dumps(value) if value else ""
    if value is None:
        return NOT_APPLICABLE if column in _METRIC_COLUMNS else ""
    if isinstance(value, Enum):
        return value.value
    return str(value)


def write_csv(records: Sequence[SynthesisRecord], path: str | Path) -> Path:
    """Write ``metrics.csv`` with one row per synthesis record."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_COLUMNS)
        for record in records:
            data = record.model_dump()
            writer.writerow([_cell(col, data.get(col)) for col in _COLUMNS])
    return out
