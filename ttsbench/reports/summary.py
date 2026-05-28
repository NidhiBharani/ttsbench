"""Plain-text run summary with p50/p95/p99 aggregation. Phase 3."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from ttsbench.schemas import SynthesisRecord

# Below this many warm samples, tail percentiles (p95/p99) are not trustworthy.
_LOW_SAMPLE_THRESHOLD = 20


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _fmt(value: float | None, unit: str = "") -> str:
    return "not_applicable" if value is None else f"{value:.2f}{unit}"


def build_summary(records: Sequence[SynthesisRecord]) -> str:
    if not records:
        return "No synthesis records.\n"

    head = records[0]
    streaming = any(r.streaming for r in records)
    cold = [r for r in records if r.cold_start]
    warm = [r for r in records if not r.cold_start]
    warm_walls = [r.wall_time_ms for r in warm if r.wall_time_ms is not None]
    model_load = next((r.model_load_ms for r in records if r.model_load_ms is not None), None)

    lines = [
        "TTSBench latency summary",
        f"  Provider:        {head.provider}",
        f"  Model:           {head.model}",
        f"  Voice:           {head.voice}",
        f"  Execution mode:  {head.execution_mode.value}",
        f"  Runtime backend: {head.runtime_backend.value}",
        f"  Device:          {head.device or 'not_applicable'}",
        f"  Streaming:       {streaming}",
        "",
        f"  Total syntheses: {len(records)}",
        f"  Cold runs:       {len(cold)}",
        f"  Warm runs:       {len(warm)}",
        f"  Model load:      {_fmt(model_load, ' ms')}",
        "",
    ]

    cold_walls = [r.wall_time_ms for r in cold if r.wall_time_ms is not None]
    lines.append(f"  Cold synthesis wall (mean): {_fmt(_mean(cold_walls), ' ms')}")
    lines.append(f"  Warm synthesis wall p50:    {_fmt(_percentile(warm_walls, 50), ' ms')}")
    lines.append(f"  Warm synthesis wall p95:    {_fmt(_percentile(warm_walls, 95), ' ms')}")
    lines.append(f"  Warm synthesis wall p99:    {_fmt(_percentile(warm_walls, 99), ' ms')}")

    rtfs = [r.realtime_factor for r in records if r.realtime_factor is not None]
    durations = [r.audio_duration_ms for r in records if r.audio_duration_ms is not None]
    lines.append(f"  Realtime factor p50:        {_fmt(_percentile(rtfs, 50))}")
    lines.append(f"  Audio duration (mean):      {_fmt(_mean(durations), ' ms')}")

    if streaming:
        ttfb = [r.ttfb_ms for r in records if r.ttfb_ms is not None]
        ttfa = [r.ttfa_ms for r in records if r.ttfa_ms is not None]
        ttfp = [r.ttfp_ms for r in records if r.ttfp_ms is not None]
        lines += [
            "",
            f"  TTFB p50:  {_fmt(_percentile(ttfb, 50), ' ms')}",
            f"  TTFA p50:  {_fmt(_percentile(ttfa, 50), ' ms')}",
            f"  TTFP p50:  {_fmt(_percentile(ttfp, 50), ' ms')}",
        ]

    if 0 < len(warm_walls) < _LOW_SAMPLE_THRESHOLD:
        lines += [
            "",
            f"  WARNING: only {len(warm_walls)} warm sample(s); p95/p99 are indicative,"
            " not reliable. Increase --repeats for stable tail percentiles.",
        ]

    return "\n".join(lines) + "\n"


def write_summary(records: Sequence[SynthesisRecord], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_summary(records))
    return out
