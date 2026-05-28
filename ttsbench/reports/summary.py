"""Plain-text run summary with p50/p95/p99 aggregation. Phase 3."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from ttsbench.schemas import PronunciationResult, Severity, SynthesisRecord

# Below this many warm samples, tail percentiles (p95/p99) are not trustworthy.
_LOW_SAMPLE_THRESHOLD = 20


def percentile(values: Sequence[float], pct: float) -> float | None:
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


def mean(values: Sequence[float]) -> float | None:
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
    lines.append(f"  Cold synthesis wall (mean): {_fmt(mean(cold_walls), ' ms')}")
    lines.append(f"  Warm synthesis wall p50:    {_fmt(percentile(warm_walls, 50), ' ms')}")
    lines.append(f"  Warm synthesis wall p95:    {_fmt(percentile(warm_walls, 95), ' ms')}")
    lines.append(f"  Warm synthesis wall p99:    {_fmt(percentile(warm_walls, 99), ' ms')}")

    rtfs = [r.realtime_factor for r in records if r.realtime_factor is not None]
    durations = [r.audio_duration_ms for r in records if r.audio_duration_ms is not None]
    lines.append(f"  Realtime factor p50:        {_fmt(percentile(rtfs, 50))}")
    lines.append(f"  Audio duration (mean):      {_fmt(mean(durations), ' ms')}")

    if streaming:
        ttfb = [r.ttfb_ms for r in records if r.ttfb_ms is not None]
        ttfa = [r.ttfa_ms for r in records if r.ttfa_ms is not None]
        ttfp = [r.ttfp_ms for r in records if r.ttfp_ms is not None]
        lines += [
            "",
            f"  TTFB p50:  {_fmt(percentile(ttfb, 50), ' ms')}",
            f"  TTFA p50:  {_fmt(percentile(ttfa, 50), ' ms')}",
            f"  TTFP p50:  {_fmt(percentile(ttfp, 50), ' ms')}",
        ]

    if 0 < len(warm_walls) < _LOW_SAMPLE_THRESHOLD:
        lines += [
            "",
            f"  WARNING: only {len(warm_walls)} warm sample(s); p95/p99 are indicative,"
            " not reliable. Increase --repeats for stable tail percentiles.",
        ]

    return "\n".join(lines) + "\n"


def build_pronunciation_summary(results: Sequence[PronunciationResult]) -> str:
    if not results:
        return ""

    scored = [r for r in results if not r.advisory]
    passed = sum(1 for r in scored if r.passed)
    pass_rate = (passed / len(scored)) if scored else None

    high = [r for r in scored if r.severity is Severity.HIGH]
    high_passed = sum(1 for r in high if r.passed)
    high_rate = (high_passed / len(high)) if high else None

    lines = [
        "",
        "Pronunciation summary",
        f"  Scored results:   {len(scored)} (advisory excluded: {len(results) - len(scored)})",
        f"  Pass rate:        {_fmt_rate(pass_rate)}",
        f"  High-severity:    {_fmt_rate(high_rate)} ({high_passed}/{len(high)})",
    ]

    # Per-category pass rate.
    categories: dict[str, list[PronunciationResult]] = {}
    for result in scored:
        categories.setdefault(result.category, []).append(result)
    lines.append("  By category:")
    for category in sorted(categories):
        group = categories[category]
        rate = sum(1 for r in group if r.passed) / len(group)
        lines.append(f"    {category}: {_fmt_rate(rate)} ({len(group)} items)")

    high_failures = [r for r in high if not r.passed]
    if high_failures:
        lines.append(f"  HIGH-SEVERITY FAILURES ({len(high_failures)}):")
        for result in high_failures:
            missed = [
                f"{m.pattern!r} (PER {m.per:.2f})" if m.per is not None else repr(m.pattern)
                for m in result.pattern_matches
                if not m.matched
            ]
            lines.append(f"    [{result.item_id} r{result.repeat_index}] {result.input!r}")
            lines.append(f"        produced phonemes: {result.produced_phonemes}")
            if result.transcript is not None:
                lines.append(f"        asr transcript (advisory): {result.transcript!r}")
            lines.append(f"        missing: {missed}")
    else:
        lines.append("  No high-severity failures.")

    return "\n".join(lines) + "\n"


def _fmt_rate(rate: float | None) -> str:
    return "not_applicable" if rate is None else f"{rate * 100:.1f}%"


def write_summary(
    records: Sequence[SynthesisRecord],
    path: str | Path,
    pronunciation: Sequence[PronunciationResult] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build_summary(records)
    if pronunciation:
        text += build_pronunciation_summary(pronunciation)
    out.write_text(text)
    return out
