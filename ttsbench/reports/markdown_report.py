"""Self-contained Markdown report per run. Phase 6.

Regenerated from run artifacts (``metadata.jsonl`` + ``pronunciation_results.jsonl``)
without re-running synthesis. Audio links are relative so the run directory stays
portable when moved or zipped. Local batch metrics and streaming metrics are kept
separate; unsupported fields render as ``not_applicable`` rather than fake zeros.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from ttsbench.reports.summary import mean, percentile
from ttsbench.schemas import PronunciationResult, Severity, SynthesisRecord


def load_records(run_dir: Path) -> list[SynthesisRecord]:
    path = run_dir / "metadata.jsonl"
    if not path.exists():
        return []
    return [
        SynthesisRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_pronunciation(run_dir: Path) -> list[PronunciationResult]:
    path = run_dir / "pronunciation_results.jsonl"
    if not path.exists():
        return []
    return [
        PronunciationResult.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _num(value: float | None, unit: str = "") -> str:
    return "not_applicable" if value is None else f"{value:.2f}{unit}"


def _rate(value: float | None) -> str:
    return "not_applicable" if value is None else f"{value * 100:.1f}%"


def _metadata_section(records: Sequence[SynthesisRecord]) -> list[str]:
    head = records[0]
    suites = sorted({r.benchmark_suite for r in records})
    repeats = max((r.repeat_index for r in records), default=-1) + 1
    rows = [
        ("Run ID", head.run_id),
        ("Provider", head.provider),
        ("Model", head.model),
        ("Voice", head.voice),
        ("Execution mode", head.execution_mode.value),
        ("Runtime backend", head.runtime_backend.value),
        ("Device", head.device or "not_applicable"),
        ("Dataset", head.dataset or "ad hoc (--text)"),
        ("Suites", ", ".join(suites)),
        ("Repeats", str(repeats)),
        ("Total syntheses", str(len(records))),
    ]
    lines = ["## Run metadata", "", "| Field | Value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return lines


def _config_section(records: Sequence[SynthesisRecord]) -> list[str]:
    head = records[0]
    rows = [
        ("Sample rate", str(head.sample_rate)),
        ("Audio format", head.format.value),
        ("Streaming", str(head.streaming)),
        ("Streaming mode", head.streaming_mode.value),
        ("Region", head.region),
    ]
    for key, value in sorted(head.provider_params.items()):
        rows.append((f"param: {key}", str(value)))
    lines = ["## Configuration", "", "| Param | Value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return lines


def _latency_section(records: Sequence[SynthesisRecord]) -> list[str]:
    cold = [r for r in records if r.cold_start]
    warm = [r for r in records if not r.cold_start]
    cold_walls = [r.wall_time_ms for r in cold if r.wall_time_ms is not None]
    warm_walls = [r.wall_time_ms for r in warm if r.wall_time_ms is not None]
    rtfs = [r.realtime_factor for r in records if r.realtime_factor is not None]
    durations = [r.audio_duration_ms for r in records if r.audio_duration_ms is not None]
    model_load = next((r.model_load_ms for r in records if r.model_load_ms is not None), None)

    rows = [
        ("Model load", _num(model_load, " ms")),
        ("Cold synthesis wall (mean)", _num(mean(cold_walls), " ms")),
        ("Warm synthesis wall p50", _num(percentile(warm_walls, 50), " ms")),
        ("Warm synthesis wall p95", _num(percentile(warm_walls, 95), " ms")),
        ("Warm synthesis wall p99", _num(percentile(warm_walls, 99), " ms")),
        ("Realtime factor p50", _num(percentile(rtfs, 50))),
        ("Audio duration (mean)", _num(mean(durations), " ms")),
    ]
    lines = [
        "## Latency (local batch)",
        "",
        "Realtime factor is wall time / audio duration; < 1 is faster than realtime.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {v} |" for k, v in rows]
    if 0 < len(warm_walls) < 20:
        lines += ["", f"> Note: only {len(warm_walls)} warm sample(s); p95/p99 are indicative."]
    return lines


def _streaming_section(records: Sequence[SynthesisRecord]) -> list[str]:
    streaming = [r for r in records if r.streaming]
    if not streaming:
        return []
    ttfb = [r.ttfb_ms for r in streaming if r.ttfb_ms is not None]
    ttfa = [r.ttfa_ms for r in streaming if r.ttfa_ms is not None]
    ttfp = [r.ttfp_ms for r in streaming if r.ttfp_ms is not None]
    walls = [r.wall_time_ms for r in streaming if r.wall_time_ms is not None]
    rows = [
        ("TTFB p50", _num(percentile(ttfb, 50), " ms")),
        ("TTFA p50", _num(percentile(ttfa, 50), " ms")),
        ("TTFP p50", _num(percentile(ttfp, 50), " ms")),
        ("Total wall p50", _num(percentile(walls, 50), " ms")),
    ]
    lines = ["## Streaming latency", "", "| Metric | Value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return lines


def _pronunciation_section(results: Sequence[PronunciationResult]) -> list[str]:
    if not results:
        return []
    scored = [r for r in results if not r.advisory]
    passed = sum(1 for r in scored if r.passed)
    pass_rate = (passed / len(scored)) if scored else None
    high = [r for r in scored if r.severity is Severity.HIGH]
    high_passed = sum(1 for r in high if r.passed)
    high_rate = (high_passed / len(high)) if high else None
    advisory_count = len(results) - len(scored)

    lines = [
        "## Pronunciation",
        "",
        "Pass/fail is phoneme-distance based. ASR transcripts are advisory only; "
        "high-severity items are flagged for human listening.",
        "",
        f"- Scored results: **{len(scored)}** (advisory excluded: {advisory_count})",
        f"- Pass rate: **{_rate(pass_rate)}**",
        f"- High-severity pass rate: **{_rate(high_rate)}** ({high_passed}/{len(high)})",
        "",
        "### Pass rate by category",
        "",
        "| Category | Pass rate | Items |",
        "| --- | --- | --- |",
    ]
    categories: dict[str, list[PronunciationResult]] = {}
    for result in scored:
        categories.setdefault(result.category, []).append(result)
    for category in sorted(categories):
        group = categories[category]
        rate = sum(1 for r in group if r.passed) / len(group)
        lines.append(f"| {category} | {_rate(rate)} | {len(group)} |")

    high_failures = [r for r in high if not r.passed]
    lines += ["", "### High-severity failures", ""]
    if not high_failures:
        lines.append("None.")
        return lines
    for result in high_failures:
        missed = ", ".join(
            f"`{m.pattern}` (PER {m.per:.2f})" if m.per is not None else f"`{m.pattern}`"
            for m in result.pattern_matches
            if not m.matched
        )
        link = f"[audio]({result.audio_path})" if result.audio_path else "n/a"
        lines += [
            f"#### `{result.item_id}` (r{result.repeat_index}) — {link}",
            "",
            f"- Input: {result.input!r}",
            f"- Missing forms: {missed}",
        ]
        if result.produced_phonemes:
            lines.append(f"- Produced phonemes: `{result.produced_phonemes}`")
        if result.transcript is not None:
            lines.append(f"- ASR transcript (advisory): {result.transcript!r}")
        lines.append("")
    return lines


def _cost_section(records: Sequence[SynthesisRecord]) -> list[str]:
    head = records[0]
    total_cost = sum(r.estimated_cost_usd for r in records)
    total_chars = sum(r.characters_sent for r in records)
    note = "Local inference: API cost is $0.00." if head.execution_mode.value == "local" else ""
    lines = [
        "## Cost & runtime",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Execution mode | {head.execution_mode.value} |",
        f"| Characters sent | {total_chars} |",
        f"| Estimated cost (USD) | ${total_cost:.4f} |",
    ]
    if note:
        lines += ["", note]
    return lines


def build_markdown(
    records: Sequence[SynthesisRecord],
    pronunciation: Sequence[PronunciationResult] = (),
    *,
    title: str | None = None,
) -> str:
    if not records:
        return "# TTSBench Report\n\nNo synthesis records found.\n"

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    blocks: list[list[str]] = [
        [f"# TTSBench Report — {title or records[0].run_id}", "", f"_Generated {generated}_"],
        _metadata_section(records),
        _config_section(records),
        _latency_section(records),
        _streaming_section(records),
        _pronunciation_section(pronunciation),
        _cost_section(records),
    ]
    return "\n\n".join("\n".join(block) for block in blocks if block) + "\n"


def write_markdown_report(run_dir: str | Path) -> Path:
    """Build report.md from the artifacts already written in ``run_dir``."""
    directory = Path(run_dir)
    records = load_records(directory)
    pronunciation = load_pronunciation(directory)
    out = directory / "report.md"
    out.write_text(build_markdown(records, pronunciation, title=directory.name))
    return out
