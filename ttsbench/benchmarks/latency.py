"""Latency benchmark. Phase 3.

Produces one :class:`SynthesisRecord` per (text, repeat) pair. Batch adapters get
model-load / cold / warm / wall / duration / realtime-factor timing; streaming
adapters get the t0-t4 checkpoints plus ttfb/ttfa/ttfp and inter-chunk gap stats.
Cloud-style fields stay ``None`` for batch adapters rather than being fabricated.

Realtime factor here is ``wall_time / audio_duration``: < 1 means synthesis is
faster than realtime.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from ttsbench.adapters.base import AudioChunk, SynthesisResult, TTSAdapter
from ttsbench.schemas import SynthesisRecord
from ttsbench.utils import write_wav

_SAMPLE_WIDTH = 2  # 16-bit PCM
_CHANNELS = 1


@dataclass(frozen=True)
class BenchmarkItem:
    """One text to synthesize, with a stable id used for audio filenames."""

    id: str
    text: str
    dataset_item_id: str | None = None


def _audio_duration_ms(num_bytes: int, sample_rate: int) -> float | None:
    bytes_per_ms = sample_rate * _SAMPLE_WIDTH * _CHANNELS / 1000.0
    return (num_bytes / bytes_per_ms) if bytes_per_ms else None


def _realtime_factor(wall_ms: float | None, duration_ms: float | None) -> float | None:
    if not wall_ms or not duration_ms:
        return None
    return wall_ms / duration_ms


def benchmark_latency(
    adapter: TTSAdapter,
    items: Sequence[BenchmarkItem],
    repeats: int,
    run_id: str,
    *,
    suite: str = "latency",
    dataset: str | None = None,
    audio_dir: Path | None = None,
    first_playable_ms: float = 250.0,
) -> list[SynthesisRecord]:
    """Run the latency suite and return one record per (item, repeat)."""
    load_start = time.perf_counter_ns()
    adapter.load()
    model_load_ms = (time.perf_counter_ns() - load_start) / 1e6

    records: list[SynthesisRecord] = []
    first_record = True
    for item in items:
        for repeat in range(repeats):
            if adapter.streaming:
                record, audio = asyncio.run(
                    _streaming_record(
                        adapter, item, repeat, run_id, suite, dataset, first_playable_ms
                    )
                )
            else:
                record, audio = _batch_record(adapter, item, repeat, run_id, suite, dataset)

            if first_record:
                record.model_load_ms = model_load_ms
                first_record = False

            if audio_dir is not None:
                filename = f"{item.id}_r{repeat}.wav"
                write_wav(audio, record.sample_rate, audio_dir / filename)
                record.audio_path = f"audio/{filename}"

            records.append(record)
    return records


def _batch_record(
    adapter: TTSAdapter,
    item: BenchmarkItem,
    repeat: int,
    run_id: str,
    suite: str,
    dataset: str | None,
) -> tuple[SynthesisRecord, bytes]:
    start = time.perf_counter_ns()
    result = adapter.synthesize(item.text)
    end = time.perf_counter_ns()
    assert isinstance(result, SynthesisResult)

    wall_ms = (end - start) / 1e6
    duration_ms = result.duration_ms
    if duration_ms is None:
        duration_ms = _audio_duration_ms(len(result.audio), result.sample_rate)

    record = SynthesisRecord(
        run_id=run_id,
        provider=adapter.provider,
        execution_mode=adapter.execution_mode,
        model=adapter.model,
        voice=adapter.voice,
        runtime_backend=adapter.runtime_backend,
        device=adapter.actual_device or adapter.requested_device,
        text=item.text,
        dataset=dataset,
        dataset_item_id=item.dataset_item_id,
        benchmark_suite=suite,
        cold_start=(repeat == 0),
        audio_duration_ms=duration_ms,
        wall_time_ms=wall_ms,
        realtime_factor=_realtime_factor(wall_ms, duration_ms),
        sample_rate=result.sample_rate,
        format=result.audio_format,
        streaming=False,
        streaming_mode=adapter.streaming_mode,
        repeat_index=repeat,
        characters_sent=len(item.text),
    )
    return record, result.audio


async def _streaming_record(
    adapter: TTSAdapter,
    item: BenchmarkItem,
    repeat: int,
    run_id: str,
    suite: str,
    dataset: str | None,
    first_playable_ms: float,
) -> tuple[SynthesisRecord, bytes]:
    sample_rate = adapter.sample_rate
    playable_bytes = first_playable_ms * sample_rate * _SAMPLE_WIDTH * _CHANNELS / 1000.0

    t0 = time.perf_counter_ns()
    output = adapter.synthesize(item.text)
    assert isinstance(output, AsyncIterator)

    t1: int | None = None
    t2: int | None = None
    t3: int | None = None
    pcm = bytearray()
    recv_times: list[int] = []
    async for chunk in output:
        assert isinstance(chunk, AudioChunk)
        recv_times.append(chunk.recv_ns)
        if t1 is None:
            t1 = chunk.recv_ns
        if t2 is None and chunk.data:
            t2 = chunk.recv_ns
        pcm += chunk.data
        if t3 is None and len(pcm) >= playable_bytes:
            t3 = chunk.recv_ns
    t4 = time.perf_counter_ns()

    def ms(later: int | None) -> float | None:
        return None if later is None else (later - t0) / 1e6

    wall_ms = (t4 - t0) / 1e6
    duration_ms = _audio_duration_ms(len(pcm), sample_rate)
    gaps = [(b - a) / 1e6 for a, b in zip(recv_times, recv_times[1:], strict=False)]
    gap_stats: dict[str, float | int] = {"chunk_count": len(recv_times)}
    if gaps:
        gap_stats["interchunk_gap_ms_mean"] = sum(gaps) / len(gaps)
        gap_stats["interchunk_gap_ms_max"] = max(gaps)

    record = SynthesisRecord(
        run_id=run_id,
        provider=adapter.provider,
        execution_mode=adapter.execution_mode,
        model=adapter.model,
        voice=adapter.voice,
        runtime_backend=adapter.runtime_backend,
        device=adapter.actual_device or adapter.requested_device,
        text=item.text,
        dataset=dataset,
        dataset_item_id=item.dataset_item_id,
        benchmark_suite=suite,
        cold_start=(repeat == 0),
        t0_request_sent_ns=t0,
        t1_first_chunk_ns=t1,
        t2_first_audio_ns=t2,
        t3_first_playable_ns=t3,
        t4_synthesis_complete_ns=t4,
        audio_duration_ms=duration_ms,
        wall_time_ms=wall_ms,
        ttfb_ms=ms(t1),
        ttfa_ms=ms(t2),
        ttfp_ms=ms(t3),
        realtime_factor=_realtime_factor(wall_ms, duration_ms),
        sample_rate=sample_rate,
        format=adapter.audio_format,
        streaming=True,
        streaming_mode=adapter.streaming_mode,
        repeat_index=repeat,
        characters_sent=len(item.text),
        provider_params=gap_stats,
    )
    return record, bytes(pcm)
