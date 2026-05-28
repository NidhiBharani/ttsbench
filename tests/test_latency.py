"""Phase 3: latency benchmark for batch and streaming adapters."""

import csv
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from ttsbench.adapters.base import AudioChunk, SynthesisResult, TTSAdapter
from ttsbench.benchmarks.latency import BenchmarkItem, benchmark_latency
from ttsbench.reports.csv_report import write_csv
from ttsbench.reports.summary import build_summary
from ttsbench.schemas import (
    NOT_APPLICABLE,
    AudioFormat,
    ExecutionMode,
    RuntimeBackend,
    StreamingMode,
)


class FakeBatchAdapter(TTSAdapter):
    provider = "fake_batch"
    execution_mode = ExecutionMode.LOCAL
    model = "fake"
    voice = "default"
    runtime_backend = RuntimeBackend.CPU
    requested_device = "cpu"
    actual_device = "cpu"
    sample_rate = 22050
    audio_format = AudioFormat.PCM_S16LE
    streaming = False
    streaming_mode = StreamingMode.NONE

    def __init__(self) -> None:
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def synthesize(self, text: str, voice: str | None = None, **params: Any) -> SynthesisResult:
        return SynthesisResult(
            audio=b"\x00\x01" * len(text),
            sample_rate=self.sample_rate,
            audio_format=self.audio_format,
            duration_ms=float(len(text)),
        )


class FakeStreamingAdapter(TTSAdapter):
    provider = "fake_stream"
    execution_mode = ExecutionMode.CLOUD
    model = "fake"
    voice = "default"
    runtime_backend = RuntimeBackend.PROVIDER_HOSTED
    requested_device = None
    actual_device = None
    sample_rate = 24000
    audio_format = AudioFormat.PCM_S16LE
    streaming = True
    streaming_mode = StreamingMode.WEBSOCKET

    def synthesize(
        self, text: str, voice: str | None = None, **params: Any
    ) -> AsyncIterator[AudioChunk]:
        return self._stream(text)

    async def _stream(self, text: str) -> AsyncIterator[AudioChunk]:
        for index, ch in enumerate(text):
            yield AudioChunk(data=ch.encode() * 600, recv_ns=time.perf_counter_ns(), index=index)


def test_batch_benchmark_produces_record_per_repeat():
    adapter = FakeBatchAdapter()
    items = [BenchmarkItem(id="item000", text="hello")]
    records = benchmark_latency(adapter, items, repeats=3, run_id="r1")

    assert len(records) == 3
    assert adapter.load_calls == 1  # model loaded once, not per synthesis
    assert [r.repeat_index for r in records] == [0, 1, 2]
    assert [r.cold_start for r in records] == [True, False, False]
    # Model-load time recorded only on the first record.
    assert records[0].model_load_ms is not None
    assert all(r.model_load_ms is None for r in records[1:])
    # Batch adapters never fabricate streaming timing.
    assert all(r.ttfb_ms is None and r.t1_first_chunk_ns is None for r in records)
    assert all(r.wall_time_ms is not None for r in records)
    assert all(r.streaming is False for r in records)


def test_streaming_benchmark_computes_checkpoints():
    adapter = FakeStreamingAdapter()
    items = [BenchmarkItem(id="item000", text="abcdef")]
    records = benchmark_latency(adapter, items, repeats=1, run_id="r1", first_playable_ms=10.0)

    (record,) = records
    assert record.streaming is True
    assert record.t0_request_sent_ns is not None
    assert record.t1_first_chunk_ns is not None
    assert record.ttfb_ms is not None
    assert record.ttfa_ms is not None
    assert record.ttfp_ms is not None
    assert record.t4_synthesis_complete_ns >= record.t1_first_chunk_ns
    assert record.provider_params["chunk_count"] == 6


def test_audio_files_written(tmp_path):
    adapter = FakeBatchAdapter()
    items = [BenchmarkItem(id="item000", text="hi")]
    records = benchmark_latency(
        adapter, items, repeats=2, run_id="r1", audio_dir=tmp_path / "audio"
    )
    assert (tmp_path / "audio" / "item000_r0.wav").exists()
    assert (tmp_path / "audio" / "item000_r1.wav").exists()
    assert records[0].audio_path == "audio/item000_r0.wav"


def test_csv_marks_unsupported_metrics(tmp_path):
    adapter = FakeBatchAdapter()
    records = benchmark_latency(adapter, [BenchmarkItem(id="i", text="x")], 1, "r1")
    csv_path = write_csv(records, tmp_path / "metrics.csv")

    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["ttfb_ms"] == NOT_APPLICABLE
    assert row["t1_first_chunk_ns"] == NOT_APPLICABLE
    assert row["wall_time_ms"] != NOT_APPLICABLE
    assert row["provider"] == "fake_batch"


def test_summary_warns_on_low_samples():
    adapter = FakeBatchAdapter()
    records = benchmark_latency(adapter, [BenchmarkItem(id="i", text="x")], 3, "r1")
    summary = build_summary(records)
    assert "Warm synthesis wall p50" in summary
    assert "WARNING" in summary  # only 2 warm samples


def test_metadata_jsonl_roundtrips():
    adapter = FakeBatchAdapter()
    records = benchmark_latency(adapter, [BenchmarkItem(id="i", text="abc")], 1, "r1")
    line = records[0].model_dump_json()
    data = json.loads(line)
    assert data["provider"] == "fake_batch"
    assert data["execution_mode"] == "local"
    assert data["ttfb_ms"] is None
