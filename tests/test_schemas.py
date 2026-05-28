"""Schema serialization and not_applicable timing behavior."""

import json

from ttsbench.schemas import (
    NOT_APPLICABLE,
    AudioFormat,
    ExecutionMode,
    RuntimeBackend,
    StreamingMode,
    SynthesisRecord,
    render_timing,
)


def _local_batch_record() -> SynthesisRecord:
    return SynthesisRecord(
        run_id="2026-05-28-piper-healthcare-cpu",
        provider="piper",
        execution_mode=ExecutionMode.LOCAL,
        model="piper",
        voice="en_US-amy",
        runtime_backend=RuntimeBackend.CPU,
        device="Apple M3 Pro CPU",
        text="Your appointment is on May sixth.",
        dataset="healthcare",
        dataset_item_id="healthcare_appt_001",
        benchmark_suite="latency",
        cold_start=False,
        t0_request_sent_ns=12345678900,
        t4_synthesis_complete_ns=12346498900,
        audio_duration_ms=3120.0,
        wall_time_ms=820.0,
        realtime_factor=3.8,
        sample_rate=22050,
        format=AudioFormat.PCM_S16LE,
        streaming=False,
        streaming_mode=StreamingMode.NONE,
        repeat_index=2,
        characters_sent=33,
    )


def test_local_batch_record_does_not_fabricate_streaming_timing():
    record = _local_batch_record()
    assert record.t1_first_chunk_ns is None
    assert record.t2_first_audio_ns is None
    assert record.t3_first_playable_ns is None
    assert record.ttfb_ms is None
    assert record.ttfa_ms is None
    assert record.ttfp_ms is None
    assert record.estimated_cost_usd == 0.0


def test_record_round_trips_through_json():
    record = _local_batch_record()
    payload = json.loads(record.model_dump_json())
    # Nullable timing fields serialize to JSON null, not 0.
    assert payload["ttfb_ms"] is None
    assert payload["streaming"] is False
    assert payload["format"] == "pcm_s16le"
    assert payload["execution_mode"] == "local"

    restored = SynthesisRecord.model_validate(payload)
    assert restored == record


def test_render_timing():
    assert render_timing(None) == NOT_APPLICABLE
    assert render_timing(0) == "0"
    assert render_timing(12.5) == "12.5"
