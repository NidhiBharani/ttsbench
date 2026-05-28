"""Phase 6: Markdown report builder and the report CLI command."""

from typer.testing import CliRunner

from ttsbench.cli import app
from ttsbench.reports.markdown_report import build_markdown, write_markdown_report
from ttsbench.schemas import (
    AudioFormat,
    ExecutionMode,
    PatternMatch,
    PronunciationResult,
    RuntimeBackend,
    Severity,
    StreamingMode,
    SynthesisRecord,
)

runner = CliRunner()


def _record(repeat: int, cold: bool, **over) -> SynthesisRecord:
    base = dict(
        run_id="piper-RUN",
        provider="piper",
        execution_mode=ExecutionMode.LOCAL,
        model="en_US-lessac-medium",
        voice="en_US-lessac-medium",
        runtime_backend=RuntimeBackend.CPU,
        device="cpu",
        text="Take Metformin",
        dataset="healthcare",
        dataset_item_id="med",
        benchmark_suite="pronunciation",
        cold_start=cold,
        audio_duration_ms=900.0,
        wall_time_ms=20.0,
        realtime_factor=0.02,
        sample_rate=22050,
        format=AudioFormat.PCM_S16LE,
        streaming=False,
        streaming_mode=StreamingMode.NONE,
        repeat_index=repeat,
        characters_sent=14,
        audio_path=f"audio/med_r{repeat}.wav",
    )
    base.update(over)
    return SynthesisRecord(**base)


def _pron(passed: bool) -> PronunciationResult:
    return PronunciationResult(
        run_id="piper-RUN",
        dataset="healthcare",
        item_id="med_008",
        repeat_index=0,
        input="Amoxicillin 500 mg",
        category="medication_dosage",
        severity=Severity.HIGH,
        advisory=False,
        audio_path="audio/med_008_r0.wav",
        pattern_matches=[PatternMatch(pattern="five hundred milligrams", matched=passed, per=0.45)],
        passed=passed,
        produced_phonemes="f a ɪ v h ʌ n d ɹ ɪ d",
        transcript="amoxicillin 500 mg",
        raw_transcript="amoxicillin 500 mg",
    )


def test_build_markdown_has_all_sections():
    records = [_record(0, True, model_load_ms=600.0), _record(1, False)]
    md = build_markdown(records, [_pron(False)], title="myrun")

    assert "# TTSBench Report — myrun" in md
    assert "## Run metadata" in md
    assert "## Configuration" in md
    assert "## Latency (local batch)" in md
    assert "## Pronunciation" in md
    assert "## Cost & runtime" in md
    # Backend/device visible.
    assert "| Runtime backend | cpu |" in md
    assert "| Device | cpu |" in md
    # Local cost note.
    assert "$0.0000" in md
    # High-severity failure with a RELATIVE audio link (portable).
    assert "[audio](audio/med_008_r0.wav)" in md
    assert "five hundred milligrams` (PER 0.45)" in md
    # No streaming section for batch-only records.
    assert "## Streaming latency" not in md


def test_build_markdown_includes_streaming_when_present():
    streaming = _record(
        0, True, streaming=True, streaming_mode=StreamingMode.WEBSOCKET, ttfb_ms=80.0, ttfp_ms=120.0
    )
    md = build_markdown([streaming])
    assert "## Streaming latency" in md
    assert "TTFB p50" in md


def test_report_cli_regenerates_from_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = [_record(0, True, model_load_ms=600.0), _record(1, False)]
    (run_dir / "metadata.jsonl").write_text("\n".join(r.model_dump_json() for r in records))
    (run_dir / "pronunciation_results.jsonl").write_text(_pron(False).model_dump_json())

    result = runner.invoke(app, ["report", "--run", str(run_dir)])
    assert result.exit_code == 0, result.output
    report = (run_dir / "report.md").read_text()
    assert "## Pronunciation" in report
    assert "[audio](audio/med_008_r0.wav)" in report


def test_report_cli_rejects_missing_run(tmp_path):
    result = runner.invoke(app, ["report", "--run", str(tmp_path / "nope")])
    assert result.exit_code != 0


def test_report_cli_rejects_bad_format(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata.jsonl").write_text(_record(0, True).model_dump_json())
    result = runner.invoke(app, ["report", "--run", str(run_dir), "--format", "pdf"])
    assert result.exit_code != 0


def test_write_markdown_report_roundtrip(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata.jsonl").write_text(_record(0, True).model_dump_json())
    path = write_markdown_report(run_dir)
    assert path == run_dir / "report.md"
    assert path.exists()
