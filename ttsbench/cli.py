"""Command-line entry point for TTSBench."""

from __future__ import annotations

import typer

app = typer.Typer(
    add_completion=False,
    help="Local-first benchmarking toolkit for text-to-speech latency, "
    "pronunciation, and runtime practicality.",
    no_args_is_help=True,
)

_NOT_IMPLEMENTED = "not implemented yet"


@app.command()
def run(
    output: str = typer.Option(..., "--output", "-o", help="Run output directory."),
    provider: str = typer.Option("piper", "--provider", help="TTS provider/adapter."),
    suite: str = typer.Option("latency", "--suite", help="Comma-separated: latency,pronunciation."),
    text: list[str] = typer.Option(None, "--text", help="Text to synthesize (repeatable)."),
    dataset: str | None = typer.Option(None, "--dataset", help="Dataset YAML file."),
    repeats: int = typer.Option(5, "--repeats", help="Repeats per item (default 5 local)."),
    voice: str | None = typer.Option(None, "--voice", help="Voice name."),
    device: str | None = typer.Option(None, "--device", help="Requested device."),
    asr_workers: int = typer.Option(2, "--asr-workers", help="Evaluation worker pool size."),
    asr_model: str = typer.Option("base.en", "--asr-model", help="faster-whisper (advisory)."),
    phoneme_model: str = typer.Option(
        "facebook/wav2vec2-lv-60-espeak-cv-ft", "--phoneme-model", help="Phoneme recognizer."
    ),
    phoneme_tolerance: float = typer.Option(
        0.30, "--phoneme-tolerance", help="Max phoneme error rate to count a pattern as matched."
    ),
    no_asr: bool = typer.Option(False, "--no-asr", help="Skip the advisory word transcript."),
    fail_on: str | None = typer.Option(None, "--fail-on", help="Reserved; not yet used."),
) -> None:
    """Run one or more benchmark suites against a provider."""
    from datetime import datetime, timezone
    from pathlib import Path

    from ttsbench.benchmarks.latency import BenchmarkItem, benchmark_latency
    from ttsbench.reports.csv_report import write_csv
    from ttsbench.reports.summary import write_summary
    from ttsbench.schemas import Dataset, DatasetItem, Severity

    suites = [s.strip() for s in suite.split(",") if s.strip()]
    allowed = {"latency", "pronunciation"}
    unsupported = [s for s in suites if s not in allowed]
    if unsupported:
        raise typer.BadParameter(f"unknown suite(s) {unsupported}; allowed: {sorted(allowed)}")
    if not suites:
        raise typer.BadParameter("provide at least one --suite")
    from ttsbench.adapters import LOCAL_PROVIDERS, get_adapter

    if provider not in LOCAL_PROVIDERS:
        raise typer.BadParameter(f"provider '{provider}' is not implemented yet")
    if "pronunciation" in suites and not dataset:
        raise typer.BadParameter("the pronunciation suite requires --dataset")
    if not dataset and not text:
        raise typer.BadParameter("provide --dataset or at least one --text")
    if fail_on is not None:
        typer.echo("note: --fail-on is reserved and currently ignored")

    if dataset:
        from ttsbench.datasets import load_dataset

        ds = load_dataset(dataset)
    else:
        ds = Dataset(
            name="adhoc",
            items=[
                DatasetItem(id=f"item{i:03d}", input=t, category="adhoc", severity=Severity.LOW)
                for i, t in enumerate(text)
            ],
        )

    adapter = get_adapter(provider, voice=voice, device=device)
    run_id = f"{provider}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = Path(output)
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    pron_results = []

    if "latency" in suites:
        items = [
            BenchmarkItem(
                id=item.id, text=item.input, dataset_item_id=(item.id if dataset else None)
            )
            for item in ds.items
        ]
        records += benchmark_latency(
            adapter,
            items,
            repeats,
            run_id,
            suite="latency",
            dataset=ds.name if dataset else None,
            audio_dir=audio_dir,
        )

    if "pronunciation" in suites:
        from ttsbench.benchmarks.pronunciation import benchmark_pronunciation
        from ttsbench.evaluators.asr_roundtrip import CachingASR, FasterWhisperASR
        from ttsbench.evaluators.phoneme_match import EspeakReference
        from ttsbench.evaluators.phoneme_recognizer import (
            CachingPhonemeRecognizer,
            Wav2Vec2PhonemeRecognizer,
        )

        recognizer = CachingPhonemeRecognizer(
            Wav2Vec2PhonemeRecognizer(model_name=phoneme_model, device=device or "cpu"),
            out_dir / ".phoneme_cache.json",
        )
        asr = (
            None
            if no_asr
            else CachingASR(FasterWhisperASR(model_size=asr_model), out_dir / ".asr_cache.json")
        )
        p_records, pron_results = benchmark_pronunciation(
            adapter,
            ds,
            repeats,
            run_id,
            recognizer,
            run_dir=out_dir,
            audio_dir=audio_dir,
            reference=EspeakReference(),
            tolerance=phoneme_tolerance,
            asr=asr,
            workers=asr_workers,
        )
        records += p_records

    with (out_dir / "metadata.jsonl").open("w") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
    write_csv(records, out_dir / "metrics.csv")

    if pron_results:
        with (out_dir / "pronunciation_results.jsonl").open("w") as fh:
            for result in pron_results:
                fh.write(result.model_dump_json() + "\n")

    summary_path = write_summary(records, out_dir / "summary.txt", pronunciation=pron_results)

    from ttsbench.reports.markdown_report import write_markdown_report

    write_markdown_report(out_dir)

    typer.echo(f"Wrote {len(records)} synthesis record(s) to {out_dir}/")
    typer.echo(summary_path.read_text())


@app.command()
def synthesize(
    text: str = typer.Option(..., "--text", help="Text to synthesize."),
    output: str = typer.Option(..., "--output", "-o", help="Output WAV path."),
    provider: str = typer.Option("piper", "--provider", help="TTS provider/adapter."),
    voice: str | None = typer.Option(None, "--voice", help="Voice name."),
    device: str | None = typer.Option(None, "--device", help="Requested device."),
) -> None:
    """Synthesize one utterance and write a WAV file (one-off helper)."""
    from ttsbench.adapters import LOCAL_PROVIDERS, get_adapter
    from ttsbench.utils import write_wav

    if provider not in LOCAL_PROVIDERS:
        raise typer.BadParameter(f"provider '{provider}' is not implemented yet")

    adapter = get_adapter(provider, voice=voice, device=device)
    result = adapter.synthesize(text)
    path = write_wav(result.audio, result.sample_rate, output)

    typer.echo(
        f"Wrote {path} "
        f"(provider={adapter.provider}, execution_mode={adapter.execution_mode.value}, "
        f"backend={adapter.runtime_backend.value}, voice={adapter.voice}, "
        f"sample_rate={result.sample_rate})"
    )


@app.command()
def compare() -> None:
    """Compare two or more runs side by side."""
    typer.echo(f"compare: {_NOT_IMPLEMENTED}")


@app.command()
def validate(
    path: str = typer.Argument(..., help="Path to a dataset YAML file."),
) -> None:
    """Validate a dataset file and print category/severity stats."""
    from pydantic import ValidationError

    from ttsbench.datasets import load_dataset, validate_dataset

    try:
        dataset = load_dataset(path)
    except ValidationError as exc:
        typer.echo(f"Schema invalid: {path}")
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc

    report = validate_dataset(dataset)
    typer.echo(f"Dataset '{dataset.name}': {report.total} items")
    typer.echo("  By category:")
    for category, count in sorted(report.per_category.items()):
        typer.echo(f"    {category}: {count}")
    typer.echo("  By severity:")
    for severity in ("high", "medium", "low"):
        if severity in report.per_severity:
            typer.echo(f"    {severity}: {report.per_severity[severity]}")
    typer.echo(f"  Advisory items: {report.advisory_count}")

    if report.warnings:
        typer.echo(f"  {len(report.warnings)} warning(s):")
        for warning in report.warnings:
            typer.echo(f"    - {warning}")
    else:
        typer.echo("  OK: no warnings")


@app.command()
def report(
    run: str = typer.Option(..., "--run", help="Run output directory."),
    output_format: str = typer.Option("md", "--format", help="Report format (md)."),
) -> None:
    """Generate a report from an existing run directory (no re-synthesis)."""
    from pathlib import Path

    if output_format != "md":
        raise typer.BadParameter(f"format '{output_format}' not supported (md only)")

    run_dir = Path(run)
    if not (run_dir / "metadata.jsonl").exists():
        raise typer.BadParameter(f"no metadata.jsonl found in {run_dir}")

    from ttsbench.reports.markdown_report import write_markdown_report

    path = write_markdown_report(run_dir)
    typer.echo(f"Wrote {path}")


@app.command()
def runtime() -> None:
    """Inspect the local runtime: platform, CPU, memory, acceleration backends."""
    from ttsbench.runtime import detect_runtime, format_runtime

    typer.echo(format_runtime(detect_runtime()))


if __name__ == "__main__":
    app()
