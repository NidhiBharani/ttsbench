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
def run() -> None:
    """Run a benchmark suite against a provider."""
    typer.echo(f"run: {_NOT_IMPLEMENTED}")


@app.command()
def synthesize(
    text: str = typer.Option(..., "--text", help="Text to synthesize."),
    output: str = typer.Option(..., "--output", "-o", help="Output WAV path."),
    provider: str = typer.Option("piper", "--provider", help="TTS provider/adapter."),
    voice: str | None = typer.Option(None, "--voice", help="Voice name."),
    device: str | None = typer.Option(None, "--device", help="Requested device."),
) -> None:
    """Synthesize one utterance and write a WAV file (one-off helper)."""
    if provider != "piper":
        raise typer.BadParameter(f"provider '{provider}' is not implemented yet")

    from ttsbench.adapters.piper import PiperAdapter
    from ttsbench.utils import write_wav

    adapter = PiperAdapter(voice=voice, device=device)
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
def validate() -> None:
    """Validate a dataset file."""
    typer.echo(f"validate: {_NOT_IMPLEMENTED}")


@app.command()
def report() -> None:
    """Generate a report from an existing run directory."""
    typer.echo(f"report: {_NOT_IMPLEMENTED}")


@app.command()
def runtime() -> None:
    """Inspect the local runtime: platform, CPU, memory, acceleration backends."""
    from ttsbench.runtime import detect_runtime, format_runtime

    typer.echo(format_runtime(detect_runtime()))


if __name__ == "__main__":
    app()
