"""Command-line entry point for TTSBench."""

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
