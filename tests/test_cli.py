"""Trivial CLI smoke tests for the Phase 0 scaffolding."""

from typer.testing import CliRunner

from ttsbench import __version__
from ttsbench.cli import app

runner = CliRunner()


def test_version_is_set():
    assert __version__ == "0.1.0"


def test_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_stub_commands_report_not_implemented():
    for command in ("run", "compare", "validate", "report"):
        result = runner.invoke(app, [command])
        assert result.exit_code == 0
        assert "not implemented yet" in result.output


def test_runtime_command_reports_platform():
    result = runner.invoke(app, ["runtime"])
    assert result.exit_code == 0
    assert "Platform" in result.output
    assert "Backends available" in result.output
