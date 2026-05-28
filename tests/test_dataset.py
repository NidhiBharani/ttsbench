"""Phase 4: dataset schema, loader, validation, and the validate CLI command."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from ttsbench.cli import app
from ttsbench.datasets import load_dataset, validate_dataset
from ttsbench.schemas import Dataset, Severity

runner = CliRunner()

HEALTHCARE = Path("ttsbench/datasets/healthcare.yaml")


def test_healthcare_dataset_loads_and_is_valid():
    dataset = load_dataset(HEALTHCARE)
    assert isinstance(dataset, Dataset)
    assert dataset.name == "healthcare"
    assert 28 <= len(dataset.items) <= 35  # "approximately 30"

    report = validate_dataset(dataset)
    assert report.ok, report.warnings
    # Categories from PRD section 10 are represented.
    for category in ("appointment", "medication_dosage", "date", "time", "insurance_id"):
        assert category in report.per_category
    assert report.advisory_count >= 2  # code-switched items
    # Safety-critical categories should be high severity.
    assert report.per_severity.get("high", 0) >= 1


def test_unique_ids_and_nonempty_expected_forms():
    dataset = load_dataset(HEALTHCARE)
    ids = [item.id for item in dataset.items]
    assert len(ids) == len(set(ids))
    assert all(item.expected_spoken_contains for item in dataset.items)


def test_loader_accepts_bare_list(tmp_path):
    yaml_text = (
        "- id: x1\n"
        "  input: hello\n"
        "  category: misc\n"
        "  severity: low\n"
        "  expected_spoken_contains: [hello]\n"
    )
    path = tmp_path / "mini.yaml"
    path.write_text(yaml_text)
    dataset = load_dataset(path)
    assert dataset.name == "mini"
    assert dataset.items[0].severity is Severity.LOW


def test_validate_warns_on_duplicates_and_empty(tmp_path):
    yaml_text = (
        "name: bad\n"
        "items:\n"
        "  - id: dup\n"
        "    input: a\n"
        "    category: c\n"
        "    severity: low\n"
        "    expected_spoken_contains: [a]\n"
        "  - id: dup\n"
        "    input: b\n"
        "    category: c\n"
        "    severity: low\n"
        "    expected_spoken_contains: []\n"
    )
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text)
    report = validate_dataset(load_dataset(path))
    assert not report.ok
    assert any("duplicate id" in w for w in report.warnings)
    assert any("empty expected" in w for w in report.warnings)


def test_invalid_severity_is_a_schema_error(tmp_path):
    yaml_text = (
        "name: bad\n"
        "items:\n"
        "  - id: x\n"
        "    input: a\n"
        "    category: c\n"
        "    severity: critical\n"
        "    expected_spoken_contains: [a]\n"
    )
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValidationError):
        load_dataset(path)


def test_validate_cli_passes_on_healthcare():
    result = runner.invoke(app, ["validate", str(HEALTHCARE)])
    assert result.exit_code == 0, result.output
    assert "By category" in result.output
    assert "OK: no warnings" in result.output


def test_validate_cli_exits_nonzero_on_schema_error(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("name: x\nitems:\n  - id: y\n    severity: low\n")
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    assert "Schema invalid" in result.output
