"""Dataset loading and validation. Phase 4.

A dataset file is YAML, either a bare list of items or a mapping with ``name``
and ``items``. ``load_dataset`` returns a validated :class:`Dataset`;
``validate_dataset`` produces stats and non-fatal warnings (duplicate ids, empty
expected forms) for the ``ttsbench validate`` command.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ttsbench.schemas import Dataset


def load_dataset(path: str | Path) -> Dataset:
    """Load and schema-validate a dataset YAML file."""
    file_path = Path(path)
    raw = yaml.safe_load(file_path.read_text())
    if isinstance(raw, list):
        raw = {"name": file_path.stem, "items": raw}
    return Dataset.model_validate(raw)


@dataclass
class ValidationReport:
    total: int
    per_category: dict[str, int]
    per_severity: dict[str, int]
    advisory_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


def validate_dataset(dataset: Dataset) -> ValidationReport:
    """Compute stats and collect non-fatal warnings for a loaded dataset."""
    per_category = Counter(item.category for item in dataset.items)
    per_severity = Counter(item.severity.value for item in dataset.items)
    advisory_count = sum(1 for item in dataset.items if item.advisory)

    warnings: list[str] = []
    seen: set[str] = set()
    for item in dataset.items:
        if item.id in seen:
            warnings.append(f"duplicate id: {item.id}")
        seen.add(item.id)
        if not item.expected_spoken_contains:
            warnings.append(f"empty expected_spoken_contains: {item.id}")

    return ValidationReport(
        total=len(dataset.items),
        per_category=dict(per_category),
        per_severity=dict(per_severity),
        advisory_count=advisory_count,
        warnings=warnings,
    )
