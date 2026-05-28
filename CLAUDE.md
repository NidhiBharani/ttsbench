# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TTSBench is a local-first benchmarking toolkit for TTS (text-to-speech) systems, measuring latency, pronunciation reliability, and runtime practicality. The project is built in sequential phases — see `@docs/ttsbench_build_plan.md` for the full roadmap and phase exit criteria.

**Current state:** Phase 0 complete (scaffolding). Most adapters, benchmarks, evaluators, and reports are stubs awaiting implementation in later phases.

## Commands

```bash
make install   # pip install -e ".[dev]"
make test      # pytest
make lint      # ruff check . && mypy ttsbench
make format    # ruff format . && ruff check --fix .
```

Run a single test: `pytest tests/test_cli.py::test_name`

## Code Style

- Line length: **100** (ruff enforces; differs from the 88 default)
- Enabled ruff rules: `E`, `F`, `I`, `UP`, `B`
- Type checking: mypy with `ignore_missing_imports = true` (light strictness)
- Python minimum: 3.10 — use union syntax (`X | Y`) not `Optional[X]`

## Testing

- `asyncio_mode = "auto"` — async test functions work without decorators
- Test path: `tests/`
- Tests are smoke/unit only; no real TTS models or API keys needed

## Architecture

- **Adapter pattern:** Each TTS provider gets an adapter in `ttsbench/adapters/`, inheriting from `adapters/base.py`. Adapters expose both streaming (`AsyncIterator[AudioChunk]`) and batch (`SynthesisResult`) modes.
- **Local-first:** Build and validate local adapters (Piper, Kokoro, Coqui XTTS) before cloud adapters (Phase 11+). Cloud adapters require API keys from `.env` (see `.env.example`).
- **Pydantic v2** for all data models in `schemas.py`.

## Commit Style

Conventional commits: `feat:`, `fix:`, `chore:`, `test:`, `docs:`, `refactor:`. Keep the subject line under 72 characters.

## Docs Reference

- Phase roadmap: `@docs/ttsbench_build_plan.md`
- Full PRD: `@docs/ttsbench_prd_v1.md`
- Metric definitions: `@docs/metrics_explained.md`
