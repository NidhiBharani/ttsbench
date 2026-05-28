# TTSBench

A local-first benchmarking toolkit for measuring text-to-speech systems on the
things that break voice agents in production: **latency**, **pronunciation
reliability**, and **runtime practicality**.

TTSBench is not a "best-sounding TTS" leaderboard. It gives engineering teams a
repeatable way to measure whether a model, voice, runtime backend, or provider
is fit for their specific real-time voice product, and to catch regressions when
any of those change.

v1 is local-first: it starts with open-source models (Piper, Kokoro, Coqui XTTS
v2) that run on a developer laptop with no paid API usage. Paid cloud providers
(Cartesia, ElevenLabs, OpenAI) come later as comparison targets.

## Status

Early development. This repo currently contains the Phase 0 scaffolding: package
layout, CLI entry point, and tooling. The benchmark commands are stubs that print
"not implemented yet" until later phases land. See
[`ttsbench_build_plan.md`](ttsbench_build_plan.md) for the phase roadmap and
[`ttsbench_prd_v1.md`](ttsbench_prd_v1.md) for the full product spec.

## Install

```bash
pip install -e ".[dev]"
ttsbench --help
```

## Development

```bash
make install   # editable install with dev dependencies
make test      # run pytest
make lint      # ruff check + mypy
make format    # ruff format + autofix
```
