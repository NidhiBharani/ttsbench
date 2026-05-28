# TTSBench Implementation Plan

A status-aware execution plan derived from [the build plan](ttsbench_build_plan.md) and the
actual state of the code. Use this to know what is done, what is next, and the concrete steps
for each remaining phase.

## Current status

| Phase | Scope | State |
|-------|-------|-------|
| **0** Repo skeleton | pyproject, CLI stubs, Makefile, CI, README | Done |
| **1** Adapter interface + runtime | `base.py`, `schemas.py`, `runtime.py`, fake-adapter tests | Done — 11 tests pass, `ttsbench runtime` works |
| **2** Piper adapter | `adapters/piper.py`, `synthesize` command | Done — real WAV synthesis works, 15 tests pass |
| **3** Local latency benchmark | `benchmarks/latency.py`, `reports/csv_report.py`, real `run` | Done — first shippable milestone; real Piper run verified, 23 tests pass |
| **4** Healthcare dataset | dataset schema, `healthcare.yaml`, `validate` | Done — 30 items, validate passes, 30 tests pass |
| **5** Pronunciation (phoneme) | `phoneme_match.py`, `phoneme_recognizer.py`, `pronunciation.py` | Done — phoneme round-trip (espeak ref + wav2vec2), Whisper kept advisory; verified, 37 tests pass |
| **6** Markdown report | `markdown_report.py`, `report` command | Done — portable report.md, standalone regen verified, 43 tests pass |
| **7** Kokoro adapter | `adapters/kokoro.py` (MLX/MPS) | Not started |
| **8** Coqui XTTS v2 | `adapters/coqui_xtts.py` | Not started |
| **9** Compare command | `ttsbench compare` | Not started |
| **10** README + example run | README, `ROADMAP.md` | Not started |
| **11** Cloud adapters | Cartesia / ElevenLabs / OpenAI | Not started |

Phase 1 contract is solid: batch vs. streaming adapters are distinguished by the `streaming`
property, and the result schema already represents local batch inference without fabricating
cloud timing fields (`None` -> rendered as `not_applicable`).

## Phase 2 — Piper adapter (next, ~2 evenings)

**Goal:** `ttsbench synthesize --provider piper --text "Hello world" --output hello.wav`
produces a valid WAV; metadata records `execution_mode: local` and the actual backend. No API key.

1. Choose the Piper path: `piper-tts` Python package (or the `piper` binary + ONNX voice model).
   Confirm clean install on Apple Silicon. Add as an optional dep group `[piper]` so the base
   install stays light.
2. Implement `adapters/piper.py` as a batch adapter (`streaming = False`) returning
   `SynthesisResult`. Implement all abstract properties from `adapters/base.py`:
   `provider="piper"`, `execution_mode=LOCAL`, `runtime_backend=CPU` (Piper runs ONNX on CPU by
   default — only report a different backend if actually used), `requested_device`/`actual_device`,
   `sample_rate` (from the voice model config), `audio_format`.
3. Add a `synthesize` CLI command that instantiates the adapter, calls `synthesize`, and writes a
   WAV via a `write_wav(pcm, sample_rate, path)` helper in `utils.py`.
4. Populate `config.py` with model/voice resolution (model dir, default voice, sample rate).
5. Tests (`tests/test_adapters.py`): mock the Piper process/wrapper output; assert metadata says
   `local` + correct backend, and that a valid WAV is written.

**Exit:** WAV produced locally, metadata honest, no paid account required.

## Phase 3 — Local latency benchmark (first shippable, ~3–4 evenings)

**Goal:** real `ttsbench run` producing a run dir with `audio/`, `metadata.jsonl`, `metrics.csv`,
`summary.txt`.

1. `benchmarks/latency.py`: adapter + texts + repeats -> one `SynthesisRecord` per (text, repeat).
   Batch path computes model-load / cold / warm synthesis time, audio duration, realtime factor.
   Leave streaming `t1–t4`/`ttfb` as `None` for batch.
2. `reports/csv_report.py`: one row per synthesis.
3. Real `ttsbench run`: `--provider --suite latency --text/--dataset --repeats (default 5)
   --device --output --fail-on`.
4. Write `metadata.jsonl` + WAVs + `metrics.csv`; aggregate p50/p95/p99 into `summary.txt`,
   warning when repeat count is too low.

**Exit:** no-cost local Piper latency benchmark — first stop point.

## Phases 4–6 — Dataset, pronunciation, report (~7–9 evenings)

- **4** Healthcare dataset: pydantic dataset schema in `schemas.py`, `datasets/healthcare.yaml`
  (~30 items), real `ttsbench validate` with stats.
- **5** Pronunciation (phoneme round-trip): pass/fail is sound-based, not word-based, to avoid
  ASR language-model "corrections" (e.g. "Cho"->"show") and digit/word normalization noise
  ("500"=="five hundred"). Reference phonemes come from espeak-ng (bundled with Piper); produced
  phonemes from a wav2vec2 eSpeak CTC recognizer (`evaluators/phoneme_recognizer.py`, decoded from
  `vocab.json` to avoid the `phonemizer`/system-espeak dependency). Matching is approximate-
  substring phoneme error rate with a tolerance (default 0.30) in `evaluators/phoneme_match.py`.
  faster-whisper word ASR is kept as **advisory** human-readable context only. SHA-256 caches for
  both recognizers; `run` gains `--suite latency,pronunciation`, `--phoneme-tolerance`, `--no-asr`.
  Verified on healthcare: pass rate rose from ~12% (word ASR, full of false negatives) to ~52%
  (phoneme), with remaining failures being genuine — e.g. Piper does not expand "mg" to
  "milligrams". The legacy `expected_form_match.py` (word substring matcher) remains for reference.
- **6** `reports/markdown_report.py` + standalone `ttsbench report --run … --format md`
  regenerating from artifacts with relative audio links.

## Phases 7–11 — More adapters, compare, cloud (~15–22 evenings)

- **7** Kokoro (MLX or PyTorch MPS) with real device/fallback reporting.
- **8** Coqui XTTS v2 (CPU/MPS), separate model-load timing.
- **9** `ttsbench compare --runs a b` side-by-side markdown with deltas + config diffs surfaced.
- **10** README quickstart + example run + `ROADMAP.md`.
- **11** Cloud adapters (Cartesia / ElevenLabs / OpenAI) with `perf_counter_ns` chunk timing,
  price tables in `config.py`, `--dry-run` cost estimation.

## Recommended path

Start with Phase 2, then Phase 3 to reach the first usable milestone (~5–6 evenings combined):
a no-cost local benchmark exercising the full run/report plumbing before any heavier model or
paid API work.
