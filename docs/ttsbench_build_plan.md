# TTSBench Build Plan

Sequential phases for building TTSBench v1. Each phase produces something usable on its own and unblocks the next.

Estimates assume solo work with Codex doing most of the typing, evenings and weekends.

The build is now **local-first**:

1. Start with open-source models that run on the laptop.
2. Record the actual runtime backend used: CPU, Apple MPS, MLX, Core ML, ONNX, CUDA, or provider-hosted.
3. Add paid cloud providers only after local benchmarking, pronunciation evaluation, and reporting are useful.

## Phase 0: Repo Skeleton and Scaffolding

**Goal:** A working repo with the directory layout, package install, CLI entry point, and a no-op `ttsbench --help` that runs.

Tasks:

1. Initialize repo with `pyproject.toml`, `ruff`, `pytest`, `mypy` (light).
2. Create the package layout from section 8 of the PRD, with empty module stubs.
3. Wire up `cli.py` with `typer` or `click`. Implement `ttsbench --help` and stub commands for `run`, `compare`, `validate`, `report`, and `runtime` that print "not implemented yet".
4. Add a `Makefile` or `justfile` with `install`, `test`, `lint`, `format`.
5. Add `.env.example`, `.gitignore` (exclude `runs/`, `*.wav`, model cache folders, `.env`), and a minimal README that says what the project is.
6. Set up GitHub Actions with one workflow: lint + test on push.

Exit criteria:

- `pip install -e .` works in a clean venv.
- `ttsbench --help` prints help text.
- CI is green on a trivial test.

Estimated time: 1 evening.

## Phase 1: Adapter Interface + Runtime Metadata

**Goal:** Define the adapter contract and runtime metadata before binding to any one model.

Tasks:

1. Define `adapters/base.py`: abstract base class with `synthesize(text, voice, **params) -> AsyncIterator[AudioChunk] | SynthesisResult`.
2. Define `AudioChunk` dataclass: raw bytes, receive timestamp in `perf_counter_ns`, sequence index.
3. Define `SynthesisResult` for local batch adapters that return a completed audio payload instead of streamed chunks.
4. Define adapter metadata properties:
   - provider/runtime name
   - execution mode: `local` or `cloud`
   - model
   - voice
   - runtime backend: `cpu`, `mps`, `mlx`, `coreml`, `onnx`, `cuda`, `provider_hosted`, or `unknown`
   - requested device
   - actual device/backend used
   - sample rate
   - audio format
   - streaming support
   - streaming mode
5. Define `schemas.py` for the result record in PRD section 9.
6. Implement `runtime.py` and `ttsbench runtime` to print platform, CPU, memory, and available acceleration backends.
7. Add tests for schema serialization, `not_applicable` timing fields, and runtime metadata.

Exit criteria:

- `ttsbench runtime` reports local platform information.
- A fake adapter can return either streamed chunks or a batch result.
- Result records can represent local batch inference without fabricating cloud timing fields.

Estimated time: 1 to 2 evenings.

## Phase 2: Piper Adapter

**Goal:** Synthesize speech locally with Piper and save a WAV file from the CLI. No paid API accounts required.

Piper is the first adapter because it is lightweight, CPU-friendly, and good for validating the benchmark pipeline.

Tasks:

1. Add Piper install/setup notes to the README. Prefer a simple CLI or Python wrapper path that works on Apple Silicon.
2. Implement `adapters/piper.py`.
3. Default Piper to CPU execution and record `runtime_backend: "cpu"` unless a specific accelerated runtime is actually used.
4. Add `ttsbench synthesize` (one-off command, not necessarily part of the final user workflow) that calls the adapter and writes a WAV.
5. Record model path, voice path/name, sample rate, audio format, execution mode, backend, and device metadata.
6. Write adapter tests with a mocked Piper process or mocked wrapper output.

Exit criteria:

- `ttsbench synthesize --provider piper --text "Hello world" --output hello.wav` produces a valid WAV.
- The result metadata says `execution_mode: local` and records the actual backend.
- No API key or paid account is required.

Estimated time: 2 evenings.

## Phase 3: Local Latency Benchmark

**Goal:** Run the latency suite against Piper and produce a CSV with metrics that make sense for local batch inference.

Tasks:

1. Implement `benchmarks/latency.py`. It takes an adapter, a list of input texts, and a repeat count, and produces one result record per (text, repeat) pair.
2. Support both streamed and batch-style adapter outputs.
3. For local batch adapters, compute:
   - model load time, if loading happens inside the run
   - cold synthesis time
   - warm synthesis time
   - total synthesis wall time
   - audio duration
   - realtime factor
4. For streaming adapters, compute:
   - t0: just before synthesize call
   - t1: first chunk received
   - t2: first decodable audio frame
   - t3: first playable buffer, configurable target, default 250ms of decoded PCM
   - t4: stream completion
   - ttfb, ttfa, ttfp, total wall time, realtime factor, inter-chunk gap stats
5. Use `not_applicable` or null fields for unsupported metrics. Do not invent `ttfb` for local batch inference.
6. Implement `reports/csv_report.py`: writes `metrics.csv` with one row per synthesis.
7. Implement the real `ttsbench run` command. Accept `--provider`, `--suite latency`, `--text` or `--dataset`, `--repeats` (default 5 for local), `--device`, `--output`, and reserved `--fail-on`.
8. Save `metadata.jsonl`, WAV files, and `metrics.csv` into the run output directory.
9. Aggregate p50/p95/p99 where sample counts are large enough and write a simple `summary.txt`. Warn when percentiles are based on too few repeats.

Exit criteria:

- `ttsbench run --provider piper --suite latency --text "Hello world" --repeats 5 --output runs/test` produces a directory with `audio/`, `metadata.jsonl`, `metrics.csv`, and `summary.txt`.
- Timing numbers are sane for local batch inference.
- Unsupported cloud-style fields are clearly marked as unavailable.

Estimated time: 3 to 4 evenings.

**This is the first shippable milestone.** After this phase you have a no-cost local benchmark that validates the core run/report plumbing.

## Phase 4: Healthcare Dataset

**Goal:** A real dataset of approximately 30 hand-crafted items, validated and ready for use.

Tasks:

1. Define dataset YAML schema in `schemas.py` with pydantic.
2. Write `datasets/healthcare.yaml` with approximately 30 items across the categories from PRD section 10: appointments, doctor names, medications, dosages, dates, times, insurance IDs, plus 2 to 3 code-switched advisory items.
3. Each item: `id`, `input`, `expected_spoken_contains` (list), `category`, `severity`.
4. Implement `ttsbench validate datasets/healthcare.yaml`: checks schema, prints stats (items per category, severity distribution), warns on duplicate IDs or empty expected forms.
5. Hand-listen to approximately 5 items synthesized from Piper, verify the inputs are realistic and the expected forms are right.

Exit criteria:

- 30 items, schema-valid, with realistic expected_spoken_contains lists.
- `ttsbench validate` passes.
- The items actually represent things a healthcare voice agent would say.

Estimated time: 2 to 3 evenings. The writing is the work, not the code.

## Phase 5: Pronunciation Benchmark with ASR Round-Trip

**Goal:** Run the pronunciation suite end to end. Produce per-item pass/fail with audio, transcript, and expected forms surfaced together.

Tasks:

1. Add a local ASR backend. Start with `faster-whisper`; optionally add an Apple Silicon-friendly backend later if needed.
2. Implement `evaluators/asr_roundtrip.py`: takes a WAV path, returns a transcript string and word-level timing when available.
3. For high-severity items, record the raw un-normalized ASR output when the backend supports it so hallucinated "corrections" are visible.
4. Implement `evaluators/expected_form_match.py`. Normalization (applied to both transcript and each pattern before matching): lowercase, strip punctuation, collapse whitespace, replace hyphens with spaces.
5. Each normalized pattern is a substring match against the normalized transcript. Return per-pattern pass/fail and an overall item pass/fail.
6. Implement `benchmarks/pronunciation.py`: synthesizes each dataset item sequentially, then runs ASR through a worker pool (`--asr-workers`, default 2).
7. Extend `ttsbench run` to accept `--suite latency,pronunciation` (comma-separated).
8. Write results into `metrics.csv` and `pronunciation_results.jsonl`.
9. Cache ASR results by audio-file SHA-256 so re-runs of just the matcher are fast.

Exit criteria:

- `ttsbench run --provider piper --suite pronunciation --dataset datasets/healthcare.yaml --repeats 3` completes.
- For each item you can see: input text, audio path, transcript, per-pattern matches, overall pass/fail.
- High-severity failures are clearly marked.

Estimated time: 3 to 4 evenings. ASR may dominate runtime on CPU.

## Phase 6: Markdown Report

**Goal:** A single self-contained markdown report per run that a reader can use to make a decision.

Tasks:

1. Implement `reports/markdown_report.py`. Sections:
   - Run metadata (provider, model, voice, execution mode, runtime backend, device, timestamp, repeat count, dataset).
   - Configuration block (all provider/runtime params, sample rate, format, streaming mode).
   - Local latency summary table: model load time, cold synthesis time, warm synthesis time, total wall time, realtime factor.
   - Streaming latency summary table when available: ttfb, ttfa, ttfp, total wall time, realtime factor, inter-chunk gaps.
   - Pronunciation results: total items, pass rate, per-category breakdown, high-severity failures listed with input + transcript + expected forms + audio link.
   - Cost/runtime summary: local API cost is zero; cloud costs shown when applicable.
2. Audio links use relative paths so the run directory is portable.
3. Implement `ttsbench report --run runs/foo --format md` as a standalone command that regenerates the report from run artifacts without re-running synthesis.

Exit criteria:

- Open `report.md` in any markdown viewer and you can understand the run without reading source code.
- Audio links work when the directory is moved or zipped.
- Local CPU/GPU backend information is visible in the report.

Estimated time: 2 evenings.

## Phase 7: Kokoro Adapter with Apple Silicon GPU Path

**Goal:** Add a higher-quality lightweight local TTS model and prefer GPU acceleration on Apple Silicon where practical.

Tasks:

1. Pick the most maintainable Kokoro runtime path for Apple Silicon. Prefer MLX if it is stable enough; otherwise use PyTorch MPS.
2. Add optional dependencies for Kokoro without making Piper installation heavy.
3. Implement `adapters/kokoro.py`.
4. Support `--device cpu`, `--device mps`, and/or `--device mlx` according to the chosen runtime.
5. Record requested device and actual backend used. If the runtime falls back to CPU, report that explicitly.
6. Run the local latency benchmark against Kokoro and compare with Piper.
7. Add adapter tests that mock model loading and synthesis.

Exit criteria:

- `ttsbench synthesize --provider kokoro --device mps --text "Hello world" --output hello.wav` works when the runtime supports it.
- If GPU is unavailable or unsupported, the report clearly records CPU fallback.
- `ttsbench compare --runs runs/piper_healthcare runs/kokoro_healthcare_mps` can show meaningful local differences once compare exists.

Estimated time: 3 to 5 evenings. Runtime packaging is the main uncertainty.

## Phase 8: Coqui XTTS v2 Adapter

**Goal:** Add a heavier local model for multilingual and voice-cloning-capable comparison.

Tasks:

1. Add optional dependencies for Coqui XTTS v2 separately from the base install.
2. Implement `adapters/coqui_xtts.py`.
3. Support `--device cpu` and `--device mps` where PyTorch MPS works.
4. Record reference-voice metadata when voice cloning is used, without storing sensitive source audio unexpectedly.
5. Record model load time separately because XTTS is heavier.
6. Run latency and pronunciation benchmarks on a small subset before running the full dataset.
7. Document known Apple Silicon limitations and fallback behavior.

Exit criteria:

- `ttsbench synthesize --provider coqui_xtts --text "Hello world" --output hello.wav` works locally.
- Runtime metadata shows whether CPU or MPS was actually used.
- The report makes it clear when XTTS is slower but more capable.

Estimated time: 3 to 6 evenings. Apple Silicon compatibility is the risk.

## Phase 9: Compare Command

**Goal:** A side-by-side comparison report across two or more runs.

Tasks:

1. Implement `ttsbench compare --runs runs/a runs/b --output reports/compare.md`.
2. Show runs side by side:
   - local latency metrics
   - streaming latency metrics when available
   - pronunciation pass rates
   - high-severity failures
   - cost/runtime summary
   - backend/device metadata
3. Highlight deltas.
4. Surface configuration differences prominently at the top, because they explain most deltas.
5. Call out items where one run passed and another failed pronunciation, with audio links to both.

Exit criteria:

- Run Piper, Kokoro, and XTTS on the healthcare dataset, run `ttsbench compare`, and get one markdown file that tells the comparison story.

Estimated time: 2 evenings.

## Phase 10: README and Local Example Run

**Goal:** A complete, navigable repo that a new user can understand and run locally within 10 minutes, assuming dependencies are installed.

Tasks:

1. README:
   - One paragraph on what TTSBench is and what problem it solves.
   - Quickstart: install, run Piper locally, see output.
   - Hardware/runtime note for Apple Silicon GPU usage: CUDA is not available; use MPS, MLX, Core ML, or CPU depending on adapter support.
   - Embedded example run comparing Piper and Kokoro on the healthcare dataset, with screenshots of the markdown report, a latency chart, and 3 to 5 embedded audio samples.
   - A "What this tool does not do" section: no universal ranking, no emotion scoring, no claim of fairness across providers or runtimes without configuration parity.
   - Link to the roadmap.
2. ROADMAP.md:
   - Cartesia, ElevenLabs, and OpenAI cloud adapters.
   - Stability benchmark.
   - Emotion/style benchmark.
   - Multilingual suite.
   - Voice-agent-context dataset.
   - Streamlit UI.

Exit criteria:

- A new user can land on the repo, read the README, run the local quickstart, and have an opinion within 10 minutes.

Estimated time: 2 to 3 evenings. The example run takes a few hours of running things, listening to audio, and picking good samples.

## Phase 11: Cloud Provider Adapters

**Goal:** Add paid cloud providers after the local benchmark is useful.

Tasks:

1. Implement `adapters/cartesia.py` with the streaming websocket API. Capture `perf_counter_ns()` when each chunk is received inside the recv loop.
2. Implement `adapters/elevenlabs.py`. Prefer websocket or another streaming path that provides fair chunk timing.
3. Implement `adapters/openai.py`. Be honest in metadata if the API behaves like HTTP chunking rather than native realtime streaming.
4. Add cloud price tables in `config.py`.
5. Add `--dry-run` cost estimation for cloud datasets.
6. Update report examples with one local-vs-cloud or cloud-vs-cloud comparison.

Exit criteria:

- `ttsbench run --provider cartesia --suite latency --text "Hello world" --repeats 5` works when `CARTESIA_API_KEY` is set.
- Cloud reports show streaming-specific metrics such as ttfb, ttfa, ttfp, and inter-chunk gaps.
- Cloud reports show estimated cost and provider billing fields where available.

Estimated time: 5 to 8 evenings across all three providers.

## Total

Roughly 25 to 36 evenings, depending mostly on local runtime compatibility for Kokoro and XTTS.

The first useful local milestone is much earlier:

- **After Phase 3**: local Piper latency benchmark with no paid API usage.
- **After Phase 5**: local pronunciation benchmark.
- **After Phase 7**: meaningful CPU-vs-GPU or Piper-vs-Kokoro comparison.
- **After Phase 10**: polished local-first project.
- **After Phase 11**: local and cloud provider comparison.

## Stop Points

You can stop after any of these and still have something useful:

- **After Phase 3**: no-cost local latency measurement for Piper.
- **After Phase 5**: no-cost local pronunciation benchmark on healthcare scripts.
- **After Phase 7**: local model comparison with Apple Silicon GPU metadata.
- **After Phase 10**: polished local-first benchmarking tool.
- **After Phase 11**: broader benchmark that compares local models and paid cloud providers.
