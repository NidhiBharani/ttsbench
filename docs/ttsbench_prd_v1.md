# TTSBench PRD (v1)

## 1. Product Summary

**TTSBench** is an open-source benchmarking toolkit for measuring text-to-speech systems on the things that actually break voice agents in production: **latency**, **pronunciation reliability**, and **runtime practicality**.

The goal is not to crown the best-sounding TTS. The goal is to give engineering teams a repeatable way to measure whether a TTS model, voice, runtime backend, or provider is fit for their specific real-time voice product, and to catch regressions when any of those change.

v1 is **local-first**. It starts with open-source TTS models that can run on a developer laptop without paid API usage:

- **Piper** as the fast CPU-friendly baseline.
- **Kokoro** as the lightweight local quality baseline, with Apple Silicon GPU support preferred where practical.
- **Coqui XTTS v2** as the heavier local reference model for multilingual and voice-cloning-capable workflows.

Paid cloud providers such as Cartesia, ElevenLabs, and OpenAI remain important comparison targets, but they move after the local benchmark path is working.

## 2. Problem

Most TTS comparisons are demo videos or one-shot blind tests. Production voice apps need something different.

Teams shipping real-time voice agents need to know:

- How quickly audio actually becomes playable, not just how quickly a command returns.
- Whether local hardware can run a model fast enough for the target interaction.
- Whether names, dates, dosages, IDs, SKUs, and domain terms are spoken correctly.
- Whether changing a model, voice, runtime backend, or text-normalization rule silently regresses output.
- What the cost profile looks like for local inference versus paid provider usage.

This kind of measurement is currently done ad hoc by every team that ships a voice agent, and the work is usually thrown away after the decision is made. TTSBench packages it.

## 3. Target Users

Primary users:

- Voice AI engineers integrating TTS into real-time agents.
- Applied AI teams evaluating local and cloud TTS options.
- Forward-deployed engineers evaluating TTS models against customer-specific scripts.
- Teams running regression checks before promoting a new model, voice, or runtime backend.

Secondary users:

- Startup teams choosing a voice stack without committing to paid providers too early.
- Solution engineers running customer-suitability evaluations.

## 4. Core Use Cases

### 4.1 Local Model Comparison

A developer compares local open-source TTS models on the same dataset, with identical text and configuration captured, to decide which one is fit for their use case.

```bash
ttsbench run \
  --provider piper \
  --suite latency,pronunciation \
  --dataset datasets/healthcare.yaml \
  --repeats 5 \
  --device cpu \
  --output runs/piper_healthcare
```

```bash
ttsbench run \
  --provider kokoro \
  --suite latency,pronunciation \
  --dataset datasets/healthcare.yaml \
  --repeats 5 \
  --device mps \
  --output runs/kokoro_healthcare_mps
```

### 4.2 Local Hardware Suitability

A developer checks whether a laptop or workstation can run a model quickly enough for the target voice product. The report captures runtime backend, device, model load time, cold synthesis time, warm synthesis time, total synthesis time, and realtime factor.

### 4.3 Regression Testing

A team ensures that switching a TTS model, voice, runtime backend, or text-normalization rule does not regress production output. Run TTSBench in CI against a fixed dataset and fail the build on threshold violations via `--fail-on` (see section 11.1).

### 4.4 Customer-Specific Evaluation

An FDE uploads a customer's actual scripts (appointment confirmations, claim status, support flows) and produces a report showing latency, pronunciation reliability, runtime metadata, and cost profile on that customer's exact workload.

### 4.5 Cloud Provider Comparison

After the local benchmark path works, a developer can run the same dataset against paid providers such as Cartesia, ElevenLabs, or OpenAI. Cloud results add API/network timing and provider billing data.

## 5. Product Goals

TTSBench should:

1. Produce reproducible latency and pronunciation measurements.
2. Distinguish between local inference speed, cloud API responsiveness, and actual usable audio.
3. Surface pronunciation failures on production-like text with both transcript evidence and audio for human review.
4. Capture enough metadata that a comparison is meaningful and reproducible later.
5. Capture runtime backend and hardware context, including CPU, Apple MPS, MLX, Core ML, ONNX, or provider-hosted execution.
6. Work as a CLI first.
7. Be extensible: a new provider or local model adapter should take well under an hour to add once its runtime is installed.

## 6. Non-Goals (v1)

The initial version will not:

- Score emotion, style, or expressiveness.
- Measure long-form stability or voice drift across many generations.
- Provide a public leaderboard or claim a universal ranking.
- Train models.
- Replace human review for regulated clinical, legal, or financial deployments.
- Ship a web UI.
- Guarantee GPU acceleration for every local model. v1 records the backend actually used and falls back honestly when a runtime does not support GPU cleanly.

These are intentionally deferred. See the roadmap.

## 7. MVP Scope

### 7.1 Adapter Interface

A common interface for local models and cloud providers. Each adapter exposes:

- Provider or runtime name.
- Model name and voice ID/name.
- Execution mode: `local` or `cloud`.
- Runtime backend: `cpu`, `mps`, `mlx`, `coreml`, `onnx`, `cuda`, `provider_hosted`, or `unknown`.
- Device label, when available.
- Streaming support flag and streaming mode.
- Native audio format and sample rate.
- A `synthesize()` method that yields audio chunks or a completed audio payload with timestamps.
- Provider-specific or runtime-specific configuration passthrough, recorded in metadata.

v1 local adapters:

- **Piper**: first adapter; CPU-first, fast, simple, good for proving the benchmark.
- **Kokoro**: second adapter; lightweight local TTS, GPU-backed execution preferred on Apple Silicon through MLX or PyTorch MPS where practical.
- **Coqui XTTS v2**: third adapter; heavier, more capable, useful for multilingual and voice-cloning-oriented comparison.

Post-local cloud adapters:

- **Cartesia**: streaming cloud TTS comparison target.
- **ElevenLabs**: streaming cloud TTS comparison target.
- **OpenAI**: cloud reference adapter, including non-streaming or HTTP-chunked behavior.

### 7.2 Latency Benchmark

The latency benchmark supports both local and cloud TTS.

For every synthesis, TTSBench records:

- **t0**: synthesis requested.
- **t1**: first response/chunk received, when applicable.
- **t2**: first decodable audio frame.
- **t3**: first playable buffer, configurable, default 250ms of decoded PCM.
- **t4**: synthesis complete.

For local non-streaming adapters, `t1`, `t2`, and `t3` may collapse to the same point or be unavailable until the model returns the full audio. The report must show `not_applicable` rather than pretending local batch inference has cloud-style network timing.

Derived metrics:

- Model load time, when the model is loaded inside the run.
- Cold synthesis time: first generation after model load.
- Warm synthesis time: later generations after the model is already loaded.
- Time to first byte/chunk (t1 - t0), when applicable.
- Time to first audio (t2 - t0), when applicable.
- **Time to first playable buffer** (t3 - t0), when streaming or incremental output is available.
- Total synthesis wall time (t4 - t0).
- Realtime factor (audio duration / wall time).
- Inter-chunk gap p50, p95, p99, when streaming chunks are available.

The headline metric depends on execution mode:

- For streaming cloud or streaming local adapters: **time to first playable buffer**.
- For local batch adapters: **warm synthesis time** and **realtime factor**.

All metrics are reported across `--repeats` runs (default 5 for local development, 10 for final comparison runs) with p50, p95, and p99 where the sample count is large enough. Reports warn when percentile estimates are based on too few samples.

### 7.3 Pronunciation Benchmark

Evaluates whether the TTS speaks production-critical text correctly.

Evaluation pipeline per item:

1. Synthesize the input text.
2. Transcribe the audio with a configurable ASR backend. Default local target: faster-whisper. For Apple Silicon, MLX Whisper or another local ASR backend can be added later.
3. For high-severity items, record the raw un-normalized ASR output when the backend supports it so hallucinated "corrections" are visible.
4. Normalize both the transcript and each expected pattern: lowercase, strip punctuation, collapse whitespace, normalize hyphens to spaces. Match each `expected_spoken_contains` pattern as a substring against the normalized transcript.
5. Record pass/fail per pattern, the normalized transcript, the raw transcript when available, and a link to the audio file for human review.

An overall item passes only if every required pattern matches. Normalization rules are documented so users can understand why "amoxicillin-clavulanate" and "amoxicillin clavulanate" both pass.

The report **always** surfaces audio + transcript + expected forms together, because ASR can silently "correct" mispronunciations of medical and technical terms back to their correct spelling. The tool is honest about this: pass/fail is treated as a signal, not a verdict, and high-severity items are flagged for human listening.

Example dataset item:

```yaml
- id: medication_dosage_001
  input: "Take Amox-Clav 625mg BID x5d."
  expected_spoken_contains:
    - "amoxicillin clavulanate"
    - "six hundred twenty five milligrams"
    - "twice daily"
    - "five days"
  category: medical_dosage
  severity: high
```

v1 ships with **one well-built healthcare dataset** of approximately 30 hand-crafted items covering appointment text, doctor names, medications and dosages, dates and times, and insurance IDs. Quality over breadth.

### 7.4 Cost and Runtime Capture

For each run, record:

- Characters sent.
- Audio duration generated.
- Execution mode: local or cloud.
- Runtime backend and device.
- Model load time, when available.
- Local API cost as zero for local adapters.
- Provider-reported billing unit, if available for cloud providers.
- Estimated cloud cost per 1k characters and per generated minute, using a configurable price table per provider.

No "cost per successful item" metric in v1. It sounds useful and isn't.

### 7.5 Reports

Two output formats:

- **CSV**: one row per synthesis with all timing points, costs, runtime metadata, and pronunciation results.
- **Markdown**: human-readable summary with latency tables, pronunciation failure list (with audio links and transcripts), runtime/backend summary, cost summary, and full run metadata.

Reports embed relative paths to audio files so a run directory is self-contained and shareable.

## 8. Repository Structure

```text
ttsbench/
  adapters/
    base.py
    piper.py
    kokoro.py
    coqui_xtts.py
    cartesia.py
    elevenlabs.py
    openai.py

  benchmarks/
    latency.py
    pronunciation.py
    cost.py

  evaluators/
    asr_roundtrip.py
    expected_form_match.py

  datasets/
    healthcare.yaml

  reports/
    markdown_report.py
    csv_report.py

  cli.py
  config.py
  runtime.py
  schemas.py
  utils.py

runs/
  example_run/
    audio/
    metadata.jsonl
    metrics.csv
    pronunciation_results.jsonl
    summary.txt
    report.md
```

`summary.txt` is a plain-text aggregate written by `ttsbench run` (p50/p95/p99 where sample counts allow). `report.md` is the rendered human-readable report, generated by `ttsbench report` from the run artifacts.

## 9. Data Model

Each synthesis result is a single JSONL record:

```json
{
  "run_id": "2026-05-28-kokoro-healthcare-mps",
  "provider": "kokoro",
  "execution_mode": "local",
  "model": "kokoro",
  "voice": "default_voice",
  "runtime_backend": "mps",
  "device": "Apple M3 Pro GPU",
  "text": "Your appointment with Dr. Iyer is on 05/06 at 3:30 PM.",
  "dataset": "healthcare",
  "dataset_item_id": "healthcare_appt_001",
  "benchmark_suite": "latency",
  "model_load_ms": 1250,
  "cold_start": false,
  "t0_request_sent_ns": 12345678900,
  "t1_first_chunk_ns": null,
  "t2_first_audio_ns": null,
  "t3_first_playable_ns": null,
  "t4_synthesis_complete_ns": 12346498900,
  "audio_duration_ms": 3120,
  "wall_time_ms": 820,
  "ttfb_ms": null,
  "ttfa_ms": null,
  "ttfp_ms": null,
  "realtime_factor": 3.8,
  "sample_rate": 24000,
  "format": "pcm_s16le",
  "streaming": false,
  "streaming_mode": "none",
  "region": "local",
  "repeat_index": 2,
  "audio_path": "audio/healthcare_appt_001_r1.wav",
  "characters_sent": 58,
  "estimated_cost_usd": 0.0,
  "provider_params": {"language": "en"}
}
```

Timing fields that do not apply to a given execution mode (for example, `t1`–`t3`, `ttfb_ms`, `ttfa_ms`, `ttfp_ms` for a local batch adapter) are stored as JSON `null`. Reports render these as `not_applicable` rather than as `0` or a blank, so the distinction between "measured zero" and "this metric does not exist for this adapter" stays explicit.

`cold_start` is `true` for the first synthesis after the model loads (`repeat_index: 1`) and `false` for every warm repeat after it. The record above is the second repeat, so it is a warm run.

API keys never appear in this record.

## 10. Healthcare Dataset (v1)

Approximately 30 items across:

- Appointment confirmations with doctor names (including non-Anglo names: Iyer, Nguyen, Okonkwo, Rodriguez).
- Medication names, abbreviated dosages, and frequency abbreviations.
- Dates in ambiguous formats (05/06 spoken as "May sixth" vs "June fifth").
- Times in 12-hour and 24-hour forms.
- Insurance and member IDs (mixed alphanumeric).
- A small set of code-switched items (Hinglish, Spanglish) flagged separately for advisory pass/fail.

Each item has `input`, `expected_spoken_contains` (a list of acceptable substring patterns, where any one match counts as a pass for that pattern), `category`, and `severity` (low, medium, high).

## 11. CLI

### 11.1 Run

Local examples:

```bash
ttsbench run \
  --provider piper \
  --suite latency,pronunciation \
  --dataset datasets/healthcare.yaml \
  --repeats 5 \
  --device cpu \
  --output runs/piper_healthcare
```

```bash
ttsbench run \
  --provider kokoro \
  --suite latency,pronunciation \
  --dataset datasets/healthcare.yaml \
  --repeats 5 \
  --device mps \
  --output runs/kokoro_healthcare_mps
```

Cloud example, added after the local path:

```bash
ttsbench run \
  --provider cartesia \
  --model sonic \
  --voice <voice_id> \
  --suite latency,pronunciation \
  --dataset datasets/healthcare.yaml \
  --repeats 10 \
  --region us-east \
  --output runs/cartesia_healthcare
```

Flags:

- `--device` requests a local runtime target such as `cpu`, `mps`, or `mlx`. The report records the backend actually used.
- `--region` is a free-form label recorded in metadata. For local runs it defaults to `local`; for cloud runs it defaults to `unknown`. It is required for meaningful cross-region comparison.
- `--fail-on` accepts threshold expressions and exits non-zero if violated, for CI use. Example: `--fail-on warm_wall_time_p95_ms>400 --fail-on pronunciation_pass_rate<0.95`. Multiple `--fail-on` flags AND together. Valid threshold keys are listed below; a key whose underlying metric is `not_applicable` for the run (for example, a streaming key on a local batch adapter) causes `--fail-on` to error rather than silently pass.

Valid `--fail-on` keys:

| Key | Meaning |
|---|---|
| `model_load_ms` | Model load time. |
| `cold_wall_time_ms` | Cold synthesis wall time (first generation after load). |
| `warm_wall_time_p50_ms`, `warm_wall_time_p95_ms`, `warm_wall_time_p99_ms` | Warm synthesis wall time percentiles. |
| `total_wall_time_p50_ms`, `total_wall_time_p95_ms`, `total_wall_time_p99_ms` | Total synthesis wall time percentiles. |
| `realtime_factor_p50`, `realtime_factor_p95` | Realtime factor percentiles. |
| `ttfb_p50_ms`, `ttfb_p95_ms` | Time to first byte/chunk percentiles (streaming only). |
| `ttfa_p50_ms`, `ttfa_p95_ms` | Time to first audio percentiles (streaming only). |
| `ttfp_p50_ms`, `ttfp_p95_ms` | Time to first playable buffer percentiles (streaming only). |
| `inter_chunk_gap_p95_ms`, `inter_chunk_gap_p99_ms` | Inter-chunk gap percentiles (streaming only). |
| `pronunciation_pass_rate` | Fraction of items that passed all required patterns. |
| `high_severity_pass_rate` | Pass rate restricted to high-severity items. |
| `estimated_cost_usd` | Estimated cost for the run. |
- `--dry-run` estimates cloud cost from the dataset without making API calls. For local providers it reports expected generated item counts and zero API cost.

### 11.2 Compare

```bash
ttsbench compare \
  --runs runs/piper_healthcare runs/kokoro_healthcare_mps \
  --output reports/local_comparison.md
```

### 11.3 Validate Dataset

```bash
ttsbench validate datasets/healthcare.yaml
```

### 11.4 Generate Report

```bash
ttsbench report --run runs/kokoro_healthcare_mps --format md
```

### 11.5 Inspect Runtime

```bash
ttsbench runtime
```

This command prints detected CPU, memory, platform, and available acceleration backends such as Apple MPS, MLX, Core ML, ONNX Runtime, or CUDA.

### 11.6 Synthesize (dev-only)

```bash
ttsbench synthesize --provider piper --text "Hello world" --output hello.wav
```

A one-off command for calling a single adapter and writing a WAV, used while building and debugging adapters. It is not part of the primary benchmarking workflow and is documented here only so the command surface is complete.

## 12. Technical Considerations

### 12.1 Local Runtime Honesty

The report records the requested device and the actual backend used. If a model is requested with `--device mps` but falls back to CPU, the report must show that fallback explicitly.

The initial local development target is an Apple Silicon MacBook Pro-class machine. On this hardware, CUDA is not available; GPU acceleration means Apple Metal-backed runtimes such as PyTorch MPS, MLX, Core ML, or another Apple-compatible backend. Piper is expected to start as a CPU baseline. Kokoro is the first practical GPU-targeted local adapter. Coqui XTTS v2 is supported later with honest CPU/MPS fallback metadata.

### 12.2 Streaming Timing Accuracy

For streaming adapters, chunk receive timestamps are captured with `time.perf_counter_ns()` at the lowest layer of each adapter's streaming code, such as inside a websocket recv loop or HTTP stream iterator.

For local batch adapters, the report does not fabricate network-like metrics. It records load time, cold synthesis time, warm synthesis time, total wall time, and realtime factor.

### 12.3 Audio Format Normalization

Internal canonical format is 16-bit PCM WAV at the model or provider's native sample rate. Resampling is recorded but not applied by default. Decoding compressed formats (MP3, Opus) is done with `pyav` or `soundfile`, and decode time is tracked separately when it would otherwise pollute first-playable timing.

### 12.4 ASR Round-Trip Honesty

The pronunciation report makes three things visible per item: the input text, the ASR transcript, and the audio file. Pass/fail is a signal. Whisper-like ASR can silently correct misspoken medical terms back to their correct spelling, so high-severity failures and high-severity passes should both be flagged for spot-checking.

### 12.5 Fairness in Comparison

The tool does not claim a universal ranking. Comparison reports include a "Configuration" section listing provider/model, voice, execution mode, runtime backend, device, region, audio format, sample rate, streaming mode, and all provider-specific parameters for each run. Without this, comparisons are not meaningful. The README states this plainly.

### 12.6 Concurrency

Latency-suite syntheses run **sequentially** to avoid local resource contention or network contention skewing timing measurements. Pronunciation-suite syntheses also run sequentially for consistency, but ASR runs in a configurable pool (`--asr-workers`, default 2) when the backend supports it. ASR results are cached by audio-file SHA-256 so a re-run of just the matcher or a re-render of the report does not re-transcribe.

### 12.7 Secrets

Local adapters do not require API keys.

Cloud API keys are loaded from environment variables only:

```bash
export CARTESIA_API_KEY=...
export ELEVENLABS_API_KEY=...
export OPENAI_API_KEY=...
```

No secrets are written to metadata, reports, or logs.

## 13. Success Criteria for v1

The v1 ships when:

- A user can run `ttsbench run` against Piper locally and get a complete report directory without creating a paid provider account.
- A user can run at least one GPU-capable local adapter on Apple Silicon when the required runtime is installed, with the actual backend recorded in metadata.
- The latency report distinguishes local batch metrics from streaming metrics and does not fabricate cloud-style numbers for local-only generation.
- The pronunciation report lists every failure with input, transcript, expected forms, and a working audio link.
- A comparison report can compare Piper, Kokoro, and Coqui XTTS v2 on the healthcare dataset.
- A new adapter can be added by implementing the base class in under an hour.
- The README contains a real local comparison run with embedded audio samples and screenshots of the report.

## 14. Roadmap (deferred from v1)

These are intentionally out of scope for v1 and live in `ROADMAP.md`:

- **Cloud provider adapters**: Cartesia, ElevenLabs, and OpenAI comparison runs after the local path is solid.
- **Stability benchmark**: variance across repeated generations.
- **Emotion and style benchmark**: human rating UI and acoustic features. The product position is that this requires a human-in-the-loop workflow to be honest, so it ships as a separate companion later.
- **Multilingual and code-switching suite**: needs careful dataset design beyond the advisory v1 items.
- **Streamlit UI** for non-CLI users.
- **Customer-specific pronunciation dictionaries** and script normalization benchmarking.
- **Voice-agent-context dataset**: TTS evaluated on the kind of text that comes out of an LLM mid-conversation, with barge-in and partial-generation patterns. This is the differentiated angle for real-time voice agents specifically and is the most interesting future addition.

## 15. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Local GPU support varies by model/runtime | Record requested device and actual backend; start with Piper CPU, then add Kokoro GPU path, then XTTS |
| ASR round-trip is imperfect | Always surface audio + transcript together; flag high-severity items for human review |
| Provider or model comparisons may be unfair | Record full configuration, runtime, hardware, and dataset metadata; state explicitly that the tool is for regression and workload fit, not universal ranking |
| Streaming timing differs between local and cloud adapters | Report local batch metrics and streaming metrics separately; use `not_applicable` for unsupported timing fields |
| Cloud API costs add up during dataset runs | Local providers are the default path; `--dry-run` estimates cost before cloud runs |
| Scope creep into emotion and stability | These are out of v1 by design; ROADMAP makes the deferral visible |

## 16. Build Order

Phases match the build plan (`ttsbench_build_plan.md`), numbered 0–11.

- **Phase 0**: Repo skeleton and scaffolding.
- **Phase 1**: Adapter base class, runtime inspection command, and schema fields for execution mode, backend, and device.
- **Phase 2**: Piper adapter and one-off local synthesis command.
- **Phase 3**: Local latency benchmark with batch-safe metrics and CSV output.
- **Phase 4**: Healthcare dataset (30 items).
- **Phase 5**: Pronunciation benchmark with ASR round-trip and expected-form matching.
- **Phase 6**: Markdown report with embedded audio links and runtime/backend summary.
- **Phase 7**: Kokoro adapter with Apple Silicon GPU path where practical.
- **Phase 8**: Coqui XTTS v2 adapter with honest CPU/MPS fallback metadata.
- **Phase 9**: Compare command for Piper vs Kokoro vs XTTS.
- **Phase 10**: README with a real local example run, screenshots, and audio samples.
- **Phase 11**: Cloud provider adapters: Cartesia, ElevenLabs, OpenAI.
