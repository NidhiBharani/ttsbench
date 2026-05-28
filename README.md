# TTSBench

**TTSBench is an open-source, local-first benchmark toolkit for text-to-speech
systems used in real-time voice products.**

Most TTS comparisons answer the wrong question: "Which demo sounds best?"
Production teams need sharper answers:

- How fast does audio become usable, not just how fast does an API return?
- Can this model run quickly enough on the hardware we actually deploy?
- Does it pronounce names, dosages, dates, IDs, SKUs, and domain terms correctly?
- Did a model, voice, runtime, or text-normalization change silently regress output?
- What does this choice cost locally versus through a paid provider?

TTSBench packages those checks into repeatable CLI benchmarks, self-contained run
artifacts, and reports that keep the audio, transcript, timing, runtime, and cost
evidence together.

## What TTSBench Is For

TTSBench is built for:

- Voice AI engineers integrating TTS into real-time agents.
- Applied AI teams comparing local and cloud TTS options.
- Forward-deployed engineers testing customer-specific scripts.
- Teams running regression checks before promoting a new model, voice, runtime
  backend, or provider.
- Startup teams choosing a voice stack before committing to paid providers.

It is **not** a universal "best TTS" leaderboard. It helps you decide whether a
specific TTS setup is fit for a specific workload.

## Project Status

TTSBench is in early development.

The repository currently contains the Phase 0 scaffolding: package layout, CLI
entry point, adapter modules, benchmark modules, report modules, and development
tooling. The public CLI surface exists, but benchmark commands still print
`not implemented yet` while the first usable benchmark path is being built.

The first working release target is a local Piper benchmark that can generate a
complete report directory without requiring any paid provider account.

## Target Workflow

The intended workflow is simple: run the same dataset against multiple TTS
systems, then compare the outputs.

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

```bash
ttsbench compare \
  --runs runs/piper_healthcare runs/kokoro_healthcare_mps \
  --output reports/local_comparison.md
```

A run directory is designed to be self-contained:

```text
runs/example_run/
  audio/
  metadata.jsonl
  metrics.csv
  pronunciation_results.jsonl
  summary.txt
  report.md
```

Reports will link to generated audio files and show the exact configuration used
for each run: provider, model, voice, execution mode, runtime backend, device,
region, sample rate, streaming mode, and provider-specific parameters.

## What It Measures

### Latency

TTSBench distinguishes local batch inference from streaming cloud behavior.

For streaming adapters, it tracks timing points such as first chunk, first audio,
first playable buffer, synthesis complete, and inter-chunk gaps.

For local non-streaming adapters, it reports the metrics that actually apply:
model load time, cold synthesis time, warm synthesis time, total wall time, and
realtime factor. Metrics that do not apply are rendered as `not_applicable`, not
as zeros.

### Pronunciation Reliability

TTSBench focuses on text that often breaks production voice agents:

- Doctor and customer names.
- Medication names, dosages, and frequency abbreviations.
- Ambiguous dates and times.
- Insurance IDs, member IDs, SKUs, and mixed alphanumeric strings.
- Domain-specific phrases from real customer workflows.

The pronunciation benchmark synthesizes each item, transcribes the generated
audio with an ASR backend, matches expected spoken forms, and reports the input,
expected forms, transcript, pass/fail signal, severity, and audio path together.

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

ASR round-trip evaluation is treated as a useful signal, not a final verdict.
The report keeps audio and transcript evidence visible so high-severity items can
be reviewed by a human.

### Runtime And Cost

Each run records execution mode, requested device, actual backend, hardware
context, generated audio duration, characters sent, and estimated provider cost
where applicable. Local adapters report zero API cost.

## Planned Adapters

TTSBench starts local-first so a developer can benchmark on a laptop before
spending money on provider APIs.

| Adapter | Role | Status |
|---|---|---|
| Piper | Fast CPU-friendly local baseline | Planned first usable adapter |
| Kokoro | Lightweight local quality baseline, with Apple Silicon acceleration where practical | Planned |
| Coqui XTTS v2 | Heavier multilingual and voice-cloning-capable local reference | Planned |
| Cartesia | Streaming cloud comparison target | Deferred until local path works |
| ElevenLabs | Streaming cloud comparison target | Deferred until local path works |
| OpenAI | Cloud reference adapter | Deferred until local path works |

## Design Principles

- **Local-first:** useful without a paid provider account.
- **Workload-specific:** compare systems on the text your product actually says.
- **Runtime-honest:** record the backend that actually ran, including CPU
  fallback.
- **Report evidence, not vibes:** keep timing, transcript, audio, and config
  together.
- **No fake metrics:** unsupported timing fields are explicit
  `not_applicable` values.
- **Extensible:** a new provider or model adapter should be straightforward once
  its runtime is installed.

## Install For Development

TTSBench requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ttsbench --help
```

Or use the Makefile:

```bash
make install
make test
make lint
make format
```

## Current CLI Surface

These commands are scaffolded:

```bash
ttsbench run       # run a benchmark suite against a provider
ttsbench compare   # compare two or more run directories
ttsbench validate  # validate a dataset file
ttsbench report    # generate a report from an existing run directory
ttsbench runtime   # inspect CPU, memory, platform, and acceleration backends
```

They are intentionally stubs until the implementation phases land.

## Documentation

- [Project overview](docs/README.md)
- [Product spec](docs/ttsbench_prd_v1.md)
- [Build plan](docs/ttsbench_build_plan.md)
- [Metrics explained](docs/metrics_explained.md)

## Contributing

The most useful contributions are the ones that preserve benchmark honesty:

- Implement a local adapter without hiding backend fallback.
- Add benchmark logic that keeps local and streaming metrics distinct.
- Improve dataset validation and report clarity.
- Add pronunciation items that reflect real production failure modes.
- Tighten documentation around metrics, caveats, and reproducibility.

Start with the [build plan](docs/ttsbench_build_plan.md) to see the intended
implementation order.
