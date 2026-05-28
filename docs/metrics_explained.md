# TTSBench Metrics Explained

This document explains the metrics TTSBench measures in simple terms: what each metric means, where it applies in a voice AI pipeline, and why it helps when building with a text-to-speech model.

## Voice AI Pipeline Context

A typical voice AI agent has this shape:

```text
User speaks
 -> Speech-to-text
 -> LLM or dialogue logic
 -> Text-to-speech
 -> Audio playback to user
```

TTSBench focuses on the final part:

```text
Text response
 -> local TTS model or cloud TTS provider
 -> generated audio or audio stream
 -> playback
```

The metrics are about whether that final step is fast, correct, affordable, practical on the target hardware, and safe to change.

## Latency Metrics

Latency metrics are slightly different for local models and cloud providers.

For local models, the useful question is usually:

```text
Can this machine generate speech fast enough?
```

For cloud providers, the useful question is usually:

```text
How quickly does usable audio arrive over the API?
```

TTSBench should report unsupported metrics as unavailable instead of pretending every adapter behaves like a streaming cloud API.

### Model Load Time

Simple meaning: how long it takes to load the TTS model into memory.

Where it applies: local models such as Piper, Kokoro, and Coqui XTTS v2.

Why it helps: a model may be fast once loaded but slow to start. This matters for apps that spin up workers on demand, run in CI, or load models only when needed.

### Cold Synthesis Time

Simple meaning: how long the first generation takes after the model is loaded.

Where it applies: local TTS models.

Why it helps: the first run can be slower because the runtime initializes kernels, allocates memory, or warms caches. Measuring this separately avoids confusing startup cost with normal steady-state performance.

### Warm Synthesis Time

Simple meaning: how long later generations take after the model is already ready.

Where it applies: local TTS models.

Why it helps: this is the main latency metric for local batch TTS. It shows what users should expect once the model is running.

### Time to First Byte

Simple meaning: how long it takes before the TTS provider sends anything back.

Where it applies: between sending text to a cloud TTS API and receiving the first response from the provider.

Why it helps: it shows whether the provider or API is responsive. By itself, it does not prove the user can hear anything yet, because the first bytes may not be playable audio.

For local batch models, this metric usually does not apply.

### Time to First Audio

Simple meaning: how long it takes until the first actual audio data is available.

Where it applies: after a streaming provider or streaming local runtime starts responding, when the app can identify usable audio data.

Why it helps: some providers may send headers, metadata, or setup messages before real audio. This metric shows when speech output actually starts to exist.

### Time to First Playable Buffer

Simple meaning: how long it takes until there is enough audio to start playback smoothly.

Where it applies: right before the app can begin playing audio to the user.

Why it helps: this is the most important latency metric for real-time voice agents. Users do not care when bytes arrive; they care when the agent starts speaking.

For local models that only return a completed WAV, this metric may be unavailable. In that case, warm synthesis time and realtime factor are more useful.

### Total Synthesis Wall Time

Simple meaning: how long the whole TTS generation takes from request start to audio completion.

Where it applies: across the full TTS request or local synthesis call.

Why it helps: it shows how long the model or provider takes to finish generating the complete response. This is useful for longer messages, saved audio, and workflows where the full output must be ready before the next step.

### Realtime Factor

Simple meaning: whether audio is generated faster or slower than it would be played.

For example, if 10 seconds of audio is generated in 5 seconds, generation is faster than real time.

Where it applies: during or after synthesis, for both local and cloud TTS.

Why it helps: if a TTS model cannot generate faster than playback speed, a live voice agent may stall while speaking.

### Inter-Chunk Gap

Simple meaning: how much delay there is between pieces of streamed audio.

Where it applies: during streaming playback.

Why it helps: a provider may start quickly but then pause between chunks. That can make speech sound choppy even if the first audio arrived quickly.

For local models that do not stream chunks, this metric does not apply.

## Pronunciation Metrics

### Expected-Form Match

Simple meaning: whether the generated speech contains the words or phrases expected for a test item.

Example input:

```text
Take Amox-Clav 625mg BID x5d.
```

Expected spoken forms might include:

```text
amoxicillin clavulanate
six hundred twenty five milligrams
twice daily
five days
```

Where it applies: after TTS generation. The generated audio is transcribed with ASR, then the transcript is compared with the expected spoken forms.

Why it helps: TTS models often struggle with names, medicines, abbreviations, IDs, dates, and domain-specific terms. This metric checks the kinds of words that can break a real production use case.

### Pronunciation Pass or Fail Per Item

Simple meaning: whether one test sentence passed all required expected-form checks.

Where it applies: after each generated audio sample is evaluated.

Why it helps: it gives a clear list of which specific inputs failed instead of only showing an average score.

### Pronunciation Pass Rate

Simple meaning: the percentage of test items that passed.

Where it applies: across a whole dataset or customer script set.

Why it helps: it gives a quick summary of whether a provider, model, or voice is reliable enough on the target workload.

Important caveat: this should be treated as a signal, not absolute truth, because ASR can sometimes correct or hide TTS mistakes in the transcript.

### High-Severity Failures

Simple meaning: failures on items that matter most.

For example, a medication dosage or insurance ID is higher severity than a generic greeting.

Where it applies: in the pronunciation report.

Why it helps: not all failures are equal. One high-risk pronunciation failure can matter more than several low-risk wording failures.

### Raw Transcript vs Normalized Transcript

Simple meaning: comparing what the ASR originally heard with the cleaned-up transcript used for matching.

Where it applies: during pronunciation evaluation.

Why it helps: ASR systems sometimes hide TTS mistakes by normalizing or correcting the transcript. Keeping the raw transcript makes the benchmark more honest, especially for medical, legal, financial, or technical speech.

## Cost Metrics

### Characters Sent

Simple meaning: how much text was sent to the TTS model or provider.

Where it applies: before or during TTS billing.

Why it helps: many cloud TTS providers charge by character count, so this helps estimate cost at production volume. For local models, it still helps normalize workload size across runs.

### Audio Duration Generated

Simple meaning: how many seconds or minutes of speech were produced.

Where it applies: after synthesis.

Why it helps: some cloud pricing models depend on generated audio length. Duration also helps explain runtime performance and user experience for local models.

## Runtime Metrics

### Execution Mode

Simple meaning: whether the run used a local model or a cloud provider.

Where it applies: across the whole run.

Why it helps: local and cloud results answer different questions. Local runs measure hardware/runtime practicality. Cloud runs measure provider/API behavior and external service cost.

### Runtime Backend

Simple meaning: what actually ran the model.

Examples:

```text
cpu
mps
mlx
coreml
onnx
cuda
provider_hosted
```

Where it applies: across the whole run, especially for local models.

Why it helps: the same model can behave very differently on CPU versus Apple MPS, MLX, Core ML, or CUDA. Recording the backend keeps comparisons honest.

### Requested Device vs Actual Device

Simple meaning: the device the user asked for compared with the device that was actually used.

Where it applies: local GPU-capable models.

Why it helps: a user might request GPU acceleration, but the runtime may fall back to CPU because an operation is unsupported. The benchmark should make that visible.

## Cloud Billing Metrics

### Provider-Reported Billing Unit

Simple meaning: the billing information returned by the provider, when available.

Where it applies: after the provider response.

Why it helps: estimated cost and actual billed cost can differ. Capturing provider-reported billing data makes the report more trustworthy.

### Estimated Cost

Simple meaning: the approximate paid API cost of running the workload.

Where it applies: per run, per dataset, or projected across production volume.

Why it helps: a TTS model can be fast and accurate but too expensive. Cost is part of deciding whether a provider or model is practical to use.

For local models, API cost is zero. Local runs can still have hardware cost, battery use, and operational cost, but v1 does not try to price those.

## Metadata

Metadata is not a performance metric, but it makes the benchmark meaningful.

Examples include:

```text
provider
execution mode
model
voice
runtime backend
requested device
actual device
region
sample rate
audio format
streaming mode
provider parameters
dataset
repeat count
timestamp
```

Simple meaning: the exact setup that produced the result.

Where it applies: across the whole benchmark run.

Why it helps: without metadata, two results cannot be compared honestly. A different voice, backend, device, region, format, or provider setting can change the outcome.

## Summary

TTSBench metrics answer practical questions for building with TTS models:

```text
Does the voice start speaking quickly?
Does it keep streaming smoothly?
Does it say important words correctly?
Does it fail on risky terms?
Can this laptop run the model fast enough?
Did the run actually use GPU or fall back to CPU?
How much will it cost?
Can the result be reproduced later?
Did a change make things worse?
```
