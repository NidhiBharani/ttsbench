"""Pronunciation benchmark with phoneme round-trip. Phase 5 (revised).

Synthesis runs sequentially (shared with the latency benchmark, tagged
``suite="pronunciation"``). Each clip is then evaluated by comparing the phonemes
recognized from the audio against the eSpeak phonemes of each expected spoken
form. Pass/fail is phoneme-distance based (robust to digit/word spelling and to
ASR language-model "corrections"). An optional word-ASR transcript is attached as
advisory, human-readable context and does NOT affect pass/fail.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ttsbench.adapters.base import TTSAdapter
from ttsbench.benchmarks.latency import BenchmarkItem, benchmark_latency
from ttsbench.evaluators.asr_roundtrip import ASRBackend
from ttsbench.evaluators.expected_form_match import normalize
from ttsbench.evaluators.phoneme_match import EspeakReference, approx_substring_per
from ttsbench.evaluators.phoneme_recognizer import PhonemeRecognizer
from ttsbench.schemas import (
    Dataset,
    PatternMatch,
    PronunciationResult,
    Severity,
    SynthesisRecord,
)

DEFAULT_PHONEME_TOLERANCE = 0.30


def benchmark_pronunciation(
    adapter: TTSAdapter,
    dataset: Dataset,
    repeats: int,
    run_id: str,
    recognizer: PhonemeRecognizer,
    *,
    run_dir: Path,
    audio_dir: Path,
    reference: EspeakReference | None = None,
    tolerance: float = DEFAULT_PHONEME_TOLERANCE,
    asr: ASRBackend | None = None,
    workers: int = 2,
) -> tuple[list[SynthesisRecord], list[PronunciationResult]]:
    """Synthesize every item, recognize phonemes, and score against expected forms."""
    reference = reference or EspeakReference()

    items = [
        BenchmarkItem(id=item.id, text=item.input, dataset_item_id=item.id)
        for item in dataset.items
    ]
    records = benchmark_latency(
        adapter,
        items,
        repeats,
        run_id,
        suite="pronunciation",
        dataset=dataset.name,
        audio_dir=audio_dir,
    )

    # Pre-load heavy models once so the worker pool does not race on first use.
    recognizer.load()
    if asr is not None:
        asr.load()

    by_id = {item.id: item for item in dataset.items}
    reference_phonemes = {
        item.id: [reference.phonemes(p) for p in item.expected_spoken_contains]
        for item in dataset.items
    }

    def evaluate(record: SynthesisRecord) -> PronunciationResult:
        item = by_id[record.dataset_item_id]
        wav_path = run_dir / record.audio_path if record.audio_path else None

        produced = recognizer.recognize(wav_path) if wav_path else []
        matches: list[PatternMatch] = []
        for pattern, ref_phones in zip(
            item.expected_spoken_contains, reference_phonemes[item.id], strict=True
        ):
            per = approx_substring_per(produced, ref_phones)
            matches.append(PatternMatch(pattern=pattern, matched=per <= tolerance, per=per))

        transcript = raw = None
        if asr is not None and wav_path is not None:
            asr_text = asr.transcribe(wav_path).text
            transcript = normalize(asr_text)
            raw = asr_text if item.severity is Severity.HIGH else None

        return PronunciationResult(
            run_id=run_id,
            dataset=dataset.name,
            item_id=item.id,
            repeat_index=record.repeat_index,
            input=item.input,
            category=item.category,
            severity=item.severity,
            advisory=item.advisory,
            audio_path=record.audio_path,
            pattern_matches=matches,
            passed=all(m.matched for m in matches),
            produced_phonemes=" ".join(produced),
            transcript=transcript,
            raw_transcript=raw,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(evaluate, records))

    return records, results
