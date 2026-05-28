"""Phase 5 (revised): phoneme matcher, recognizer caching, pronunciation, run CLI.

The phoneme recognizer and ASR are faked (canned phoneme/transcript output) so
these tests need neither torch/transformers nor faster-whisper. The eSpeak
reference is exercised for real (it ships with Piper and is lightweight).
"""

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from ttsbench.adapters.base import SynthesisResult, TTSAdapter
from ttsbench.benchmarks.pronunciation import benchmark_pronunciation
from ttsbench.cli import app
from ttsbench.evaluators.phoneme_match import (
    EspeakReference,
    approx_substring_per,
    normalize_phonemes,
)
from ttsbench.evaluators.phoneme_recognizer import CachingPhonemeRecognizer, PhonemeRecognizer
from ttsbench.schemas import (
    AudioFormat,
    Dataset,
    DatasetItem,
    ExecutionMode,
    RuntimeBackend,
    Severity,
    StreamingMode,
)

runner = CliRunner()


# --- phoneme normalization + distance --------------------------------------


def test_normalize_explodes_diphthongs_and_strips_stress():
    # Recognizer emits "aɪ"/"oː" as single tokens; espeak emits per-character.
    assert normalize_phonemes(["aɪ", "oː"]) == ["a", "ɪ", "o"]
    assert normalize_phonemes(["ˈ", "t", "ʃ", "ˌ"]) == ["t", "ʃ"]


def test_approx_substring_per():
    produced = list("abcdefg")
    assert approx_substring_per(produced, list("cde")) == 0.0  # exact substring
    assert approx_substring_per(produced, list("cxe")) == 1 / 3  # one mismatch
    assert approx_substring_per(produced, []) == 0.0
    assert approx_substring_per([], list("ab")) == 1.0


def test_espeak_reference_real():
    ref = EspeakReference()
    # "Cho" -> /tʃoʊ/; digits and words phonemize the same way.
    assert ref.phonemes("Cho")[:2] == ["t", "ʃ"]
    assert ref.phonemes("500") == ref.phonemes("five hundred")


# --- recognizer caching ----------------------------------------------------


class _CountingRecognizer(PhonemeRecognizer):
    def __init__(self, phonemes: list[str]) -> None:
        self.calls = 0
        self._phonemes = phonemes

    @property
    def name(self) -> str:
        return "counting"

    def recognize(self, wav_path: Any) -> list[str]:
        self.calls += 1
        return self._phonemes


def test_caching_recognizer(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake")
    backend = _CountingRecognizer(["t", "ʃ", "o", "ʊ"])
    cache = tmp_path / "ph.json"

    rec1 = CachingPhonemeRecognizer(backend, cache)
    assert rec1.recognize(wav) == ["t", "ʃ", "o", "ʊ"]
    assert rec1.recognize(wav) == ["t", "ʃ", "o", "ʊ"]
    assert backend.calls == 1

    rec2 = CachingPhonemeRecognizer(backend, cache)  # reloads persisted cache
    assert rec2.recognize(wav) == ["t", "ʃ", "o", "ʊ"]
    assert backend.calls == 1


# --- pronunciation benchmark ----------------------------------------------


class _StubAdapter(TTSAdapter):
    provider = "stub"
    execution_mode = ExecutionMode.LOCAL
    model = "stub"
    voice = "default"
    runtime_backend = RuntimeBackend.CPU
    requested_device = "cpu"
    actual_device = "cpu"
    sample_rate = 22050
    audio_format = AudioFormat.PCM_S16LE
    streaming = False
    streaming_mode = StreamingMode.NONE

    def synthesize(self, text: str, voice: str | None = None, **params: Any) -> SynthesisResult:
        return SynthesisResult(
            audio=b"\x00\x01" * 100,
            sample_rate=self.sample_rate,
            audio_format=self.audio_format,
            duration_ms=10.0,
        )


class _MapRecognizer(PhonemeRecognizer):
    """Returns produced phonemes keyed by item id (parsed from the wav filename)."""

    def __init__(self, by_item: dict[str, list[str]]) -> None:
        self._by_item = by_item

    @property
    def name(self) -> str:
        return "map"

    def recognize(self, wav_path: Any) -> list[str]:
        item_id = Path(wav_path).stem.rsplit("_r", 1)[0]
        return self._by_item.get(item_id, [])


def _dataset() -> Dataset:
    return Dataset(
        name="t",
        items=[
            DatasetItem(
                id="good",
                input="Cho",
                category="doctor_name",
                severity=Severity.HIGH,
                expected_spoken_contains=["cho"],
            ),
            DatasetItem(
                id="bad",
                input="Cho",
                category="doctor_name",
                severity=Severity.HIGH,
                expected_spoken_contains=["cho"],
            ),
        ],
    )


def test_pronunciation_phoneme_pass_and_fail(tmp_path):
    ref = EspeakReference()
    cho = ref.phonemes("cho")  # ['t','ʃ','o','ʊ']
    # "good" produces the exact phonemes; "bad" drops everything -> high PER.
    recognizer = _MapRecognizer({"good": cho, "bad": ["m", "n", "p"]})

    records, results = benchmark_pronunciation(
        _StubAdapter(),
        _dataset(),
        repeats=1,
        run_id="r1",
        recognizer=recognizer,
        run_dir=tmp_path,
        audio_dir=tmp_path / "audio",
        reference=ref,
        tolerance=0.30,
        asr=None,
        workers=2,
    )

    assert len(records) == 2
    by_id = {r.item_id: r for r in results}
    assert by_id["good"].passed is True
    assert by_id["good"].pattern_matches[0].per == 0.0
    assert by_id["bad"].passed is False
    assert by_id["bad"].pattern_matches[0].per > 0.30
    # Pass/fail is phoneme-driven; ASR was not supplied so transcript is absent.
    assert by_id["good"].transcript is None
    assert by_id["good"].produced_phonemes == " ".join(cho)


# --- run CLI ---------------------------------------------------------------


def test_run_pronunciation_writes_results(tmp_path, monkeypatch):
    out = tmp_path / "run"
    ds_path = tmp_path / "ds.yaml"
    ds_path.write_text(
        "name: t\nitems:\n"
        "  - id: good\n    input: Cho\n    category: doctor_name\n"
        "    severity: high\n    expected_spoken_contains: [cho]\n"
    )

    from ttsbench.adapters.piper import PiperAdapter

    monkeypatch.setattr(PiperAdapter, "_load", lambda self: None)
    monkeypatch.setattr(
        PiperAdapter,
        "synthesize",
        lambda self, text, voice=None, **p: SynthesisResult(
            audio=b"\x00\x01" * 100,
            sample_rate=22050,
            audio_format=AudioFormat.PCM_S16LE,
            duration_ms=10.0,
        ),
    )

    cho = EspeakReference().phonemes("cho")

    class _FixedRecognizer(PhonemeRecognizer):
        name = "fixed"

        def recognize(self, wav_path: Any) -> list[str]:
            return cho

    monkeypatch.setattr(
        "ttsbench.evaluators.phoneme_recognizer.Wav2Vec2PhonemeRecognizer",
        lambda *a, **k: _FixedRecognizer(),
    )

    result = runner.invoke(
        app,
        [
            "run",
            "--provider",
            "piper",
            "--suite",
            "pronunciation",
            "--dataset",
            str(ds_path),
            "--repeats",
            "2",
            "--no-asr",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "pronunciation_results.jsonl").exists()

    lines = (out / "pronunciation_results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["passed"] is True
    assert parsed["pattern_matches"][0]["per"] == 0.0
    assert "Pronunciation summary" in (out / "summary.txt").read_text()


def test_pronunciation_suite_requires_dataset(tmp_path):
    result = runner.invoke(
        app, ["run", "--suite", "pronunciation", "--text", "hi", "--output", str(tmp_path / "x")]
    )
    assert result.exit_code != 0
