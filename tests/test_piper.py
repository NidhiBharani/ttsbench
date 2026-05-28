"""Phase 2: Piper adapter and the `synthesize` CLI command.

Piper itself is mocked: a fake voice yields fake int16 chunks, so these tests run
without the optional [piper] dependency or a downloaded voice model.
"""

import wave
from dataclasses import dataclass

from typer.testing import CliRunner

from ttsbench.adapters.base import SynthesisResult
from ttsbench.adapters.piper import PiperAdapter
from ttsbench.cli import app
from ttsbench.schemas import AudioFormat, ExecutionMode, RuntimeBackend, StreamingMode

runner = CliRunner()


@dataclass
class _FakeChunk:
    audio_int16_bytes: bytes
    sample_rate: int = 22050
    sample_width: int = 2
    sample_channels: int = 1


class _FakeConfig:
    sample_rate = 22050


class _FakeVoice:
    config = _FakeConfig()

    def synthesize(self, text):
        # One chunk per character so output length tracks the input deterministically.
        for ch in text:
            yield _FakeChunk(audio_int16_bytes=ch.encode().ljust(2, b"\x00"))


def _adapter_with_fake_engine(**kwargs) -> PiperAdapter:
    adapter = PiperAdapter(**kwargs)
    adapter._loaded = _FakeVoice()
    adapter._sample_rate = _FakeConfig.sample_rate
    return adapter


def test_piper_metadata():
    adapter = PiperAdapter(voice="en_US-lessac-medium", device="cpu")
    assert adapter.provider == "piper"
    assert adapter.execution_mode is ExecutionMode.LOCAL
    assert adapter.runtime_backend is RuntimeBackend.CPU
    assert adapter.audio_format is AudioFormat.PCM_S16LE
    assert adapter.streaming is False
    assert adapter.streaming_mode is StreamingMode.NONE
    assert adapter.requested_device == "cpu"
    assert adapter.actual_device == "cpu"
    assert adapter.voice == "en_US-lessac-medium"


def test_piper_synthesize_concatenates_chunks():
    adapter = _adapter_with_fake_engine()
    result = adapter.synthesize("abc")
    assert isinstance(result, SynthesisResult)
    assert result.audio == b"a\x00b\x00c\x00"
    assert result.sample_rate == 22050
    assert result.audio_format is AudioFormat.PCM_S16LE
    # 3 frames of 16-bit mono at 22050 Hz.
    assert result.duration_ms == 3 / 22050 * 1000.0


def test_synthesize_cli_writes_valid_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(PiperAdapter, "_load", lambda self: _FakeVoice())
    out = tmp_path / "hello.wav"
    result = runner.invoke(app, ["synthesize", "--text", "hello", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "execution_mode=local" in result.output
    assert "backend=cpu" in result.output

    with wave.open(str(out), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 22050
        assert wav.getnframes() == len("hello")


def test_synthesize_cli_rejects_unknown_provider(tmp_path):
    out = tmp_path / "x.wav"
    result = runner.invoke(
        app,
        ["synthesize", "--text", "hi", "--output", str(out), "--provider", "bogus"],
    )
    assert result.exit_code != 0
    assert not out.exists()
