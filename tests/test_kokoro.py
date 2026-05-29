"""Phase 7: Kokoro adapter (onnxruntime, CoreML/CPU) with honest backend reporting.

kokoro-onnx and the model files are mocked, so these tests need neither the
optional dependency nor a ~300MB download.
"""

import wave

import numpy as np
from typer.testing import CliRunner

from ttsbench.adapters import get_adapter
from ttsbench.adapters.base import SynthesisResult
from ttsbench.adapters.kokoro import KokoroAdapter
from ttsbench.cli import app
from ttsbench.schemas import AudioFormat, ExecutionMode, RuntimeBackend, StreamingMode

runner = CliRunner()


class _FakeEngine:
    def create(self, text, voice, speed=1.0, lang="en-us"):
        # 0.5s of quiet audio at 24 kHz.
        return np.zeros(12000, dtype=np.float32), 24000


def _adapter_with_fake(device=None, coreml=False) -> KokoroAdapter:
    adapter = KokoroAdapter(device=device)
    adapter._engine = _FakeEngine()
    adapter._coreml_active = coreml
    return adapter


def test_kokoro_metadata_defaults_to_onnx_cpu():
    adapter = _adapter_with_fake(device="cpu", coreml=False)
    assert adapter.provider == "kokoro"
    assert adapter.execution_mode is ExecutionMode.LOCAL
    assert adapter.model == "kokoro-v1.0"
    assert adapter.runtime_backend is RuntimeBackend.ONNX
    assert adapter.actual_device == "cpu"
    assert adapter.sample_rate == 24000
    assert adapter.streaming is False
    assert adapter.streaming_mode is StreamingMode.NONE


def test_kokoro_reports_coreml_when_active():
    adapter = _adapter_with_fake(device="mps", coreml=True)
    assert adapter.requested_device == "mps"
    assert adapter.runtime_backend is RuntimeBackend.COREML
    assert adapter.actual_device == "coreml"


def test_mps_request_maps_to_coreml_providers():
    assert KokoroAdapter(device="mps")._requested_providers()[0] == "CoreMLExecutionProvider"
    assert KokoroAdapter(device="coreml")._requested_providers()[0] == "CoreMLExecutionProvider"
    assert KokoroAdapter(device="cpu")._requested_providers() == ["CPUExecutionProvider"]
    # Default (no device) stays on CPU for reliability.
    assert KokoroAdapter()._requested_providers() == ["CPUExecutionProvider"]


def test_kokoro_synthesize_produces_pcm():
    adapter = _adapter_with_fake(device="cpu")
    result = adapter.synthesize("hello")
    assert isinstance(result, SynthesisResult)
    assert result.sample_rate == 24000
    assert result.audio_format is AudioFormat.PCM_S16LE
    assert len(result.audio) == 12000 * 2  # int16 = 2 bytes/sample
    assert result.duration_ms == 12000 / 24000 * 1000.0


def test_get_adapter_factory():
    assert get_adapter("kokoro").provider == "kokoro"
    assert get_adapter("piper").provider == "piper"


def _fake_load(self: KokoroAdapter) -> _FakeEngine:
    self._engine = _FakeEngine()
    return self._engine


def _fake_synthesize(
    self: KokoroAdapter, text: str, voice: str | None = None, **p: object
) -> SynthesisResult:
    return SynthesisResult(
        audio=b"\x00\x00" * 12000,
        sample_rate=24000,
        audio_format=AudioFormat.PCM_S16LE,
        duration_ms=500.0,
    )


def test_synthesize_cli_with_kokoro(tmp_path, monkeypatch):
    # Patch model load + engine so no download or onnxruntime session is needed.
    monkeypatch.setattr(KokoroAdapter, "_load", _fake_load)
    monkeypatch.setattr(KokoroAdapter, "synthesize", _fake_synthesize)
    out = tmp_path / "k.wav"
    result = runner.invoke(
        app,
        ["synthesize", "--provider", "kokoro", "--text", "hi", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "provider=kokoro" in result.output
    with wave.open(str(out), "rb") as wav:
        assert wav.getframerate() == 24000
