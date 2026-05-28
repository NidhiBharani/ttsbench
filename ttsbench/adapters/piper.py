"""Piper adapter: CPU-first local baseline. Phase 2.

Piper runs an ONNX voice model on the CPU. It is the lightest local adapter and
exists to validate the benchmark pipeline before heavier models or cloud
providers. ``synthesize`` is batch (not streaming): it concatenates Piper's
internal chunks into a single 16-bit PCM payload and returns a ``SynthesisResult``.

The ``piper`` package is imported lazily so importing this module (and running the
test suite) does not require the optional ``[piper]`` dependency to be installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ttsbench.adapters.base import SynthesisResult, TTSAdapter
from ttsbench.config import DEFAULT_PIPER_VOICE, resolve_piper_voice
from ttsbench.schemas import AudioFormat, ExecutionMode, RuntimeBackend, StreamingMode


class PiperAdapter(TTSAdapter):
    """Local Piper TTS adapter (batch, CPU)."""

    def __init__(
        self,
        voice: str | None = None,
        model_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self._voice = voice or DEFAULT_PIPER_VOICE
        self._requested_device = device
        self._model_path = Path(model_path) if model_path else None
        self._loaded: Any | None = None
        self._sample_rate = 22050

    def _load(self) -> Any:
        if self._loaded is not None:
            return self._loaded

        from piper import PiperVoice

        model_path = self._model_path or resolve_piper_voice(self._voice)
        voice = PiperVoice.load(str(model_path), use_cuda=False)
        self._sample_rate = voice.config.sample_rate
        self._loaded = voice
        return voice

    @property
    def provider(self) -> str:
        return "piper"

    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode.LOCAL

    @property
    def model(self) -> str:
        return str(self._model_path) if self._model_path else self._voice

    @property
    def voice(self) -> str:
        return self._voice

    @property
    def runtime_backend(self) -> RuntimeBackend:
        # Piper uses onnxruntime; the default execution provider is CPU.
        return RuntimeBackend.CPU

    @property
    def requested_device(self) -> str | None:
        return self._requested_device

    @property
    def actual_device(self) -> str | None:
        return "cpu"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def audio_format(self) -> AudioFormat:
        return AudioFormat.PCM_S16LE

    @property
    def streaming(self) -> bool:
        return False

    @property
    def streaming_mode(self) -> StreamingMode:
        return StreamingMode.NONE

    def synthesize(self, text: str, voice: str | None = None, **params: Any) -> SynthesisResult:
        engine = self._load()

        pcm = bytearray()
        sample_rate = self._sample_rate
        sample_width = 2
        channels = 1
        for chunk in engine.synthesize(text):
            pcm += chunk.audio_int16_bytes
            sample_rate = chunk.sample_rate
            sample_width = chunk.sample_width
            channels = chunk.sample_channels

        self._sample_rate = sample_rate
        frames = (len(pcm) // (sample_width * channels)) if sample_width and channels else 0
        duration_ms = (frames / sample_rate * 1000.0) if sample_rate else None

        return SynthesisResult(
            audio=bytes(pcm),
            sample_rate=sample_rate,
            audio_format=AudioFormat.PCM_S16LE,
            duration_ms=duration_ms,
        )
