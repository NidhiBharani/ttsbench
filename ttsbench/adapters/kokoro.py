"""Kokoro adapter: lightweight local model with Apple Silicon GPU path. Phase 7.

Runs Kokoro via ``kokoro-onnx`` (onnxruntime). On Apple Silicon the CoreML
execution provider offloads to the GPU/ANE; otherwise it runs on CPU. The
PyTorch and MLX runtimes both require ``misaki``/``spacy`` for English G2P, which
has no Python 3.14 wheel, so the ONNX path is the maintainable choice here.

``--device mps``/``coreml``/``gpu`` request CoreML; ``--device cpu`` forces CPU.
We record the requested device and the backend actually registered, so a CoreML
request that silently falls back to CPU (e.g. on non-Apple hardware) is visible.
``kokoro-onnx`` is imported lazily (optional ``[kokoro]`` extra).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ttsbench.adapters.base import SynthesisResult, TTSAdapter
from ttsbench.config import DEFAULT_KOKORO_VOICE, resolve_kokoro_files
from ttsbench.schemas import AudioFormat, ExecutionMode, RuntimeBackend, StreamingMode

_SAMPLE_RATE = 24000
_COREML_DEVICES = {"mps", "coreml", "gpu", "metal"}


class KokoroAdapter(TTSAdapter):
    """Local Kokoro TTS adapter (batch) on onnxruntime, CoreML or CPU."""

    def __init__(self, voice: str | None = None, device: str | None = None) -> None:
        self._voice = voice or DEFAULT_KOKORO_VOICE
        self._requested_device = device
        self._engine: Any | None = None
        self._coreml_active = False

    def _requested_providers(self) -> list[str]:
        if self._requested_device in _COREML_DEVICES:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def load(self) -> None:
        self._load()

    def _load(self) -> Any:
        if self._engine is not None:
            return self._engine

        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        model_path, voices_path = resolve_kokoro_files()
        session = ort.InferenceSession(str(model_path), providers=self._requested_providers())
        self._coreml_active = "CoreMLExecutionProvider" in session.get_providers()
        self._engine = Kokoro.from_session(session, str(voices_path))
        return self._engine

    @property
    def provider(self) -> str:
        return "kokoro"

    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode.LOCAL

    @property
    def model(self) -> str:
        return "kokoro-v1.0"

    @property
    def voice(self) -> str:
        return self._voice

    @property
    def runtime_backend(self) -> RuntimeBackend:
        # Determined after load(); CoreML when the GPU/ANE provider registered.
        return RuntimeBackend.COREML if self._coreml_active else RuntimeBackend.ONNX

    @property
    def requested_device(self) -> str | None:
        return self._requested_device

    @property
    def actual_device(self) -> str | None:
        return "coreml" if self._coreml_active else "cpu"

    @property
    def sample_rate(self) -> int:
        return _SAMPLE_RATE

    @property
    def audio_format(self) -> AudioFormat:
        return AudioFormat.PCM_S16LE

    @property
    def streaming(self) -> bool:
        return False

    @property
    def streaming_mode(self) -> StreamingMode:
        return StreamingMode.NONE

    def synthesize(
        self, text: str, voice: str | None = None, **params: Any
    ) -> SynthesisResult:
        engine = self._load()
        samples, sample_rate = engine.create(
            text, voice=voice or self._voice, speed=params.get("speed", 1.0), lang="en-us"
        )
        samples = np.asarray(samples, dtype=np.float32)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        duration_ms = (len(samples) / sample_rate * 1000.0) if sample_rate else None

        return SynthesisResult(
            audio=pcm,
            sample_rate=int(sample_rate),
            audio_format=AudioFormat.PCM_S16LE,
            duration_ms=duration_ms,
        )
