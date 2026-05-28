"""Adapter contract for local models and cloud providers.

An adapter is either *streaming* (``synthesize`` returns an ``AsyncIterator`` of
``AudioChunk``) or *batch* (``synthesize`` returns a completed ``SynthesisResult``).
The ``streaming`` property tells callers which shape to expect, so the benchmark
never fabricates streaming timing for a batch adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ttsbench.schemas import AudioFormat, ExecutionMode, RuntimeBackend, StreamingMode


@dataclass(slots=True)
class AudioChunk:
    """A single streamed audio chunk with its receive timestamp."""

    data: bytes
    recv_ns: int  # time.perf_counter_ns() captured when the chunk arrived
    index: int  # 0-based sequence index


@dataclass(slots=True)
class SynthesisResult:
    """A completed audio payload from a batch (non-streaming) adapter."""

    audio: bytes
    sample_rate: int
    audio_format: AudioFormat
    duration_ms: float | None = None


SynthesisOutput = AsyncIterator[AudioChunk] | SynthesisResult


class TTSAdapter(ABC):
    """Base class every TTS adapter implements."""

    def load(self) -> None:  # noqa: B027  (intentional optional no-op hook, not abstract)
        """Eagerly load the model so model-load time can be measured separately.

        Default is a no-op (cloud/provider-hosted adapters have nothing to load).
        Local adapters override this to load weights up front.
        """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider or runtime name, e.g. ``piper`` or ``cartesia``."""

    @property
    @abstractmethod
    def execution_mode(self) -> ExecutionMode: ...

    @property
    @abstractmethod
    def model(self) -> str: ...

    @property
    @abstractmethod
    def voice(self) -> str: ...

    @property
    @abstractmethod
    def runtime_backend(self) -> RuntimeBackend:
        """The backend that actually ran the model (records CPU fallback honestly)."""

    @property
    @abstractmethod
    def requested_device(self) -> str | None:
        """Device the caller asked for, e.g. ``mps``; ``None`` when unspecified."""

    @property
    @abstractmethod
    def actual_device(self) -> str | None:
        """Human label of the device that actually ran, e.g. ``Apple M3 Pro GPU``."""

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def audio_format(self) -> AudioFormat: ...

    @property
    @abstractmethod
    def streaming(self) -> bool:
        """True if ``synthesize`` yields chunks, False if it returns a batch result."""

    @property
    @abstractmethod
    def streaming_mode(self) -> StreamingMode: ...

    @abstractmethod
    def synthesize(self, text: str, voice: str | None = None, **params: Any) -> SynthesisOutput:
        """Synthesize ``text``.

        Streaming adapters return an ``AsyncIterator[AudioChunk]``; batch adapters
        return a ``SynthesisResult``.
        """
