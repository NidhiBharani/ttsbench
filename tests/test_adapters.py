"""A fake adapter can return either streamed chunks or a batch result."""

import time
from collections.abc import AsyncIterator
from typing import Any

from ttsbench.adapters.base import AudioChunk, SynthesisResult, TTSAdapter
from ttsbench.schemas import AudioFormat, ExecutionMode, RuntimeBackend, StreamingMode


class _FakeBatchAdapter(TTSAdapter):
    provider = "fake_batch"
    execution_mode = ExecutionMode.LOCAL
    model = "fake"
    voice = "default"
    runtime_backend = RuntimeBackend.CPU
    requested_device = "cpu"
    actual_device = "Test CPU"
    sample_rate = 22050
    audio_format = AudioFormat.PCM_S16LE
    streaming = False
    streaming_mode = StreamingMode.NONE

    def synthesize(self, text: str, voice: str | None = None, **params: Any) -> SynthesisResult:
        return SynthesisResult(
            audio=b"\x00\x01" * len(text),
            sample_rate=self.sample_rate,
            audio_format=self.audio_format,
            duration_ms=float(len(text)),
        )


class _FakeStreamingAdapter(TTSAdapter):
    provider = "fake_stream"
    execution_mode = ExecutionMode.CLOUD
    model = "fake"
    voice = "default"
    runtime_backend = RuntimeBackend.PROVIDER_HOSTED
    requested_device = None
    actual_device = None
    sample_rate = 24000
    audio_format = AudioFormat.PCM_S16LE
    streaming = True
    streaming_mode = StreamingMode.WEBSOCKET

    def synthesize(
        self, text: str, voice: str | None = None, **params: Any
    ) -> AsyncIterator[AudioChunk]:
        return self._stream(text)

    async def _stream(self, text: str) -> AsyncIterator[AudioChunk]:
        for index, ch in enumerate(text):
            yield AudioChunk(data=ch.encode(), recv_ns=time.perf_counter_ns(), index=index)


def test_batch_adapter_returns_synthesis_result():
    adapter = _FakeBatchAdapter()
    assert adapter.streaming is False
    result = adapter.synthesize("hello")
    assert isinstance(result, SynthesisResult)
    assert result.sample_rate == 22050
    assert adapter.requested_device == "cpu"
    assert adapter.actual_device == "Test CPU"


async def test_streaming_adapter_yields_chunks():
    adapter = _FakeStreamingAdapter()
    assert adapter.streaming is True
    chunks = [chunk async for chunk in adapter.synthesize("hey")]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert b"".join(c.data for c in chunks) == b"hey"
    assert all(c.recv_ns > 0 for c in chunks)
