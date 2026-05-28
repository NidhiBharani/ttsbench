"""Shared utility helpers."""

from __future__ import annotations

import wave
from pathlib import Path


def write_wav(
    pcm: bytes,
    sample_rate: int,
    path: str | Path,
    sample_width: int = 2,
    channels: int = 1,
) -> Path:
    """Write signed little-endian PCM to a WAV file.

    ``sample_width`` is bytes per sample (2 = 16-bit, the Piper default).
    Returns the written path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return out
