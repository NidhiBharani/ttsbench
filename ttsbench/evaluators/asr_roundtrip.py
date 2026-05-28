"""ASR round-trip transcription via a local backend. Phase 5.

Synthesized audio is transcribed back to text so pronunciation can be checked
against expected spoken forms. The default backend is faster-whisper (lazily
imported so the base install does not require it). Results are cached by audio
SHA-256 so re-running just the matcher or re-rendering a report does not
re-transcribe.

ASR is imperfect and can silently "correct" misspoken medical terms, so callers
should always surface audio + transcript together for human spot-checking.
"""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class ASRResult:
    text: str
    words: list[Word] = field(default_factory=list)


class ASRBackend(ABC):
    """Transcribes a WAV file to text (and word timings when available)."""

    @abstractmethod
    def transcribe(self, wav_path: str | Path) -> ASRResult: ...

    def load(self) -> None:  # noqa: B027  (optional no-op hook, not abstract)
        """Eagerly load the model (call before a worker pool to avoid load races)."""

    @property
    @abstractmethod
    def name(self) -> str: ...


class FasterWhisperASR(ASRBackend):
    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def load(self) -> None:
        self._load()

    @property
    def name(self) -> str:
        return f"faster-whisper:{self._model_size}"

    def transcribe(self, wav_path: str | Path) -> ASRResult:
        model = self._load()
        segments, _info = model.transcribe(str(wav_path), word_timestamps=True)
        text_parts: list[str] = []
        words: list[Word] = []
        for segment in segments:
            text_parts.append(segment.text)
            for word in segment.words or []:
                words.append(Word(word=word.word, start=word.start, end=word.end))
        return ASRResult(text="".join(text_parts).strip(), words=words)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CachingASR(ASRBackend):
    """Wraps a backend, caching transcripts by audio SHA-256 in a JSON file."""

    def __init__(self, backend: ASRBackend, cache_path: str | Path) -> None:
        self._backend = backend
        self._cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] = {}
        if self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text())

    def load(self) -> None:
        self._backend.load()

    @property
    def name(self) -> str:
        return self._backend.name

    def transcribe(self, wav_path: str | Path) -> ASRResult:
        key = f"{self._backend.name}:{_sha256(wav_path)}"
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return ASRResult(
                text=cached["text"],
                words=[Word(**w) for w in cached.get("words", [])],
            )

        result = self._backend.transcribe(wav_path)
        with self._lock:
            self._cache[key] = {
                "text": result.text,
                "words": [vars(w) for w in result.words],
            }
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache))
        return result
