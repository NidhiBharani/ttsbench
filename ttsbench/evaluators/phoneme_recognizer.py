"""Phoneme recognition from audio via wav2vec2 (eSpeak phoneme set). Phase 5.

A CTC phoneme model has no word-level language model, so it reports the sounds it
hears without snapping them to real words. Output uses the same eSpeak phoneme
inventory as the espeak reference (see ``phoneme_match``), so the two compare
directly. torch/transformers are imported lazily (optional ``[phoneme]`` extra).
Results are cached by audio SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import threading
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ttsbench.evaluators.phoneme_match import normalize_phonemes

_TARGET_SR = 16000


class PhonemeRecognizer(ABC):
    @abstractmethod
    def recognize(self, wav_path: str | Path) -> list[str]:
        """Return normalized phoneme tokens recognized from the audio."""

    def load(self) -> None:  # noqa: B027  (optional no-op hook, not abstract)
        """Eagerly load the model (call before a worker pool to avoid load races)."""

    @property
    @abstractmethod
    def name(self) -> str: ...


def _load_audio_16k_mono(path: str | Path) -> Any:
    """Read a 16-bit PCM WAV and return float32 mono samples at 16 kHz."""
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        frames = wav.readframes(wav.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sr != _TARGET_SR and audio.size:
        # Linear resample; adequate for phoneme recognition.
        new_len = int(round(audio.size * _TARGET_SR / sr))
        audio = np.interp(
            np.linspace(0, audio.size - 1, new_len, dtype=np.float64),
            np.arange(audio.size),
            audio,
        ).astype(np.float32)
    return audio


class Wav2Vec2PhonemeRecognizer(PhonemeRecognizer):
    # Tokens in the model vocab that are not phonemes.
    _SPECIAL = {"<pad>", "<s>", "</s>", "<unk>", "|", ""}

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._feature_extractor: Any | None = None
        self._model: Any | None = None
        self._id_to_token: dict[int, str] = {}

    def _load(self) -> None:
        # We decode CTC output to phonemes ourselves from vocab.json, avoiding the
        # Wav2Vec2Phoneme tokenizer (which pulls in phonemizer + system espeak-ng).
        if self._model is None:
            import json

            import torch
            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor

            self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self._model_name)
            self._model = AutoModelForCTC.from_pretrained(self._model_name).to(self._device)
            self._model.eval()
            vocab = json.loads(Path(hf_hub_download(self._model_name, "vocab.json")).read_text())
            self._id_to_token = {idx: tok for tok, idx in vocab.items()}
            self._torch = torch

    def load(self) -> None:
        self._load()

    @property
    def name(self) -> str:
        return f"wav2vec2:{self._model_name}"

    def recognize(self, wav_path: str | Path) -> list[str]:
        self._load()
        audio = _load_audio_16k_mono(wav_path)
        inputs = self._feature_extractor(  # type: ignore[misc]
            audio, sampling_rate=_TARGET_SR, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        ids = self._torch.argmax(logits, dim=-1)[0].tolist()

        # Greedy CTC decode: collapse runs of equal ids, drop special/blank tokens.
        tokens: list[str] = []
        previous: int | None = None
        for token_id in ids:
            if token_id != previous:
                token = self._id_to_token.get(token_id, "")
                if token not in self._SPECIAL:
                    tokens.append(token)
            previous = token_id
        return normalize_phonemes(tokens)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CachingPhonemeRecognizer(PhonemeRecognizer):
    """Caches recognized phoneme tokens by audio SHA-256 in a JSON file."""

    def __init__(self, backend: PhonemeRecognizer, cache_path: str | Path) -> None:
        self._backend = backend
        self._cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._cache: dict[str, list[str]] = {}
        if self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text())

    def load(self) -> None:
        self._backend.load()

    @property
    def name(self) -> str:
        return self._backend.name

    def recognize(self, wav_path: str | Path) -> list[str]:
        key = f"{self._backend.name}:{_sha256(wav_path)}"
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        phonemes = self._backend.recognize(wav_path)
        with self._lock:
            self._cache[key] = phonemes
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache))
        return phonemes
