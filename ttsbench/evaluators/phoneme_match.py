"""Phoneme-level pronunciation matching. Phase 5 (revised).

Instead of comparing ASR *words* (which a language model snaps toward real words,
e.g. "Cho" -> "show") we compare *sounds*. The reference is the eSpeak phoneme
sequence of the intended spoken form; the produced phonemes come from a phoneme
recognizer run on the audio. Both use the eSpeak inventory, so they compare
directly after stripping stress/length markers.

Matching a pattern is approximate-substring edit distance: the pattern's phoneme
sequence may appear anywhere inside the produced stream, and it counts as found
when its phoneme error rate is within a tolerance. This is normalization-proof:
"500" and "five hundred" yield the same phonemes, so the digit/word mismatch that
plagued word-ASR disappears.
"""

from __future__ import annotations

import unicodedata

# eSpeak stress/length/boundary markers carry no phonemic identity for our purpose.
_DROP = {"ˈ", "ˌ", "ː", "ˑ", "‖", "|", " ", ""}
# A few eSpeak-specific symbols normalized toward their nearest base phone so the
# recognizer's inventory and espeak's reference line up.
_FOLD = {"ᵻ": "ɪ", "ᵿ": "ʊ", "ɚ": "ə", "ɝ": "ɜ"}


def normalize_phonemes(tokens: list[str]) -> list[str]:
    """Normalize to a comparable per-character phone stream.

    espeak emits one base phone per token (it already splits diphthongs and
    affricates: ``aɪ`` -> ``a`` ``ɪ``), but the recognizer emits whole phonemes
    like ``aɪ``/``oː`` as single tokens. Exploding every token to characters and
    dropping stress/length markers puts both on the same footing.
    """
    out: list[str] = []
    for raw in tokens:
        token = unicodedata.normalize("NFD", raw)
        for ch in token:
            if unicodedata.combining(ch):  # length/stress diacritics, nasalization
                continue
            ch = _FOLD.get(ch, ch)
            if ch in _DROP:
                continue
            out.append(ch)
    return out


def approx_substring_per(produced: list[str], pattern: list[str]) -> float:
    """Min phoneme error rate of ``pattern`` against any substring of ``produced``.

    Edit distance where matching can start and end anywhere in ``produced`` (free
    leading/trailing deletions), divided by the pattern length.
    """
    if not pattern:
        return 0.0
    if not produced:
        return 1.0

    m, n = len(pattern), len(produced)
    prev = [0] * (n + 1)  # row 0: empty pattern prefix matches anywhere at cost 0
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        pi = pattern[i - 1]
        for j in range(1, n + 1):
            sub = prev[j - 1] + (0 if pi == produced[j - 1] else 1)
            cur[j] = min(sub, prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return min(prev) / m


class EspeakReference:
    """Grapheme-to-phoneme for the intended spoken form, via Piper's bundled eSpeak."""

    def __init__(self, voice: str = "en-us") -> None:
        self._voice = voice
        self._phonemizer: object | None = None

    def _engine(self) -> object:
        if self._phonemizer is None:
            from piper.phonemize_espeak import EspeakPhonemizer

            self._phonemizer = EspeakPhonemizer()
        return self._phonemizer

    def phonemes(self, text: str) -> list[str]:
        sentences = self._engine().phonemize(self._voice, text)  # type: ignore[attr-defined]
        flat: list[str] = []
        for sentence in sentences:
            flat.extend(sentence)
        return normalize_phonemes(flat)
