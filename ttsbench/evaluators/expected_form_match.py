"""Normalize and substring-match expected spoken forms. Phase 5.

Normalization is applied to BOTH the ASR transcript and each expected pattern so
that, for example, "amoxicillin-clavulanate" and "amoxicillin clavulanate" match.
Order matters: hyphens become spaces *before* punctuation is stripped, otherwise
"amox-clav" would collapse to "amoxclav".
"""

from __future__ import annotations

import re
import unicodedata

from ttsbench.schemas import PatternMatch

# Punctuation to drop (hyphens are handled separately, as spaces).
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")
_HYPHENS = "-‐‑‒–—−"


def normalize(text: str) -> str:
    """Lowercase, hyphens->spaces, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    for hyphen in _HYPHENS:
        text = text.replace(hyphen, " ")
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def match_patterns(transcript: str, patterns: list[str]) -> list[PatternMatch]:
    """Substring-match each normalized pattern against the normalized transcript."""
    normalized_transcript = normalize(transcript)
    matches: list[PatternMatch] = []
    for pattern in patterns:
        normalized_pattern = normalize(pattern)
        found = bool(normalized_pattern) and normalized_pattern in normalized_transcript
        matches.append(PatternMatch(pattern=pattern, matched=found))
    return matches


def item_passes(matches: list[PatternMatch]) -> bool:
    """An item passes only when every required pattern matched.

    An item with no patterns has nothing to verify and is treated as a pass.
    """
    return all(match.matched for match in matches)
