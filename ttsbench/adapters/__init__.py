"""TTS adapters for local models and cloud providers."""

from __future__ import annotations

from ttsbench.adapters.base import TTSAdapter

LOCAL_PROVIDERS = ("piper", "kokoro")


def get_adapter(provider: str, voice: str | None = None, device: str | None = None) -> TTSAdapter:
    """Build an adapter by provider name (lazy imports keep optional deps optional)."""
    if provider == "piper":
        from ttsbench.adapters.piper import PiperAdapter

        return PiperAdapter(voice=voice, device=device)
    if provider == "kokoro":
        from ttsbench.adapters.kokoro import KokoroAdapter

        return KokoroAdapter(voice=voice, device=device)
    raise ValueError(f"provider '{provider}' is not implemented yet")
