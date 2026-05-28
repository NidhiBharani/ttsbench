"""Configuration and model/voice resolution.

Cloud price tables are added in a later phase. For now this resolves local Piper
voice models to on-disk ``.onnx`` paths, downloading them on demand into a cache
directory.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PIPER_VOICE = "en_US-lessac-medium"


def piper_model_dir() -> Path:
    """Directory where Piper voice models are cached.

    Overridable with ``TTSBENCH_PIPER_MODEL_DIR``; defaults to
    ``~/.cache/ttsbench/piper``.
    """
    env = os.environ.get("TTSBENCH_PIPER_MODEL_DIR")
    base = Path(env) if env else Path.home() / ".cache" / "ttsbench" / "piper"
    return base


def resolve_piper_voice(
    voice: str | None = None,
    model_dir: Path | None = None,
    download: bool = True,
) -> Path:
    """Resolve a Piper voice name to its ``.onnx`` model path.

    ``voice`` is a Piper voice key like ``en_US-lessac-medium``. If the model is
    not already present in ``model_dir`` and ``download`` is true, it is fetched.
    Raises ``FileNotFoundError`` if the model is missing and downloading is off.
    """
    name = voice or DEFAULT_PIPER_VOICE
    directory = model_dir or piper_model_dir()
    model_path = directory / f"{name}.onnx"

    if model_path.exists():
        return model_path

    if not download:
        raise FileNotFoundError(
            f"Piper voice '{name}' not found at {model_path}. "
            "Download it or set TTSBENCH_PIPER_MODEL_DIR to its location."
        )

    directory.mkdir(parents=True, exist_ok=True)
    from piper.download_voices import download_voice

    download_voice(name, directory)
    return model_path
