"""Runtime detection reports sane local platform information."""

from ttsbench.runtime import detect_runtime, format_runtime


def test_detect_runtime_basic_fields():
    info = detect_runtime()
    assert info.platform
    assert info.python_version
    assert info.total_memory_gb > 0
    # cpu is always present; detection only lists usable backends.
    assert "cpu" in info.available_backends


def test_format_runtime_includes_platform_and_backends():
    text = format_runtime(detect_runtime())
    assert "Platform" in text
    assert "Backends available" in text
    assert "cpu" in text
