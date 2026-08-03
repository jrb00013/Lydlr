"""Tests for preview helpers that do not require Redis/Django apps."""
import base64
from pathlib import Path


def _load_preview_constants():
    """Same placeholder JPEG used by preview_views."""
    b64 = (
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
        "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIy"
        "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIA"
        "AhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEB"
        "AQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8A/9k="
    )
    return base64.b64decode(b64)


def test_placeholder_jpeg_is_valid_soi():
    data = _load_preview_constants()
    assert data[:2] == b"\xff\xd8"
    assert len(data) > 20


def test_b64_roundtrip_placeholder():
    raw = _load_preview_constants()
    b64 = base64.b64encode(raw).decode("ascii")
    assert base64.b64decode(b64) == raw


def test_preview_sides_constant_in_source():
    path = Path(__file__).resolve().parents[1] / "api" / "views" / "preview_views.py"
    text = path.read_text()
    assert '"raw"' in text and '"reconstructed"' in text and '"heatmap"' in text
