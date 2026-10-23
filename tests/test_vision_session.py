"""Live vision session: one real encode per tick, decode matches."""

from __future__ import annotations

from lydlr_ai.model.temporal_dct import decode_frame, encode_frame
from lydlr_ai.model.vision_session import VisionSession


def test_encode_frame_roundtrip_and_rate():
    import numpy as np

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, :, 1] = 180
    payload, recon, stats = encode_frame(frame, None, qstep=16.0)
    back = decode_frame(payload, None)
    np.testing.assert_array_equal(back, recon)
    assert stats["bytes_out"] == len(payload)
    assert stats["bytes_out"] < stats["bytes_in"]
    assert stats["codec_path"] == "temporal_dct_rans"


def test_session_ticks_are_countable():
    session = VisionSession(qstep=16.0, max_width=64, n_frames=4)
    first = session.tick()["metric"]
    second = session.tick()["metric"]
    assert first["keyframe"] is True
    assert second["keyframe"] is False
    assert second["bytes_out"] < first["bytes_out"]
    assert second["compression_ratio"] > 1.0
    assert second["psnr"] > 30.0
    assert second["modality_bytes_in"]["camera"] == second["bytes_in"]
