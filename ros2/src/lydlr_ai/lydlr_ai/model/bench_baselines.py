# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# Matched-rate baseline codecs and the Lydlr rate/distortion adapter.
# See docs/architecture/BENCHMARK_PROTOCOL.md

"""Real baseline codecs + a matched-rate comparison against the Lydlr path.

The project's stated goal is to beat JPEG. That goal is only falsifiable if
something measures JPEG on the same pixels at the same rate. This module is
that something: one interface over every codec we can actually run on this
machine, a bisection search that lands each codec on a target bits-per-pixel,
and an adapter that reports the Lydlr path's *countable* wire rate.

Invariants enforced here (see BENCHMARK_PROTOCOL.md):

- A reported ``bits`` value is always countable payload. The differentiable
  entropy proxy is carried in a separate field and is never promoted to bits.
- If no real entropy coder is importable, ``rate_source`` is
  ``"fixed_length_bits"`` and ``bits == fixed_length_bits`` exactly.
- A codec that cannot be produced on this host is reported as unavailable with
  a reason. It is never approximated, and its numbers are never invented.
- Rate matching lands within a declared tolerance or is flagged ``matched=False``
  with the reason it could not.

Pluggable Lydlr entropy-coder contract
--------------------------------------
The Lydlr side of the comparison is a plugin point, because the rANS coder in
``lydlr_ai.model.entropy_coder`` is written by a different part of the project
and may not exist yet. To make the harness measure real bits when it does, the
module must expose any one of:

- ``rans_encode(indices, probabilities) -> bytes`` (preferred), or
- ``encode_indices(indices, probabilities) -> bytes``, or
- ``rans_encode_indices(indices, probabilities) -> bytes``, or
- a class ``RansCoder`` / ``RansEncoder`` / ``EntropyCoder`` with an
  ``encode(indices, probabilities) -> bytes`` method.

``indices`` is a 1-D ``np.ndarray`` of ``np.uint8`` symbol values and
``probabilities`` is a 1-D ``float64`` array summing to ~1 over the same
alphabet. The returned ``bytes`` is the payload the decoder must read. If none
of those exist, the harness falls back to fixed-length bits and says so in
``rate_source``.

Every call is wrapped in ``try/except``; a coder that imports but raises is
treated exactly like a coder that is absent, with the traceback message
recorded as the reason.
"""

from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

RGB_DTYPE = np.uint8
PSNR_MAX = 99.0

PROXY = "entropy_proxy"
FIXED_LENGTH = "fixed_length_bits"
RANS = "rans_bits"
RATE_SOURCES = (RANS, FIXED_LENGTH)


# ---------------------------------------------------------------------------
# metrics (self-contained; no cross-file coupling)
# ---------------------------------------------------------------------------


def to_float01(img: np.ndarray) -> np.ndarray:
    """Convert a uint8 HxWx3 image to float32 HxWx3 in [0, 1]."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two float arrays in the same range."""
    a = to_float01(a) if a.dtype == np.uint8 else a.astype(np.float32)
    b = to_float01(b) if b.dtype == np.uint8 else b.astype(np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch in mse: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB with data_range=1.0. Identical inputs return PSNR_MAX."""
    err = mse(a, b)
    if err <= 1e-12:
        return PSNR_MAX
    return float(10.0 * math.log10(1.0 / err))


def _luma(img: np.ndarray) -> np.ndarray:
    f = to_float01(img) if img.dtype == np.uint8 else img.astype(np.float32)
    if f.ndim == 2:
        return f
    return (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).astype(np.float64)


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


_SSIM_WINDOW = _gaussian_window()


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM (Wang et al. 2004) with an 11x11 Gaussian window, sigma=1.5.

    Computed on Rec.601 luma with data_range=1.0 and 'valid' cropping. Returns
    exactly 1.0 for identical inputs. Self-contained so the benchmark does not
    depend on skimage being installed.
    """
    x = _luma(a)
    y = _luma(b)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch in ssim: {x.shape} vs {y.shape}")
    if min(x.shape) < 11:
        small = min(x.shape)
        return _ssim_uniform(x, y, win=small if small % 2 else small - 1)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    w = _SSIM_WINDOW
    mu_x = _filter2d(x, w)
    mu_y = _filter2d(y, w)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = _filter2d(x * x, w) - mu_x2
    sigma_y2 = _filter2d(y * y, w) - mu_y2
    sigma_xy = _filter2d(x * y, w) - mu_xy
    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(num / den))


def _filter2d(img: np.ndarray, window: np.ndarray) -> np.ndarray:
    k = window.shape[0]
    h, w = img.shape
    if h < k or w < k:
        pad = k - min(h, w)
        img = np.pad(img, pad, mode="edge")
        h, w = img.shape
    out = np.zeros((h - k + 1, w - k + 1), dtype=np.float64)
    for dy in range(k):
        for dx in range(k):
            coef = window[dy, dx]
            if coef == 0.0:
                continue
            out += coef * img[dy : dy + out.shape[0], dx : dx + out.shape[1]]
    return out


def _ssim_uniform(x: np.ndarray, y: np.ndarray, win: int) -> float:
    if win < 3:
        return 1.0 if np.array_equal(x, y) else 0.0
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    yp = np.pad(y, pad, mode="edge")
    mu_x = _box(xp, win)
    mu_y = _box(yp, win)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    s_x2 = _box(xp * xp, win) - mu_x2
    s_y2 = _box(yp * yp, win) - mu_y2
    s_xy = _box(xp * yp, win) - mu_xy
    num = (2.0 * mu_xy + c1) * (2.0 * s_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (s_x2 + s_y2 + c2)
    return float(np.mean(num / den))


def _box(img: np.ndarray, win: int) -> np.ndarray:
    c = np.cumsum(np.cumsum(np.pad(img, ((1, 0), (1, 0))), axis=0), axis=1)
    c = np.pad(c, ((0, 1), (0, 1)))
    h, w = img.shape
    return (
        c[win : win + h, win : win + w]
        - c[0:h, win : win + w]
        - c[win : win + h, 0:w]
        + c[0:h, 0:w]
    ) / float(win * win)


class LpipsProbe:
    """Lazily-built, failure-tolerant LPIPS metric.

    LPIPS pulls an AlexNet checkpoint from download.pytorch.org on first use.
    On a box with no egress that raises; we record the reason once and report
    ``available=False`` forever after instead of retrying per frame.
    """

    def __init__(self, net: str = "alex", device: str = "cpu") -> None:
        self.net = net
        self.device = device
        self._model = None
        self._tried = False
        self.available = False
        self.reason = "not requested"

    def ensure(self) -> bool:
        if self._tried:
            return self.available
        self._tried = True
        try:
            import lpips  # noqa: WPS433 (optional, lazy by design)
            import torch
        except Exception as exc:
            self.reason = f"lpips import failed: {type(exc).__name__}: {exc}"
            self.available = False
            return False
        try:
            self._model = lpips.LPIPS(net=self.net).to(self.device).eval()
            self.available = True
            self.reason = "ok"
        except Exception as exc:
            self.reason = f"lpips weights unavailable: {type(exc).__name__}: {exc}"
            self.available = False
        return self.available

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        if not self.ensure():
            return float("nan")
        import torch

        ta = torch.from_numpy(to_float01(a)).permute(2, 0, 1)[None].to(self.device) * 2 - 1
        tb = torch.from_numpy(to_float01(b)).permute(2, 0, 1)[None].to(self.device) * 2 - 1
        if ta.shape[-2:] != tb.shape[-2:]:
            tb = torch.nn.functional.interpolate(
                tb, size=ta.shape[-2:], mode="bilinear", align_corners=False
            )
        with torch.no_grad():
            return float(self._model(ta, tb).mean().item())


# ---------------------------------------------------------------------------
# clip container
# ---------------------------------------------------------------------------


@dataclass
class Clip:
    """A benchmark clip: RGB uint8 frames plus provenance.

    ``source`` is ``"synthetic"`` or ``"real"`` and it is not cosmetic. The
    structured Markov scenes in ``scripts/structured_synthetic_data.py`` are
    band-limited by construction, which is the easiest possible material for a
    transform coder. Any Lydlr number measured on them is circular.
    """

    name: str
    frames: List[np.ndarray]
    source: str = "unknown"
    provenance: str = ""
    side_inputs: Optional[List[dict]] = None

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"

    @property
    def height(self) -> int:
        return int(self.frames[0].shape[0])

    @property
    def width(self) -> int:
        return int(self.frames[0].shape[1])

    @property
    def pixels(self) -> int:
        return int(self.frames[0].shape[0] * self.frames[0].shape[1])

    def describe(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "synthetic": self.is_synthetic,
            "provenance": self.provenance,
            "frames_used": len(self.frames),
            "height": self.height,
            "width": self.width,
            "raw_bpp": 24.0,
            "has_side_inputs": self.side_inputs is not None,
        }


def load_clip_npz(
    path: str,
    *,
    max_frames: Optional[int] = None,
    stride: int = 1,
    source: Optional[str] = None,
) -> Clip:
    """Load a fixture/recorded clip NPZ written by ``scripts/record_sensor_clip.py``.

    The NPZ stores an object array of per-frame dicts with at least ``image``
    (uint8 HxWx3). A clip with a ``real`` marker recorded in its metadata is
    treated as real; fixtures generated by ``record_sensor_clip.py`` are
    synthetic and are labelled as such unless the caller overrides.
    """
    data = np.load(path, allow_pickle=True)
    if "frames" not in data.files:
        raise ValueError(f"{path}: no 'frames' array")
    records = list(data["frames"])
    vertical = str(data["vertical"]) if "vertical" in data.files else "clip"
    frames: List[np.ndarray] = []
    side: List[dict] = []
    for rec in records[:: max(1, stride)]:
        img = np.asarray(rec["image"])
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"{path}: expected HxWx3 uint8 image, got {img.shape}")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        frames.append(img)
        side.append({k: np.asarray(v) for k, v in dict(rec).items() if k != "image"})
        if max_frames is not None and len(frames) >= max_frames:
            break
    resolved = source or _sniff_source(path, data)
    return Clip(
        name=f"{vertical}_clip",
        frames=frames,
        source=resolved,
        provenance=str(path),
        side_inputs=side,
    )


def _sniff_source(path: str, data) -> str:
    if "source" in data.files:
        return str(data["source"])
    base = os.path.basename(path).lower()
    if "real" in base or "recorded" in base:
        return "real"
    return "synthetic"


def synthetic_markov_clip(
    *,
    name: str = "markov_synthetic",
    frames: int = 8,
    height: int = 224,
    width: int = 224,
    num_blobs: int = 5,
    seed: int = 0,
) -> Clip:
    """Generate a clip from ``scripts/structured_synthetic_data.py`` (SYNTHETIC).

    Import is lazy and defensive: if the ROS package path is unavailable the
    caller gets a clear error rather than a silently different scene process.
    """
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))))
    scripts_dir = os.path.join(root, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        import torch

        from structured_synthetic_data import init_scene, step_scene
    except Exception as exc:
        raise RuntimeError(
            f"synthetic scene generator unavailable: {type(exc).__name__}: {exc}"
        ) from exc

    torch.manual_seed(seed)
    device = torch.device("cpu")
    scene = init_scene(1, device, height=height, width=width, num_blobs=num_blobs)
    out: List[np.ndarray] = []
    with torch.no_grad():
        for _ in range(frames):
            scene, obs = step_scene(scene, cut_prob=0.03)
            img = obs["image"][0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            out.append((img * 255.0).round().astype(np.uint8))
    return Clip(
        name=name,
        frames=out,
        source="synthetic",
        provenance="scripts/structured_synthetic_data.py (structured Markov scenes)",
    )


# ---------------------------------------------------------------------------
# baseline codec interface
# ---------------------------------------------------------------------------


@dataclass
class OperatingPoint:
    """One measured (rate, distortion) point for one codec on one clip."""

    codec: str
    param: float
    bytes_total: int
    n_frames: int
    pixels_per_frame: int
    bpp: float
    psnr_mean: float
    psnr_std: float
    psnr_min: float
    psnr_max: float
    ssim_mean: float
    ssim_std: float
    lpips_mean: float
    rate_measurement: str
    extras: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "codec": self.codec,
            "param": round(float(self.param), 4),
            "bytes_total": int(self.bytes_total),
            "bytes_per_frame": self.bytes_total / max(self.n_frames, 1),
            "n_frames": int(self.n_frames),
            "bpp": round(float(self.bpp), 6),
            "psnr_mean": round(float(self.psnr_mean), 4),
            "psnr_std": round(float(self.psnr_std), 4),
            "psnr_min": round(float(self.psnr_min), 4),
            "psnr_max": round(float(self.psnr_max), 4),
            "ssim_mean": round(float(self.ssim_mean), 5),
            "ssim_std": round(float(self.ssim_std), 5),
            "lpips_mean": None
            if not math.isfinite(self.lpips_mean)
            else round(float(self.lpips_mean), 5),
        }
        d.update(self.extras)
        return d


class BaselineCodec:
    """Interface every baseline implements.

    Subclasses must define ``name``, ``param_label``, ``rate_ordered_params``,
    and ``run(param, frames, want_decode)``.
    """

    name: str = "unnamed"
    kind: str = "image"
    param_label: str = "quality"
    param_is_integral: bool = True
    rate_measurement: str = "payload_bytes"
    notes: str = ""
    has_rate_knob: bool = True
    reference_only: bool = False

    def rate_ordered_params(self, step: float = 1.0) -> List[float]:
        """Quality parameters sorted by ASCENDING expected encoded size."""
        raise NotImplementedError

    def run(
        self, param: float, frames: Sequence[np.ndarray], want_decode: bool = True
    ) -> Tuple[int, List[np.ndarray]]:
        """Encode ``frames``; return (total payload bytes, decoded frames)."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "param_label": self.param_label,
            "rate_measurement": self.rate_measurement,
            "has_rate_knob": self.has_rate_knob,
            "reference_only": self.reference_only,
            "notes": self.notes,
        }


# ---- raw / lossless reference -------------------------------------------


class RawBaseline(BaselineCodec):
    """Uncompressed RGB24. The 24 bpp ceiling every other codec is measured under."""

    name = "raw_rgb24"
    kind = "image"
    has_rate_knob = False
    reference_only = True
    notes = "identity; rate is exactly 24 bpp by construction"

    def rate_ordered_params(self, step: float = 1.0) -> List[float]:
        return [0.0]

    def run(self, param, frames, want_decode: bool = True):
        total = int(sum(int(f.nbytes) for f in frames))
        return total, [np.ascontiguousarray(f) for f in frames] if want_decode else []


# ---- JPEG / WebP / PNG via PIL / cv2 ------------------------------------


def _pil_available() -> Tuple[bool, str]:
    try:
        import PIL.Image  # noqa: F401

        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _cv2_available() -> Tuple[bool, str]:
    try:
        import cv2  # noqa: F401

        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


class PILImageBaseline(BaselineCodec):
    """Single-image baseline through Pillow (JPEG / WebP / PNG)."""

    kind = "image"

    def __init__(
        self,
        name: str,
        fmt: str,
        param_label: str,
        save_kw_for: Callable[[float], dict],
        params: Sequence[float],
        *,
        notes: str = "",
        reference_only: bool = False,
    ) -> None:
        self.name = name
        self._fmt = fmt
        self.param_label = param_label
        self._save_kw = save_kw_for
        self._params = list(params)
        self.rate_measurement = "compressed_file_bytes"
        self.notes = notes
        self.reference_only = reference_only

    def rate_ordered_params(self, step: float = 1.0) -> List[float]:
        return list(self._params)

    def run(self, param, frames, want_decode: bool = True):
        from PIL import Image

        total = 0
        decoded: List[np.ndarray] = []
        for f in frames:
            buf = io.BytesIO()
            Image.fromarray(f).save(buf, self._fmt, **self._save_kw(float(param)))
            payload = buf.getvalue()
            total += len(payload)
            if want_decode:
                with Image.open(io.BytesIO(payload)) as im:
                    decoded.append(np.asarray(im.convert("RGB")))
        return total, decoded


class Cv2ImageBaseline(BaselineCodec):
    """Single-image baseline through ``cv2.imencode`` (JPEG / WebP / PNG).

    Arrays are RGB in and RGB out; the BGR swap is done explicitly so the
    measured distortion is never a channel-order artifact.
    """

    kind = "image"

    def __init__(
        self,
        name: str,
        ext: str,
        param_flags: Callable[[float], list],
        params: Sequence[float],
        *,
        notes: str = "",
    ) -> None:
        self.name = name
        self._ext = ext
        self._flags = param_flags
        self._params = list(params)
        self.rate_measurement = "compressed_file_bytes"
        self.notes = notes

    def rate_ordered_params(self, step: float = 1.0) -> List[float]:
        return list(self._params)

    def run(self, param, frames, want_decode: bool = True):
        import cv2

        total = 0
        decoded: List[np.ndarray] = []
        for f in frames:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(self._ext, bgr, self._flags(float(param)))
            if not ok:
                raise RuntimeError(f"{self.name}: cv2.imencode failed at {param}")
            total += int(buf.size)
            if want_decode:
                out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if out is None:
                    raise RuntimeError(f"{self.name}: cv2.imdecode failed at {param}")
                decoded.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        return total, decoded


class ZlibBaseline(BaselineCodec):
    """Raw zlib on the RGB24 buffer. Reference point, not a real image format."""

    name = "zlib_rgb24"
    kind = "image"
    param_label = "level"
    param_is_integral = True
    has_rate_knob = True
    rate_measurement = "compressed_file_bytes"
    notes = "headerless zlib stream on the raw RGB24 buffer"

    def rate_ordered_params(self, step: float = 1.0) -> List[float]:
        return [float(p) for p in range(9, -1, -1)]

    def run(self, param, frames, want_decode: bool = True):
        import zlib

        total = 0
        decoded: List[np.ndarray] = []
        shape = frames[0].shape
        for f in frames:
            payload = zlib.compress(np.ascontiguousarray(f).tobytes(), int(param))
            total += len(payload)
            if want_decode:
                raw = zlib.decompress(payload)
                decoded.append(np.frombuffer(raw, dtype=np.uint8).reshape(shape))
        return total, decoded


# ---- ffmpeg video codec baseline -----------------------------------------


def ffmpeg_available() -> Tuple[bool, str]:
    """Check for an ffmpeg/ffprobe pair that can encode at least one codec."""
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg binary not on PATH"
    if shutil.which("ffprobe") is None:
        return False, "ffprobe binary not on PATH (needed for per-packet sizes)"
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception as exc:
        return False, f"ffmpeg query failed: {type(exc).__name__}: {exc}"
    if not out:
        return False, "ffmpeg -encoders returned nothing"
    return True, "ok"


def _ffmpeg_has_encoder(name: str) -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


class FFmpegVideoBaseline(BaselineCodec):
    """H.264 / H.265 / VP9 / AV1 baseline through the ffmpeg CLI.

    Rate is the sum of coded packet sizes reported by ``ffprobe
    -show_entries packet=size``, so container/header overhead is excluded and
    the number is coded payload. In ``intra`` mode every packet is one frame,
    so the per-frame size is exact. In ``gop`` mode the reported rate is the
    mean over the clip and the inter-frame prediction it exploits is real
    temporal redundancy, which no still-image baseline gets to use; that mode is
    therefore reported separately and never mixed into the still-image table.
    """

    kind = "video"

    def __init__(
        self,
        name: str,
        encoder: str,
        *,
        mode: str = "intra",
        param_label: str = "crf",
        params: Optional[Sequence[float]] = None,
        pix_fmt: str = "yuv444p",
        container: str = "mp4",
        notes: str = "",
    ) -> None:
        self.name = name
        self.encoder = encoder
        self.mode = mode
        self.param_label = param_label
        self.pix_fmt = pix_fmt
        self.container = container
        self._params = list(params) if params is not None else [
            float(p) for p in (51, 46, 42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 0)
        ]
        self.rate_measurement = (
            "ffprobe_packet_sum" if mode == "intra" else "ffprobe_packet_sum_over_gop"
        )
        self.notes = notes or (
            "all-intra, per-frame packet size" if mode == "intra"
            else "default GOP, mean per-frame packet size (uses inter prediction)"
        )

    def rate_ordered_params(self, step: float = 1.0) -> List[float]:
        return sorted(self._params, key=lambda p: -p)

    def _encode_args(self, param: float) -> List[str]:
        args = ["-c:v", self.encoder, "-pix_fmt", self.pix_fmt, "-preset", "medium",
                "-b:v", "0", "-crf", f"{int(param)}"]
        if self.mode == "intra":
            args += ["-g", "1", "-keyint_min", "1", "-bf", "0", "-sc_threshold", "0"]
        else:
            args += ["-g", "8", "-keyint_min", "8", "-bf", "2"]
        if self.encoder == "libx264":
            args += ["-x264-params", "keyint=1:min-keyint=1:scenecut=0" if self.mode == "intra"
                     else "keyint=8:min-keyint=8:scenecut=0"]
        if self.encoder == "libx265":
            args += ["-x265-params",
                     "keyint=1:min-keyint=1:scenecut=0:log-level=error" if self.mode == "intra"
                     else "keyint=8:min-keyint=8:scenecut=0:log-level=error"]
        if self.encoder == "libvpx-vp9":
            args += ["-auto-alt-ref", "0", "-lag-in-frames", "0", "-row-mt", "1",
                     "-deadline", "good", "-cpu-used", "2"]
        if self.encoder == "libaom-av1":
            args += ["-cpu-used", "6", "-row-mt", "1", "-tiles", "1x1"]
        return args

    def run(self, param, frames, want_decode: bool = True):
        if not frames:
            return 0, []
        h, w = frames[0].shape[0], frames[0].shape[1]
        raw = b"".join(np.ascontiguousarray(f).tobytes() for f in frames)
        with tempfile.TemporaryDirectory(prefix="lydlr_bench_") as tmp:
            path = os.path.join(tmp, f"clip.{self.container}")
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "10",
                "-i", "-",
            ] + self._encode_args(float(param)) + [path]
            proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=300)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{self.name}: ffmpeg encode failed at param={param}: "
                    f"{proc.stderr.decode(errors='replace')[:200]}"
                )
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise RuntimeError(f"{self.name}: ffmpeg produced no output")
            sizes = _ffprobe_packet_sizes(path)
            if not sizes:
                raise RuntimeError(f"{self.name}: ffprobe reported no packet sizes")
            total = int(sum(sizes))
            decoded: List[np.ndarray] = []
            if want_decode:
                dec = subprocess.run(
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path,
                     "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                    capture_output=True,
                    timeout=300,
                )
                if dec.returncode != 0:
                    raise RuntimeError(
                        f"{self.name}: ffmpeg decode failed: "
                        f"{dec.stderr.decode(errors='replace')[:200]}"
                    )
                buf = np.frombuffer(dec.stdout, dtype=np.uint8)
                per = h * w * 3
                n = buf.size // per
                if n != len(frames):
                    raise RuntimeError(
                        f"{self.name}: decoded {n} frames, expected {len(frames)}"
                    )
                decoded = [
                    buf[i * per : (i + 1) * per].reshape(h, w, 3).copy()
                    for i in range(n)
                ]
        return total, decoded


def _ffprobe_packet_sizes(path: str) -> List[int]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=size", "-of", "csv=p=0", path],
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        return []
    out: List[int] = []
    for tok in proc.stdout.decode(errors="replace").split():
        tok = tok.strip().strip(",")
        if tok.isdigit():
            out.append(int(tok))
    return out


class Cv2VideoBaseline(BaselineCodec):
    """H.264/H.265 baseline through ``cv2.VideoWriter`` (whole-file size).

    Kept because it is the only video path that needs no ffmpeg CLI. The rate
    it reports is the whole container size divided by the frame count, which
    includes muxer overhead and therefore overstates payload; that is recorded
    in ``rate_measurement`` so nobody mistakes it for coded bytes.
    """

    kind = "video"
    param_label = "quality"
    rate_measurement = "container_file_bytes_div_nframes"
    param_is_integral = True

    def __init__(self, name: str, fourcc: str, container: str, notes: str = "") -> None:
        self.name = name
        self.fourcc = fourcc
        self.container = container
        self.notes = notes or (
            f"cv2.VideoWriter fourcc={fourcc}; rate = container bytes / frames "
            "(includes muxer overhead, overstates payload)"
        )

    def rate_ordered_params(self, step: float = 1.0) -> List[float]:
        return [float(p) for p in (95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10)]

    def run(self, param, frames, want_decode: bool = True):
        import cv2

        if not frames:
            return 0, []
        h, w = frames[0].shape[0], frames[0].shape[1]
        with tempfile.TemporaryDirectory(prefix="lydlr_bench_") as tmp:
            path = os.path.join(tmp, f"clip.{self.container}")
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*self.fourcc), 10.0, (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError(
                    f"{self.name}: cv2.VideoWriter could not open fourcc={self.fourcc}"
                )
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise RuntimeError(f"{self.name}: cv2.VideoWriter produced no output")
            total = int(os.path.getsize(path))
            decoded: List[np.ndarray] = []
            if want_decode:
                cap = cv2.VideoCapture(path)
                got: List[np.ndarray] = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    got.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                if len(got) != len(frames):
                    raise RuntimeError(
                        f"{self.name}: decoded {len(got)} frames, expected {len(frames)}"
                    )
                decoded = got
        return total, decoded


# ---- optional learned codecs (compressai) --------------------------------


def compressai_available() -> Tuple[bool, str]:
    """Probe for compressai WITHOUT importing it eagerly at module import time."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("compressai")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if spec is None:
        return False, "compressai not installed (optional learned-codec baseline)"
    return True, "ok"


class CompressaiBaseline(BaselineCodec):
    """Published learned-codec baseline (compressai zoo). Optional.

    compressai is never a hard dependency: it is imported inside
    ``run``/``ensure`` and every failure degrades to "unavailable with a
    reason". Quantile levels are the rate knob, ordered by DESCENDING quantile
    (higher quality level = lower rate), matching how the zoo is indexed.
    """

    kind = "learned"
    param_label = "quality_index"
    param_is_integral = True
    rate_measurement = "packed_string_bytes"
    notes = "compressai zoo; weights download on first use"

    def __init__(self, name: str, zoo_key: str, levels: Sequence[int]) -> None:
        self.name = name
        self.zoo_key = zoo_key
        self._levels = list(levels)
        self._cache: Dict[str, object] = {}

    def rate_ordered_params(self, step: float = 1.0) -> List[float]:
        return sorted((float(p) for p in self._levels), key=lambda p: -p)

    def _model(self):
        if self.zoo_key in self._cache:
            return self._cache[self.zoo_key]
        import compressai.zoo as zoo

        kwargs = {}
        if self.zoo_key in ("cheng2020", "cheng2020-anchor"):
            kwargs["anchor_mode"] = "seq-anchor"
        net = getattr(zoo, self.zoo_key)(pretrained=True, **kwargs)
        net.eval()
        self._cache[self.zoo_key] = net
        return net

    def run(self, param, frames, want_decode: bool = True):
        import torch

        net = self._model()
        total = 0
        decoded: List[np.ndarray] = []
        for f in frames:
            x = torch.from_numpy(to_float01(f)).permute(2, 0, 1)[None]
            with torch.no_grad():
                out = net.compress(x)
                payload = out["strings"][0]
                total += len(payload)
                if want_decode:
                    rec = net.decompress(out["strings"], out["shape"])
                    rec_img = rec["x_hat"][0].clamp(0, 1).permute(1, 2, 0).numpy()
                    decoded.append((rec_img * 255.0).round().astype(np.uint8))
        return total, decoded


# ---------------------------------------------------------------------------
# registry / availability
# ---------------------------------------------------------------------------

VIDEO_SPECS = (
    ("h264_intra", "libx264", "intra"),
    ("h265_intra", "libx265", "intra"),
    ("vp9_intra", "libvpx-vp9", "intra"),
    ("av1_intra", "libaom-av1", "intra"),
    ("h264_gop", "libx264", "gop"),
    ("h265_gop", "libx265", "gop"),
)

CV2_VIDEO_SPECS = (
    ("h264_cv2", ("a", "v", "c", "1"), "mp4"),
    ("h265_cv2", ("h", "e", "v", "c"), "mp4"),
    ("h264x_cv2", ("X", "2", "6", "4"), "mp4"),
)

COMPRESSAI_SPECS = (
    ("bmshj2018_hyperprior", "bmshj2018-hyperprior", (1, 2, 3, 4, 5, 6, 7, 8)),
    ("mbt2018", "mbt2018", (1, 2, 3, 4, 5, 6, 7, 8)),
    ("cheng2020_anchor", "cheng2020-anchor", (1, 2, 3, 4, 5, 6)),
)


def build_registry(
    *,
    include: Optional[Iterable[str]] = None,
    include_video: bool = True,
    include_learned: bool = False,
    include_cv2_video: bool = True,
    pix_fmt: str = "yuv444p",
) -> Tuple[Dict[str, BaselineCodec], Dict[str, str]]:
    """Return (available codecs by name, unavailable name -> reason).

    Nothing here raises because of a missing optional dependency; a missing
    dependency is data, not an error.
    """
    want = set(include) if include is not None else None

    def enabled(name: str) -> bool:
        return want is None or name in want

    codecs: Dict[str, BaselineCodec] = {}
    unavailable: Dict[str, str] = {}

    if enabled("raw_rgb24"):
        codecs["raw_rgb24"] = RawBaseline()
    if enabled("zlib_rgb24"):
        codecs["zlib_rgb24"] = ZlibBaseline()

    pil_ok, pil_reason = _pil_available()
    if not pil_ok:
        for n in ("jpeg444", "jpeg420", "webp", "png"):
            if enabled(n):
                unavailable[n] = f"Pillow unavailable: {pil_reason}"
    else:
        jpeg_levels = [float(q) for q in range(1, 101)]
        if enabled("jpeg444"):
            codecs["jpeg444"] = PILImageBaseline(
                "jpeg444", "JPEG", "quality",
                lambda q: {"quality": int(q), "subsampling": 0},
                jpeg_levels,
                notes="Pillow JPEG, 4:4:4 (no chroma subsampling)",
            )
        if enabled("jpeg420"):
            codecs["jpeg420"] = PILImageBaseline(
                "jpeg420", "JPEG", "quality",
                lambda q: {"quality": int(q), "subsampling": 2},
                jpeg_levels,
                notes="Pillow JPEG, 4:2:0 (libjpeg default; the weaker but "
                      "deployment-realistic baseline)",
            )
        if enabled("webp"):
            codecs["webp"] = PILImageBaseline(
                "webp", "WEBP", "quality",
                lambda q: {"quality": int(q)},
                [float(q) for q in range(1, 101)],
                notes="Pillow WebP (VP8 lossy)",
            )
        if enabled("png"):
            codecs["png"] = PILImageBaseline(
                "png", "PNG", "compress_level",
                lambda q: {"compress_level": int(q), "optimize": False},
                [float(q) for q in range(0, 10)],
                notes="Pillow PNG; lossless, so its bpp floor is the entropy of "
                      "the pixels (a ceiling no lossy codec is measured against)",
            )

    cv2_ok, cv2_reason = _cv2_available()
    if not cv2_ok:
        for n in ("jpeg_cv2", "webp_cv2"):
            if enabled(n):
                unavailable[n] = f"opencv unavailable: {cv2_reason}"
    else:
        if enabled("jpeg_cv2"):
            codecs["jpeg_cv2"] = Cv2ImageBaseline(
                "jpeg_cv2", ".jpg",
                lambda q: [__import__("cv2").IMWRITE_JPEG_QUALITY, int(q)],
                [float(q) for q in range(1, 101)],
                notes="cv2.imencode JPEG (libjpeg default 4:2:0); cross-check "
                      "backend for jpeg420",
            )
        if enabled("webp_cv2"):
            codecs["webp_cv2"] = Cv2ImageBaseline(
                "webp_cv2", ".webp",
                lambda q: [__import__("cv2").IMWRITE_WEBP_QUALITY, int(q)],
                [float(q) for q in range(1, 101)],
                notes="cv2.imencode WebP; cross-check backend for webp",
            )

    if include_video and (enabled("h264_intra") or enabled("h265_intra")):
        ff_ok, ff_reason = ffmpeg_available()
        if not ff_ok:
            for name, _enc, _mode in VIDEO_SPECS:
                if enabled(name):
                    unavailable[name] = ff_reason
        else:
            for name, enc, mode in VIDEO_SPECS:
                if not enabled(name):
                    continue
                if not _ffmpeg_has_encoder(enc):
                    unavailable[name] = f"ffmpeg built without encoder {enc}"
                    continue
                codecs[name] = FFmpegVideoBaseline(name, enc, mode=mode, pix_fmt=pix_fmt)

    if include_cv2_video:
        if not cv2_ok:
            for name, _cc, _ext in CV2_VIDEO_SPECS:
                if enabled(name):
                    unavailable[name] = f"opencv unavailable: {cv2_reason}"
        else:
            probe_frames = [np.zeros((32, 32, 3), dtype=np.uint8)]
            for name, cc, ext in CV2_VIDEO_SPECS:
                if not enabled(name):
                    continue
                try:
                    codecs[name] = Cv2VideoBaseline(name, cc, ext)
                    codecs[name].run(60.0, probe_frames, want_decode=False)
                    del codecs[name]
                    unavailable[name] = (
                        f"cv2.VideoWriter fourcc={''.join(cc)} is not a real H.264/"
                        f"H.265 encoder in this OpenCV build (it is a raw/mjpeg "
                        f"tag, not the codec)"
                    )
                except Exception as exc:
                    codecs.pop(name, None)
                    unavailable[name] = f"cv2.VideoWriter {''.join(cc)} failed: {exc}"

    if include_learned:
        ok, reason = compressai_available()
        if not ok:
            for name, _key, _lv in COMPRESSAI_SPECS:
                if enabled(name):
                    unavailable[name] = reason
        else:
            for name, key, levels in COMPRESSAI_SPECS:
                if not enabled(name):
                    continue
                try:
                    probe = CompressaiBaseline(name, key, levels)
                    probe.run(float(levels[0]), [np.zeros((64, 64, 3), dtype=np.uint8)])
                    codecs[name] = probe
                except Exception as exc:
                    codecs.pop(name, None)
                    unavailable[name] = f"compressai {key} unavailable: {exc}"

    return codecs, unavailable


# ---------------------------------------------------------------------------
# rate matching
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    codec: str
    target_bpp: float
    param: float
    bytes_total: int
    bpp: float
    rel_err: float
    matched: bool
    reason: str
    iterations: int


def codec_rate(
    codec: BaselineCodec, param: float, frames: Sequence[np.ndarray]
) -> int:
    """Encoded payload bytes only (no decode) — the cheap inner loop."""
    return int(codec.run(float(param), frames, want_decode=False)[0])


def _clip_bpp(nbytes: int, frames: Sequence[np.ndarray]) -> float:
    pixels = sum(int(f.shape[0] * f.shape[1]) for f in frames)
    return (nbytes * 8.0) / max(pixels, 1)


def match_rate(
    codec: BaselineCodec,
    frames: Sequence[np.ndarray],
    target_bpp: float,
    *,
    tol: float = 0.10,
    max_iter: int = 14,
) -> MatchResult:
    """Find the quality parameter whose encoded rate lands nearest ``target_bpp``.

    Method: the codec's parameter grid is ordered by ASCENDING encoded rate
    (for CRF-style knobs that means descending parameter value). We first walk
    the grid once to find the bracketing pair ``(lo, hi)`` with
    ``bpp(lo) <= target <= bpp(hi)``, then bisect that interval, halving it each
    step, until the relative rate error is within ``tol`` or the interval holds
    no further grid points. Every probe is memoised, so an N-point grid costs at
    most N encodes no matter how many iterations run.

    Why 10% is the default tolerance: at these operating points a one-step JPEG
    quality change moves bpp by roughly 1-2%, and a 10% rate gap between two
    codecs moves PSNR by less than the frame-to-frame spread this harness
    reports. A conclusion that flips inside a 10% rate window is not a
    conclusion. Anything that cannot land inside the window is reported
    ``matched=False`` with the reason, never quietly rounded into the table.
    """
    if not frames:
        return MatchResult(codec.name, target_bpp, 0.0, 0, 0.0, 1.0, False,
                           "no frames", 0)
    grid = codec.rate_ordered_params()
    if not grid:
        return MatchResult(codec.name, target_bpp, 0.0, 0, 0.0, 1.0, False,
                           "codec exposes no quality parameter", 0)

    measured: Dict[int, Tuple[float, int]] = {}

    def rel_of(bpp: float) -> float:
        return abs(bpp - target_bpp) / max(abs(target_bpp), 1e-9)

    def probe(idx: float) -> Tuple[int, float, int]:
        key = max(0, min(int(round(idx)), len(grid) - 1))
        if key not in measured:
            nbytes = codec_rate(codec, grid[key], frames)
            measured[key] = (_clip_bpp(nbytes, frames), nbytes)
        bpp, nbytes = measured[key]
        return key, bpp, nbytes

    _, bpp_lo, bytes_lo = probe(0)
    _, bpp_hi, bytes_hi = probe(len(grid) - 1)
    bpp_hi = max(bpp_hi, bpp_lo)

    if target_bpp <= bpp_lo:
        return MatchResult(codec.name, target_bpp, float(grid[0]), bytes_lo, bpp_lo,
                           rel_of(bpp_lo), rel_of(bpp_lo) <= tol,
                           f"target below codec minimum rate ({bpp_lo:.6f} bpp); "
                           "reported at minimum rate", len(measured))
    if target_bpp >= bpp_hi:
        top = float(grid[len(grid) - 1])
        return MatchResult(codec.name, target_bpp, top, bytes_hi, bpp_hi,
                           rel_of(bpp_hi), rel_of(bpp_hi) <= tol,
                           f"target above codec maximum rate ({bpp_hi:.6f} bpp); "
                           "reported at maximum rate", len(measured))

    lo = 0.0
    hi = float(len(grid) - 1)
    for idx in range(1, len(grid)):
        _, bpp, _nbytes = probe(idx)
        if bpp >= target_bpp:
            lo, hi = float(idx - 1), float(idx)
            break

    for _ in range(max_iter):
        if hi - lo <= 1.0:
            break
        mid = 0.5 * (lo + hi)
        _, bpp, _nbytes = probe(mid)
        if rel_of(bpp) <= tol:
            break
        if bpp < target_bpp:
            lo = mid
        else:
            hi = mid

    best_key = min(measured, key=lambda k: rel_of(measured[k][0]))
    bpp_best, bytes_best = measured[best_key]
    rel = rel_of(bpp_best)
    return MatchResult(
        codec=codec.name,
        target_bpp=float(target_bpp),
        param=float(grid[best_key]),
        bytes_total=int(bytes_best),
        bpp=float(bpp_best),
        rel_err=float(rel),
        matched=bool(rel <= tol),
        reason="" if rel <= tol else f"rate search exhausted at rel_err={rel:.4f}",
        iterations=len(measured),
    )


def measure_point(
    codec: BaselineCodec,
    param: float,
    frames: Sequence[np.ndarray],
    *,
    lpips: Optional[LpipsProbe] = None,
) -> OperatingPoint:
    """Encode, decode, and score one operating point of ``codec``."""
    nbytes, decoded = codec.run(float(param), frames, want_decode=True)
    if len(decoded) != len(frames):
        raise RuntimeError(
            f"{codec.name}: decoded {len(decoded)} frames, expected {len(frames)}"
        )
    ps: List[float] = []
    ss: List[float] = []
    lp: List[float] = []
    for src, rec in zip(frames, decoded):
        if rec.shape != src.shape:
            raise RuntimeError(
                f"{codec.name}: recon shape {rec.shape} != source {src.shape}"
            )
        ps.append(psnr(src, rec))
        ss.append(ssim(src, rec))
        if lpips is not None:
            lp.append(lpips.score(src, rec))
    n = len(ps)
    pixels = sum(int(f.shape[0] * f.shape[1]) for f in frames)
    lp_mean = float(np.mean(lp)) if lp and all(math.isfinite(v) for v in lp) else float("nan")
    return OperatingPoint(
        codec=codec.name,
        param=float(param),
        bytes_total=int(nbytes),
        n_frames=n,
        pixels_per_frame=pixels // max(len(frames), 1),
        bpp=(nbytes * 8.0) / max(pixels, 1),
        psnr_mean=float(np.mean(ps)),
        psnr_std=float(np.std(ps)),
        psnr_min=float(np.min(ps)),
        psnr_max=float(np.max(ps)),
        ssim_mean=float(np.mean(ss)),
        ssim_std=float(np.std(ss)),
        lpips_mean=lp_mean,
        rate_measurement=codec.rate_measurement,
        extras={"param_label": codec.param_label, "notes": codec.notes},
    )


def sweep_codec(
    codec: BaselineCodec,
    frames: Sequence[np.ndarray],
    *,
    max_points: int = 13,
    lpips: Optional[LpipsProbe] = None,
) -> List[OperatingPoint]:
    """Full RD curve: evenly spaced indices across the codec's rate-ordered grid."""
    grid = codec.rate_ordered_params()
    if not grid:
        return []
    k = max(2, min(max_points, len(grid)))
    idxs = np.unique(np.linspace(0, len(grid) - 1, k).round().astype(int))
    return [measure_point(codec, grid[i], frames, lpips=lpips) for i in idxs]


def interp_rd(points: Sequence[OperatingPoint], target_bpp: float) -> Optional[dict]:
    """Interpolate PSNR/SSIM at ``target_bpp`` along a measured RD curve.

    Interpolation is done in (log bpp, PSNR) space, which is the near-linear
    region of a typical RD curve. Returns ``None`` when the target falls
    outside the measured range, so callers can never quote an extrapolated
    number as a measurement.
    """
    usable = [p for p in points if p.bpp > 0]
    if len(usable) < 2:
        return None
    usable = sorted(usable, key=lambda p: p.bpp)
    xs = np.log(np.array([p.bpp for p in usable]))
    if target_bpp <= usable[0].bpp * 1.001 or target_bpp >= usable[-1].bpp * 0.999:
        return None
    lx = math.log(target_bpp)
    psnr_at = float(np.interp(lx, xs, np.array([p.psnr_mean for p in usable])))
    ssim_at = float(np.interp(lx, xs, np.array([p.ssim_mean for p in usable])))
    return {
        "bpp": float(target_bpp),
        "psnr": psnr_at,
        "ssim": ssim_at,
        "method": "log-bpp linear interpolation between measured points",
        "in_range": True,
    }


# ---------------------------------------------------------------------------
# Lydlr adapter
# ---------------------------------------------------------------------------

LYDLR_ENTROPY_MODULE = "lydlr_ai.model.entropy_coder"
LYDLR_ENTROPY_FUNCTION_ALIASES = (
    "rans_encode",
    "encode_indices",
    "rans_encode_indices",
    "rans_compress_indices",
    "encode_with_probabilities",
)
LYDLR_ENTROPY_CLASS_NAMES = ("RansCoder", "RansEncoder", "EntropyCoder", "Rans")


def _import_lydlr_entropy():
    """Import the project's rANS module if it exists. Returns (module, reason)."""
    try:
        import importlib

        mod = importlib.import_module(LYDLR_ENTROPY_MODULE)
        return mod, "ok"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _resolve_rans_callable():
    """Find a callable matching the documented pluggable contract.

    Returns (callable, description, reason). A module that exists but exposes
    no recognisable entry point is reported as a reason, not as success.
    """
    mod, reason = _import_lydlr_entropy()
    if mod is None:
        return None, "", f"entropy coder not importable ({LYDLR_ENTROPY_MODULE}): {reason}"
    for fname in LYDLR_ENTROPY_FUNCTION_ALIASES:
        fn = getattr(mod, fname, None)
        if callable(fn):
            return fn, f"{LYDLR_ENTROPY_MODULE}.{fname}", "ok"
    for cname in LYDLR_ENTROPY_CLASS_NAMES:
        cls = getattr(mod, cname, None)
        if cls is None:
            continue
        try:
            inst = cls()
        except Exception as exc:
            return None, "", f"{cname} present but not constructible: {exc}"
        fn = getattr(inst, "encode", None)
        if callable(fn):
            return fn, f"{LYDLR_ENTROPY_MODULE}.{cname}().encode", "ok"
    return None, "", (
        f"{LYDLR_ENTROPY_MODULE} imports but exposes none of "
        f"{LYDLR_ENTROPY_FUNCTION_ALIASES} or {LYDLR_ENTROPY_CLASS_NAMES}"
    )


def empirical_order0_bits(indices: np.ndarray) -> float:
    """Order-0 entropy of the symbol histogram, in bits.

    This is the Shannon lower bound on any coder for these symbols with no
    context modelling (TRUE_RATE_APPLIED_MATH: R_wire >= H(q)). It is NOT a
    wire rate — it is reported as the gap the missing coder would have to
    close, and it is never placed in a ``bits`` field.
    """
    flat = np.asarray(indices).reshape(-1).astype(np.int64)
    if flat.size == 0:
        return 0.0
    counts = np.bincount(flat)
    probs = counts[counts > 0] / float(flat.size)
    return float(-(probs * np.log2(probs)).sum() * flat.size)


def _prepare_side_inputs(
    side: Optional[List[dict]], height: int, width: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if not side:
        return None, None, None
    l, i, a = [], [], []
    for rec in side:
        if "lidar" in rec:
            v = np.asarray(rec["lidar"], dtype=np.float32).reshape(-1)
            l.append(v)
        if "imu" in rec:
            i.append(np.asarray(rec["imu"], dtype=np.float32).reshape(-1))
        if "audio" in rec:
            a.append(np.asarray(rec["audio"], dtype=np.float32).reshape(-1))
    if not l:
        return None, None, None
    n = min(len(l), len(i), len(a))
    return (
        np.stack(l[:n]),
        np.stack(i[:n]),
        np.stack(a[:n]),
    )


def _lidar_for_model(lidar: np.ndarray) -> np.ndarray:
    """Reshape a flat LiDAR vector to the (B, 3, P) layout the encoder expects.

    The compressor's lidar encoder consumes ``3 * lidar_dim`` inputs, i.e. a
    three-component point (x, y, z). The stored fixtures keep a flat vector, so
    we split it into three contiguous channels and edge-pad the last partial
    point. This is a shape adapter only: it invents no values and it does not
    affect any image rate or distortion number.
    """
    v = np.asarray(lidar, dtype=np.float32).reshape(-1)
    points = max(1, (v.size + 2) // 3)
    if v.size < 3 * points:
        v = np.pad(v, (0, 3 * points - v.size), mode="edge")
    return v[: 3 * points].reshape(1, 3, points)


def lydlr_rate_and_recon(
    frames: Sequence[np.ndarray],
    *,
    side_inputs: Optional[List[dict]] = None,
    checkpoint: str = "",
    target_qualities: Sequence[float] = (0.2, 0.5, 0.8),
    device: Optional[str] = None,
    seed: int = 0,
    lpips_probe: Optional[LpipsProbe] = None,
    latency: bool = True,
) -> dict:
    """Measure the Lydlr path: countable bits per frame plus reconstruction.

    Returns a dict with:

    - ``points``: one entry per ``target_qualities`` value, each with
      ``bits`` (COUNTABLE), ``bpp``, ``psnr``, ``ssim``, ``rate_source``,
      ``fixed_length_bits`` and ``proxy_bits``.
    - ``trained``: ``False`` when no checkpoint was loaded. Every number
      produced under ``trained=False`` is random-weight output and is not a
      codec claim; the flag is propagated into the report so it cannot be lost.
    - ``rate_source``: which of the two honest rate paths was taken.
    - ``available`` / ``reason``: ``False`` plus a reason when the model or
      torch is missing, instead of an exception.

    Rate policy: ``bits`` is the byte length of a real coded payload when the
    rANS coder is importable and callable. Otherwise ``bits`` is
    ``fixed_length_bits = 8 * d`` — literally what a decoder has to read with no
    entropy coding — and ``rate_source`` says so. ``proxy_bits`` (the
    differentiable cross-entropy) is reported beside it and is never assigned
    to ``bits``.
    """
    import time

    n = len(frames)
    out: dict = {
        "available": False,
        "reason": "",
        "trained": False,
        "rate_source": None,
        "points": [],
        "latent_dim": None,
        "num_levels": 256,
        "checkpoint": checkpoint or "",
        "device": "",
    }
    try:
        import torch
    except Exception as exc:
        out["reason"] = f"torch unavailable: {exc}"
        return out
    try:
        import sys
        import os

        here = os.path.dirname(os.path.abspath(__file__))
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
        if pkg_root not in sys.path:
            sys.path.insert(0, pkg_root)
        from lydlr_ai.model.compressor import EnhancedMultimodalCompressor, unpack_compressor_output
    except Exception as exc:
        out["reason"] = f"compressor import failed: {type(exc).__name__}: {exc}"
        return out

    h, w = int(frames[0].shape[0]), int(frames[0].shape[1])
    lat, imu, aud = _prepare_side_inputs(side_inputs, h, w)
    if lat is None:
        out["reason"] = (
            "no LiDAR/IMU/audio side inputs: EnhancedMultimodalCompressor is a "
            "4-input multimodal model and cannot be evaluated from the image alone"
        )
        return out

    dev = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out["device"] = str(dev)
    torch.manual_seed(seed)
    lidar_rows = [_lidar_for_model(lat[i])[0] for i in range(lat.shape[0])]
    lidar_dim = int(lidar_rows[0].shape[-1])
    audio_dim = int(aud.shape[-1])
    model = EnhancedMultimodalCompressor(
        image_shape=(3, h, w), lidar_dim=lidar_dim, audio_dim=audio_dim
    ).to(dev)
    trained = False
    if checkpoint:
        try:
            blob = torch.load(checkpoint, map_location=dev)
            state = blob.get("model_state_dict", blob) if isinstance(blob, dict) else blob
            model.load_state_dict(state, strict=False)
            trained = True
        except Exception as exc:
            out["reason"] = f"checkpoint load failed ({exc}); evaluating random weights"
    model.eval()

    rans_fn, rans_desc, rans_reason = _resolve_rans_callable()
    out["entropy_coder"] = {"available": rans_fn is not None, "entry": rans_desc,
                            "reason": rans_reason}
    rate_source = RANS if rans_fn is not None else FIXED_LENGTH
    out["rate_source"] = rate_source
    out["trained"] = trained

    lat_t = torch.from_numpy(np.ascontiguousarray(np.stack(lidar_rows))).to(dev)
    imu_t = torch.from_numpy(np.ascontiguousarray(imu)).to(dev)
    aud_t = torch.from_numpy(np.ascontiguousarray(aud)).to(dev)
    pixels = int(h * w)

    for tq in target_qualities:
        model.reset_temporal_state()
        acc_bits = 0.0
        acc_proxy = 0.0
        acc_fixed = 0.0
        acc_entropy = 0.0
        ps: List[float] = []
        ss: List[float] = []
        lp: List[float] = []
        rans_used = 0
        rans_errors: List[str] = []
        lat_ms: List[float] = []
        latent_dim = None
        for idx, frame in enumerate(frames):
            img = torch.from_numpy(
                (frame.astype(np.float32) / 255.0).transpose(2, 0, 1)
            )[None].to(dev)
            t0 = time.perf_counter()
            with torch.no_grad():
                packed = unpack_compressor_output(
                    model(
                        img,
                        lat_t[idx : idx + 1],
                        imu_t[idx : idx + 1],
                        aud_t[idx : idx + 1],
                        target_quality=float(tq),
                    )
                )
            if dev.type == "cuda":
                torch.cuda.synchronize()
            if latency:
                lat_ms.append((time.perf_counter() - t0) * 1000.0)

            rec = packed["recon_img"].clamp(0, 1)
            if tuple(rec.shape[-2:]) != (h, w):
                rec = torch.nn.functional.interpolate(
                    rec, size=(h, w), mode="bilinear", align_corners=False
                )
            rec_np = rec[0].cpu().numpy().transpose(1, 2, 0)
            ps.append(psnr(frame, rec_np))
            ss.append(ssim(frame, rec_np))
            if lpips_probe is not None:
                lp.append(lpips_probe.score(frame, rec_np))

            proxy = float(packed["rate_bits"].mean().detach().cpu()) if (
                packed["rate_bits"] is not None and packed["rate_bits"].numel()
            ) else 0.0
            acc_proxy += proxy

            indices = packed.get("quant_indices")
            if indices is None:
                out["reason"] = "compressor returned no quant_indices; no rate to report"
                return out
            idx_np = indices.detach().cpu().numpy().reshape(-1).astype(np.int64)
            latent_dim = int(idx_np.size)
            levels = int(idx_np.max()) + 1
            fixed_bits = float(latent_dim * max(1, math.ceil(math.log2(max(levels, 2)))))
            acc_fixed += fixed_bits
            acc_entropy += empirical_order0_bits(idx_np)

            if rans_fn is not None:
                probs = _lydlr_symbol_probs(model, packed, levels, dev)
                ok, payload = _try_rans(rans_fn, idx_np, probs, rans_errors)
                if ok:
                    acc_bits += float(len(payload) * 8)
                    rans_used += 1
                else:
                    acc_bits += fixed_bits
            else:
                acc_bits += fixed_bits

        if rate_source == RANS and rans_used == 0 and rans_errors:
            out["entropy_coder"] = {
                "available": False,
                "entry": rans_desc,
                "reason": rans_errors[0],
            }
            out["rate_source"] = FIXED_LENGTH
        used_source = RANS if (rate_source == RANS and rans_used == n) else (
            FIXED_LENGTH if rans_used < n else RANS
        )
        nbytes = acc_bits / 8.0
        out["points"].append({
            "target_quality": float(tq),
            "bits": float(acc_bits / max(len(frames), 1)),
            "bytes_per_frame": float(nbytes / max(len(frames), 1)),
            "bpp": float(nbytes * 8.0 / (pixels * max(len(frames), 1))),
            "rate_source": used_source,
            "fixed_length_bits": float(acc_fixed / max(len(frames), 1)),
            "empirical_entropy_bits": float(acc_entropy / max(len(frames), 1)),
            "proxy_bits": float(acc_proxy / max(len(frames), 1)),
            "trained": bool(trained),
            "latent_dim": latent_dim,
            "psnr_mean": float(np.mean(ps)),
            "psnr_std": float(np.std(ps)),
            "psnr_min": float(np.min(ps)),
            "psnr_max": float(np.max(ps)),
            "ssim_mean": float(np.mean(ss)),
            "ssim_std": float(np.std(ss)),
            "lpips_mean": float(np.mean(lp)) if lp and all(math.isfinite(v) for v in lp)
            else float("nan"),
            "latency_ms_mean": float(np.mean(lat_ms)) if lat_ms else float("nan"),
        })
    out["latent_dim"] = out["points"][0]["latent_dim"] if out["points"] else None
    out["available"] = bool(out["points"])
    if not out["reason"]:
        out["reason"] = "ok"
    return out


def _lydlr_symbol_probs(model, packed, levels: int, dev):
    """Stationary symbol distribution for the latent, from the learned model."""
    import torch

    coder = getattr(model, "entropy_coder", None)
    if coder is None:
        raise RuntimeError("model has no entropy_coder module")
    with torch.no_grad():
        _entropy, probs = coder(packed["compressed"].float())
    p = probs[0].detach().float().cpu().numpy().astype(np.float64)
    if p.size < levels:
        p = np.pad(p, (0, levels - p.size))
    p = p[:levels]
    total = p.sum()
    if total <= 0:
        p = np.full(levels, 1.0 / max(levels, 1))
    else:
        p = p / total
    return p


def _try_rans(fn, indices: np.ndarray, probs: np.ndarray, errors: List[str]):
    try:
        payload = fn(indices.astype(np.uint8), probs)
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return True, bytes(payload)
        errors.append(f"coder returned {type(payload).__name__}, expected bytes")
        return False, b""
    except Exception as exc:
        errors.append(f"coder raised {type(exc).__name__}: {exc}")
        return False, b""
