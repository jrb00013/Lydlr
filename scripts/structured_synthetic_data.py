#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# Structured multimodal synthetic clips for RD training.
# See docs/architecture/TRAINING_DATA_APPLIED_MATH.md

"""Stateful scene process → camera / LiDAR / IMU / audio observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class SceneState:
    """Minimal Markov state for one clip (batch of independent scenes)."""

    background: torch.Tensor  # (B, 3, H, W)
    blob_pos: torch.Tensor  # (B, K, 2) in [-1, 1] normalized coords
    blob_vel: torch.Tensor  # (B, K, 2)
    blob_radius: torch.Tensor  # (B, K)
    blob_color: torch.Tensor  # (B, K, 3)
    ego_vel: torch.Tensor  # (B, 2) camera translation in normalized coords / frame
    audio_phase: torch.Tensor  # (B,)
    frames_since_cut: torch.Tensor  # (B,)


def _mesh(h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_y, grid_x


def _low_freq_background(batch: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Band-limited field: structure without white-noise spectrum."""
    # Start small and upsample for spatial correlation
    tiny = torch.rand(batch, 3, max(8, h // 32), max(8, w // 32), device=device)
    field = F.interpolate(tiny, size=(h, w), mode="bilinear", align_corners=False)
    # Add a second coarser scale
    coarse = torch.rand(batch, 3, 4, 4, device=device)
    coarse = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
    bg = 0.55 * field + 0.35 * coarse + 0.05
    return bg.clamp(0, 1)


def init_scene(
    batch_size: int,
    device: torch.device,
    *,
    height: int = 480,
    width: int = 640,
    num_blobs: int = 5,
) -> SceneState:
    bg = _low_freq_background(batch_size, height, width, device)
    k = num_blobs
    pos = torch.empty(batch_size, k, 2, device=device).uniform_(-0.7, 0.7)
    vel = torch.empty(batch_size, k, 2, device=device).uniform_(-0.02, 0.02)
    radius = torch.empty(batch_size, k, device=device).uniform_(0.06, 0.18)
    color = torch.empty(batch_size, k, 3, device=device).uniform_(0.15, 0.95)
    ego = torch.empty(batch_size, 2, device=device).uniform_(-0.015, 0.015)
    phase = torch.rand(batch_size, device=device) * 6.2832
    since = torch.zeros(batch_size, device=device)
    return SceneState(bg, pos, vel, radius, color, ego, phase, since)


def _render_rgb(state: SceneState) -> torch.Tensor:
    b, _, h, w = state.background.shape
    device = state.background.device
    grid_y, grid_x = _mesh(h, w, device)
    img = state.background.clone()

    # Ego translation as a circular shift in normalized coords via grid_sample
    # Build sampling grid shifted by -ego (camera moves → scene opposite)
    # grid_sample expects (N, H, W, 2) with x,y in [-1,1]
    ego = state.ego_vel  # (B, 2)
    ones_y = grid_y.unsqueeze(0).expand(b, -1, -1)
    ones_x = grid_x.unsqueeze(0).expand(b, -1, -1)
    samp_x = (ones_x - ego[:, 0].view(b, 1, 1)).clamp(-1, 1)
    samp_y = (ones_y - ego[:, 1].view(b, 1, 1)).clamp(-1, 1)
    grid = torch.stack([samp_x, samp_y], dim=-1)
    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)

    # Soft blobs (translation-equivariant local structure)
    for k in range(state.blob_pos.size(1)):
        cx = state.blob_pos[:, k, 0].view(b, 1, 1) - ego[:, 0].view(b, 1, 1)
        cy = state.blob_pos[:, k, 1].view(b, 1, 1) - ego[:, 1].view(b, 1, 1)
        r = state.blob_radius[:, k].view(b, 1, 1).clamp_min(1e-3)
        dist2 = (ones_x - cx) ** 2 + (ones_y - cy) ** 2
        mask = torch.exp(-dist2 / (2 * r * r))
        col = state.blob_color[:, k, :].view(b, 3, 1, 1)
        img = img * (1 - 0.85 * mask.unsqueeze(1)) + col * (0.85 * mask.unsqueeze(1))

    # Mild sensor noise (not dominant energy)
    img = (img + 0.01 * torch.randn_like(img)).clamp(0, 1)
    return img


def _lidar_from_scene(state: SceneState, n_points: int = 1024) -> torch.Tensor:
    """Noisy 3D points: ground plane + blob centers as obstacles (shared geometry)."""
    b, k, _ = state.blob_pos.shape
    device = state.blob_pos.device
    # Ground points
    n_ground = n_points // 2
    gx = torch.empty(b, n_ground, device=device).uniform_(-1, 1)
    gy = torch.empty(b, n_ground, device=device).uniform_(-1, 1)
    gz = 0.02 * torch.randn(b, n_ground, device=device)
    ground = torch.stack([gx, gy, gz], dim=-1)

    # Obstacle points around blobs
    n_obs = n_points - n_ground
    idx = torch.randint(0, k, (b, n_obs), device=device)
    # gather blob centers
    batch_idx = torch.arange(b, device=device).unsqueeze(1).expand_as(idx)
    centers = state.blob_pos[batch_idx, idx]  # (B, n_obs, 2)
    radii = state.blob_radius[batch_idx, idx].unsqueeze(-1)
    jitter = 0.15 * radii * torch.randn(b, n_obs, 2, device=device)
    xy = centers + jitter - state.ego_vel.unsqueeze(1)
    z = 0.3 + 0.5 * radii.squeeze(-1) + 0.05 * torch.randn(b, n_obs, device=device)
    obs = torch.stack([xy[..., 0], xy[..., 1], z], dim=-1)
    return torch.cat([ground, obs], dim=1)


def _imu_from_ego(prev_ego: Optional[torch.Tensor], ego: torch.Tensor) -> torch.Tensor:
    """6-vector: accel(~Δv), gyro proxy, and small bias noise."""
    b = ego.size(0)
    device = ego.device
    if prev_ego is None:
        accel = torch.zeros(b, 2, device=device)
    else:
        accel = (ego - prev_ego) * 50.0  # scale to noticeable units
    gyro_z = (ego[:, 0] - ego[:, 1]).unsqueeze(-1) * 5.0
    pad = torch.zeros(b, 1, device=device)
    imu = torch.cat([accel, pad, gyro_z, pad, pad], dim=-1)  # (B, 6)
    return imu + 0.02 * torch.randn_like(imu)


def _audio_from_motion(state: SceneState, mel_bins: int = 128) -> torch.Tensor:
    """Slow envelope + motion coupling → flattened mel-sized vector."""
    b = state.audio_phase.size(0)
    device = state.audio_phase.device
    speed = state.ego_vel.norm(dim=-1) + state.blob_vel.norm(dim=-1).mean(dim=-1)
    t = torch.linspace(0, 1, mel_bins * mel_bins, device=device)
    phase = state.audio_phase.view(b, 1)
    wave = 0.4 + 0.3 * torch.sin(6.2832 * (2 + 8 * speed.view(b, 1)) * t + phase)
    wave = wave + 0.05 * torch.randn_like(wave)
    return wave.clamp(0, 1)


def step_scene(
    state: SceneState,
    *,
    cut_prob: float = 0.03,
) -> Tuple[SceneState, dict]:
    """Advance Markov state; occasional cuts resample the background/blobs."""
    b = state.background.size(0)
    device = state.background.device
    prev_ego = state.ego_vel.clone()

    # Random cuts (adversarial temporal case)
    cut = torch.rand(b, device=device) < cut_prob
    if cut.any():
        fresh = init_scene(
            int(cut.sum().item()),
            device,
            height=state.background.size(2),
            width=state.background.size(3),
            num_blobs=state.blob_pos.size(1),
        )
        # scatter fresh into cut slots
        state.background = state.background.clone()
        state.blob_pos = state.blob_pos.clone()
        state.blob_vel = state.blob_vel.clone()
        state.blob_radius = state.blob_radius.clone()
        state.blob_color = state.blob_color.clone()
        state.ego_vel = state.ego_vel.clone()
        state.audio_phase = state.audio_phase.clone()
        state.frames_since_cut = state.frames_since_cut.clone()
        idxs = cut.nonzero(as_tuple=False).squeeze(-1)
        state.background[idxs] = fresh.background
        state.blob_pos[idxs] = fresh.blob_pos
        state.blob_vel[idxs] = fresh.blob_vel
        state.blob_radius[idxs] = fresh.blob_radius
        state.blob_color[idxs] = fresh.blob_color
        state.ego_vel[idxs] = fresh.ego_vel
        state.audio_phase[idxs] = fresh.audio_phase
        state.frames_since_cut[idxs] = 0

    # Continuous dynamics
    state.blob_pos = (state.blob_pos + state.blob_vel).clamp(-0.95, 0.95)
    # bounce
    hit = state.blob_pos.abs() > 0.9
    state.blob_vel = torch.where(hit, -state.blob_vel, state.blob_vel)
    # slow ego random walk (ν ≪ 1)
    state.ego_vel = (0.9 * state.ego_vel + 0.01 * torch.randn_like(state.ego_vel)).clamp(-0.03, 0.03)
    state.audio_phase = (state.audio_phase + 0.15 + 2.0 * state.ego_vel.norm(dim=-1)) % 6.2832
    state.frames_since_cut = state.frames_since_cut + 1
    state.frames_since_cut = torch.where(cut, torch.zeros_like(state.frames_since_cut), state.frames_since_cut)

    image = _render_rgb(state)
    lidar = _lidar_from_scene(state)
    imu = _imu_from_ego(prev_ego, state.ego_vel)
    audio = _audio_from_motion(state)

    # Relative residual diagnostic vs previous render if provided externally
    meta = {
        "cut_fraction": float(cut.float().mean().item()),
        "ego_speed": float(state.ego_vel.norm(dim=-1).mean().item()),
        "frames_since_cut": float(state.frames_since_cut.mean().item()),
    }
    return state, {
        "image": image,
        "lidar": lidar,
        "imu": imu,
        "audio": audio,
        "meta": meta,
        "cut_mask": cut,
    }


def relative_residual(prev: torch.Tensor, curr: torch.Tensor) -> float:
    """Dimensionless φ = ‖x_t − x_{t−1}‖ / ‖x_t‖."""
    num = (curr - prev).float().norm().item()
    den = curr.float().norm().item() + 1e-8
    return num / den


def smoke_invariants(device: Optional[torch.device] = None) -> dict:
    """Adversarial sanity: structured smooth φ ≪ cut φ; noise φ stays high."""
    device = device or torch.device("cpu")
    state = init_scene(2, device, height=128, width=160, num_blobs=4)
    state, out0 = step_scene(state, cut_prob=0.0)
    prev = out0["image"]
    phis_smooth = []
    for _ in range(5):
        state, out = step_scene(state, cut_prob=0.0)
        phis_smooth.append(relative_residual(prev, out["image"]))
        prev = out["image"]
    # force cut
    state, out_cut = step_scene(state, cut_prob=1.0)
    phi_cut = relative_residual(prev, out_cut["image"])

    noise_a = torch.rand_like(prev)
    noise_b = torch.rand_like(prev)
    phi_noise = relative_residual(noise_a, noise_b)
    return {
        "phi_smooth_mean": sum(phis_smooth) / len(phis_smooth),
        "phi_cut": phi_cut,
        "phi_noise": phi_noise,
        "ok_smooth_lt_cut": (sum(phis_smooth) / len(phis_smooth)) < phi_cut,
        "ok_smooth_lt_noise": (sum(phis_smooth) / len(phis_smooth)) < phi_noise,
    }


if __name__ == "__main__":
    report = smoke_invariants()
    print(report)
    assert report["ok_smooth_lt_cut"], report
    assert report["ok_smooth_lt_noise"], report
    print("structured synthetic invariants OK")
