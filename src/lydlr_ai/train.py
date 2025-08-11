# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# ~/lydlr/lydlr_ws/src/lydlr_ai/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import lpips
from tqdm import tqdm

from lydlr_ai.model.compressor import MultimodalCompressor, QualityAssessor

# === Hyperparameters ===
BATCH_SIZE = 2
SEQ_LEN = 4
EPOCHS = 10
LR = 1e-4
LATENT_WEIGHT = 1e-4  # rate-distortion weight
SAVE_PATH = "multimodal_compressor.pth"

# === Dummy Dataset Generator ===
def generate_dummy_sequence(batch_size, seq_len):
    image = torch.rand(batch_size, seq_len, 3, 480, 640)
    lidar = torch.rand(batch_size, seq_len, 1024)
    imu = torch.rand(batch_size, seq_len, 6)
    audio = torch.rand(batch_size, seq_len, 128 * 128)
    return image, lidar, imu, audio

# === Training ===
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MultimodalCompressor().to(device)
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        image, lidar, imu, audio = generate_dummy_sequence(BATCH_SIZE, SEQ_LEN)
        image, lidar, imu, audio = image.to(device), lidar.to(device), imu.to(device), audio.to(device)

        hidden_state = None
        total_loss = 0

        for t in range(SEQ_LEN):
            img_t = image[:, t]
            lidar_t = lidar[:, t]
            imu_t = imu[:, t]
            audio_t = audio[:, t]

            # Forward pass with dropout
            encoded, decoded, hidden_state, recon_img = model(
                img_t, lidar_t, imu_t, audio_t, hidden_state, compression_level=0.8
            )

            # Compute loss
            recon_loss = mse_loss(recon_img, img_t)
            perceptual = lpips_loss_fn((img_t * 2 - 1), (recon_img * 2 - 1)).mean()
            latent_bits = encoded.numel() * 32  # assume 32-bit floats
            rate_loss = LATENT_WEIGHT * latent_bits

            loss = recon_loss + perceptual + rate_loss
            total_loss += loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"[Epoch {epoch+1}/{EPOCHS}] Total Loss: {total_loss.item():.4f} | Recon: {recon_loss.item():.4f} | LPIPS: {perceptual.item():.4f} | Rate: {rate_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f" Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()
