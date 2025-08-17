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

# === Data Loading ===
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from data_loader import create_data_loader

# === Training Configuration ===
DATA_DIR = "~/lydlr_ws/data/training_data"  # Path to your training data
USE_SYNTHETIC = True  # Set to False when you have real data

# === Training ===
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loader
    data_loader = create_data_loader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        sequence_length=SEQ_LEN,
        synthetic=USE_SYNTHETIC
    )
    
    print(f"Data loader created: {'Synthetic' if USE_SYNTHETIC else 'Real'} data")
    print(f"Number of batches per epoch: {len(data_loader)}")

    model = MultimodalCompressor().to(device)
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        # Training loop over batches
        for batch_idx, (image, lidar, imu, audio) in enumerate(data_loader):
            image, lidar, imu, audio = image.to(device), lidar.to(device), imu.to(device), audio.to(device)
            
            # Ensure correct shapes for LiDAR (flatten to 2D if needed)
            if len(lidar.shape) == 4:  # [B, T, N, 3]
                lidar = lidar.view(lidar.size(0), lidar.size(1), -1)  # [B, T, N*3]
            
            hidden_state = None
            batch_loss = 0

            # Process sequence timesteps
            for t in range(SEQ_LEN):
                img_t = image[:, t]
                lidar_t = lidar[:, t]
                imu_t = imu[:, t]
                audio_t = audio[:, t]

                # Forward pass
                try:
                    encoded, decoded, hidden_state, recon_img = model(
                        img_t, lidar_t, imu_t, audio_t, hidden_state, compression_level=0.8
                    )

                    # Compute loss
                    recon_loss = mse_loss(recon_img, img_t)
                    perceptual = lpips_loss_fn((img_t * 2 - 1), (recon_img * 2 - 1)).mean()
                    latent_bits = encoded.numel() * 32  # assume 32-bit floats
                    rate_loss = LATENT_WEIGHT * latent_bits

                    loss = recon_loss + perceptual + rate_loss
                    batch_loss += loss
                    
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    continue

            # Backward pass
            if batch_loss > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"[Epoch {epoch+1}/{EPOCHS}] Batch {batch_idx}/{len(data_loader)} | "
                          f"Batch Loss: {batch_loss.item():.4f} | "
                          f"Recon: {recon_loss.item():.4f} | "
                          f"LPIPS: {perceptual.item():.4f} | "
                          f"Rate: {rate_loss:.4f}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Average Loss: {avg_epoch_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()
