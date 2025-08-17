# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Training Script with Enhancements
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import lpips
from tqdm import tqdm
import json
from datetime import datetime

from lydlr_ai.model.enhanced_compressor import EnhancedMultimodalCompressor, compute_enhanced_loss

# === Enhanced Hyperparameters ===
BATCH_SIZE = 4
SEQ_LEN = 8
EPOCHS = 20
LR = 1e-4
BETA_VAE = 0.1  # β-VAE weight
TARGET_QUALITY = 0.8
SAVE_PATH = "enhanced_multimodal_compressor.pth"

# === Data Loading ===
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from scripts.data_loader import create_data_loader

# === Training Configuration ===
DATA_DIR = "~/lydlr_ws/data/training_data"
USE_SYNTHETIC = True  # Set to False when you have real data

class EnhancedTrainer:
    """Enhanced trainer with all improvements"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create enhanced model
        self.model = EnhancedMultimodalCompressor().to(self.device)
        
        # Loss functions
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.mse_loss = nn.MSELoss()
        
        # Optimizer with different learning rates for different components
        self.optimizer = optim.AdamW([
            {'params': self.model.vae.parameters(), 'lr': LR * 0.1},  # VAE: slower
            {'params': self.model.fusion.parameters(), 'lr': LR},      # Fusion: normal
            {'params': self.model.temporal_transformer.parameters(), 'lr': LR},  # Transformer: normal
            {'params': self.model.quality_controller.parameters(), 'lr': LR * 2},  # Quality: faster
        ], lr=LR, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2
        )
        
        # Data loader
        self.data_loader = create_data_loader(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            sequence_length=SEQ_LEN,
            synthetic=USE_SYNTHETIC
        )
        
        print(f"Data loader created: {'Synthetic' if USE_SYNTHETIC else 'Real'} data")
        print(f"Number of batches per epoch: {len(self.data_loader)}")
        
        # Training history
        self.history = {
            'epoch': [],
            'total_loss': [],
            'vae_loss': [],
            'compression_loss': [],
            'quality_loss': [],
            'rate_loss': [],
            'predicted_quality': [],
            'compression_ratio': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {}
        
        progress_bar = tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (image, lidar, imu, audio) in enumerate(progress_bar):
            # Move to device
            image, lidar, imu, audio = image.to(self.device), lidar.to(self.device), imu.to(self.device), audio.to(self.device)
            
            # Ensure correct shapes for LiDAR
            if len(lidar.shape) == 4:  # [B, T, N, 3]
                lidar = lidar.view(lidar.size(0), lidar.size(1), -1)  # [B, T, N*3]
            
            batch_loss = 0
            batch_metrics = {}
            
            # Process sequence timesteps
            hidden_state = None
            
            for t in range(SEQ_LEN):
                img_t = image[:, t]
                lidar_t = lidar[:, t]
                imu_t = imu[:, t]
                audio_t = audio[:, t]
                
                # Forward pass with enhanced model
                try:
                    (compressed, temporal_out, predicted, recon_img, mu, logvar, 
                     adjusted_compression, predicted_quality) = self.model(
                        img_t, lidar_t, imu_t, audio_t, hidden_state, 
                        compression_level=0.8, target_quality=TARGET_QUALITY
                    )
                    
                    # Compute enhanced loss
                    loss, metrics = compute_enhanced_loss(
                        recon_img, img_t, mu, logvar, compressed, temporal_out,
                        predicted_quality, TARGET_QUALITY, BETA_VAE
                    )
                    
                    batch_loss += loss
                    
                    # Accumulate metrics
                    for key, value in metrics.items():
                        if key not in batch_metrics:
                            batch_metrics[key] = []
                        batch_metrics[key].append(value)
                    
                    # Update hidden state for next timestep
                    hidden_state = temporal_out
                    
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    continue
            
            # Backward pass
            if batch_loss > 0:
                self.optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update progress bar
                avg_loss = batch_loss.item() / SEQ_LEN
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Quality': f'{np.mean(batch_metrics.get("quality_loss", [0])):.4f}',
                    'Compression': f'{np.mean(batch_metrics.get("compression_loss", [0])):.4f}'
                })
                
                epoch_losses.append(avg_loss)
                
                # Store metrics
                for key, values in batch_metrics.items():
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].extend(values)
        
        # Update learning rate
        self.scheduler.step()
        
        # Return epoch summary
        return np.mean(epoch_losses), epoch_metrics
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_losses = []
        val_metrics = {}
        
        with torch.no_grad():
            for image, lidar, imu, audio in self.data_loader:
                # Move to device
                image, lidar, imu, audio = image.to(self.device), lidar.to(self.device), imu.to(self.device), audio.to(self.device)
                
                # Ensure correct shapes
                if len(lidar.shape) == 4:
                    lidar = lidar.view(lidar.size(0), lidar.size(1), -1)
                
                batch_loss = 0
                batch_metrics = {}
                
                hidden_state = None
                
                for t in range(SEQ_LEN):
                    img_t = image[:, t]
                    lidar_t = lidar[:, t]
                    imu_t = imu[:, t]
                    audio_t = audio[:, t]
                    
                    try:
                        (compressed, temporal_out, predicted, recon_img, mu, logvar, 
                         adjusted_compression, predicted_quality) = self.model(
                            img_t, lidar_t, imu_t, audio_t, hidden_state,
                            compression_level=0.8, target_quality=TARGET_QUALITY
                        )
                        
                        loss, metrics = compute_enhanced_loss(
                            recon_img, img_t, mu, logvar, compressed, temporal_out,
                            predicted_quality, TARGET_QUALITY, BETA_VAE
                        )
                        
                        batch_loss += loss
                        
                        for key, value in metrics.items():
                            if key not in batch_metrics:
                                batch_metrics[key] = []
                            batch_metrics[key].append(value)
                        
                        hidden_state = temporal_out
                        
                    except Exception as e:
                        continue
                
                if batch_loss > 0:
                    val_losses.append(batch_loss.item() / SEQ_LEN)
                    
                    for key, values in batch_metrics.items():
                        if key not in val_metrics:
                            val_metrics[key] = []
                        val_metrics[key].extend(values)
        
        return np.mean(val_losses), val_metrics
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pth")
        print(f"Checkpoint saved: checkpoint_epoch_{epoch}.pth")
    
    def train(self):
        """Main training loop"""
        print("Starting enhanced training...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_loss = float('inf')
        
        for epoch in range(EPOCHS):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"{'='*50}")
            
            # Training
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.validate()
            
            # Update history
            self.history['epoch'].append(epoch)
            self.history['total_loss'].append(train_loss)
            
            for key in ['vae_loss', 'compression_loss', 'quality_loss', 'rate_loss']:
                if key in train_metrics:
                    self.history[key].append(np.mean(train_metrics[key]))
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Training Loss: {train_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), SAVE_PATH)
                print(f"  New best model saved: {SAVE_PATH}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, train_metrics)
        
        # Save final model
        torch.save(self.model.state_dict(), f"final_{SAVE_PATH}")
        print(f"\nTraining complete! Final model saved: final_{SAVE_PATH}")
        
        # Save training history
        with open('training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        print("Training history saved: training_history.json")

def main():
    """Main function"""
    trainer = EnhancedTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
