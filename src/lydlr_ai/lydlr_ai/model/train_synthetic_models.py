# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Revolutionary Training Pipeline for Synthetic Data
- Trains multiple compression models
- Generates synthetic sensor/motor data
- Real-time model versioning
- Automated deployment preparation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import lpips

from lydlr_ai.model.compressor import EnhancedMultimodalCompressor, compute_enhanced_loss

# Import SensorMotorCompressor - define it here if not available from edge_compressor_node
try:
    from lydlr_ai.model.edge_compressor_node import SensorMotorCompressor
except ImportError:
    # Define SensorMotorCompressor here for training (without ROS2 dependencies)
    import torch.nn as nn
    
    class SensorMotorCompressor(nn.Module):
        """Advanced compressor for sensor and motor data"""
        
        def __init__(self, sensor_dim=256, motor_dim=6, latent_dim=64):
            super().__init__()
            self.sensor_dim = sensor_dim
            self.motor_dim = motor_dim
            self.latent_dim = latent_dim
            
            # Sensor encoder
            self.sensor_encoder = nn.Sequential(
                nn.Linear(sensor_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, latent_dim)
            )
            
            # Motor encoder (for motor commands)
            self.motor_encoder = nn.Sequential(
                nn.Linear(motor_dim, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim // 2)
            )
            
            # Temporal compression
            self.temporal_compressor = nn.LSTM(
                latent_dim + latent_dim // 2, 
                latent_dim, 
                batch_first=True,
                num_layers=2
            )
            
            # Adaptive compression controller
            self.compression_controller = nn.Sequential(
                nn.Linear(latent_dim + 1, 64),  # +1 for bandwidth signal
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
            # Decoders
            self.sensor_decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, sensor_dim)
            )
            
            self.motor_decoder = nn.Sequential(
                nn.Linear(latent_dim // 2, 32),
                nn.ReLU(),
                nn.Linear(32, motor_dim)
            )
        
        def forward(self, sensor_data, motor_data=None, bandwidth_signal=1.0, hidden_state=None):
            # Encode
            sensor_encoded = self.sensor_encoder(sensor_data)
            
            if motor_data is not None:
                motor_encoded = self.motor_encoder(motor_data)
                combined = torch.cat([sensor_encoded, motor_encoded], dim=-1)
            else:
                combined = sensor_encoded
            
            # Temporal compression
            combined_seq = combined.unsqueeze(1)  # Add time dimension
            temporal_out, hidden_state = self.temporal_compressor(combined_seq, hidden_state)
            temporal_out = temporal_out.squeeze(1)
            
            # Adaptive compression based on bandwidth
            bandwidth_tensor = torch.full((temporal_out.size(0), 1), bandwidth_signal, 
                                         device=temporal_out.device)
            compression_level = self.compression_controller(
                torch.cat([temporal_out, bandwidth_tensor], dim=-1)
            )
            
            # Apply compression
            compressed = temporal_out * compression_level
            
            # Decode
            sensor_decoded = self.sensor_decoder(compressed[:, :self.latent_dim])
            
            if motor_data is not None:
                motor_decoded = self.motor_decoder(compressed[:, self.latent_dim:])
            else:
                motor_decoded = None
            
            return compressed, sensor_decoded, motor_decoded, hidden_state, compression_level


class SyntheticMultimodalDataset(Dataset):
    """Generates synthetic multimodal sensor and motor data"""
    
    def __init__(self, num_samples=1000, sequence_length=4, image_shape=(3, 480, 640)):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.image_shape = image_shape
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic image (simulate camera)
        image = torch.rand(self.sequence_length, *self.image_shape)
        
        # Generate synthetic LiDAR (point cloud)
        lidar = torch.rand(self.sequence_length, 1024, 3) * 20 - 10  # Points in [-10, 10]
        
        # Generate synthetic IMU (6-axis: accel + gyro)
        imu = torch.randn(self.sequence_length, 6) * 2.0
        
        # Generate synthetic audio (spectrogram-like)
        audio = torch.rand(self.sequence_length, 128, 128)
        
        # Generate synthetic motor commands (6-DOF: linear + angular)
        motor = torch.randn(self.sequence_length, 6) * 0.5
        
        # Add temporal correlation
        for t in range(1, self.sequence_length):
            image[t] = image[t-1] * 0.8 + torch.rand(*self.image_shape) * 0.2
            lidar[t] = lidar[t-1] * 0.9 + torch.rand(1024, 3) * 0.1
            imu[t] = imu[t-1] * 0.7 + torch.randn(6) * 0.3
            motor[t] = motor[t-1] * 0.8 + torch.randn(6) * 0.2
        
        return {
            'image': image,
            'lidar': lidar,
            'imu': imu,
            'audio': audio,
            'motor': motor
        }


class AdvancedCompressionTrainer:
    """Trains advanced compression models on synthetic data"""
    
    def __init__(self, model_dir="models", version=None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.version = version
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Training on device: {self.device}")
        
        # Initialize models
        self.multimodal_model = EnhancedMultimodalCompressor().to(self.device)
        self.sensor_motor_model = SensorMotorCompressor().to(self.device)
        
        # Loss functions
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Optimizers
        self.optimizer_mm = optim.AdamW(
            self.multimodal_model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        self.optimizer_sm = optim.AdamW(
            self.sensor_motor_model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # Learning rate schedulers
        self.scheduler_mm = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_mm, T_0=5, T_mult=2
        )
        self.scheduler_sm = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_sm, T_0=5, T_mult=2
        )
        
        # Training history
        self.history = {
            'epoch': [],
            'multimodal_loss': [],
            'sensor_motor_loss': [],
            'total_loss': [],
            'compression_ratio': [],
            'quality_score': []
        }
    
    def train_multimodal(self, dataloader, epochs=20):
        """Train multimodal compression model"""
        print(f"\n{'='*60}")
        print(f"🎯 Training Multimodal Compressor (v{self.version})")
        print(f"{'='*60}")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.multimodal_model.train()
            epoch_losses = []
            
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress:
                image = batch['image'].to(self.device)
                lidar = batch['lidar'].to(self.device)
                imu = batch['imu'].to(self.device)
                audio = batch['audio'].to(self.device)
                
                batch_size, seq_len = image.shape[:2]
                
                # Flatten sequence for processing
                image = image.view(batch_size * seq_len, *image.shape[2:])
                lidar = lidar.view(batch_size * seq_len, -1)
                imu = imu.view(batch_size * seq_len, -1)
                audio = audio.view(batch_size * seq_len, -1)
                
                self.optimizer_mm.zero_grad()
                
                hidden_state = None
                total_loss = 0
                
                for t in range(seq_len):
                    idx = t
                    img_t = image[idx::seq_len]
                    lidar_t = lidar[idx::seq_len]
                    imu_t = imu[idx::seq_len]
                    audio_t = audio[idx::seq_len]
                    
                    # Forward pass
                    (compressed, temporal_out, predicted, recon_img, mu, logvar,
                     adjusted_compression, predicted_quality) = self.multimodal_model(
                        img_t, lidar_t, imu_t, audio_t, hidden_state,
                        compression_level=0.8, target_quality=0.8
                    )
                    
                    # Compute loss
                    loss, metrics = compute_enhanced_loss(
                        recon_img, img_t, mu, logvar, compressed, temporal_out,
                        predicted_quality, 0.8, 0.1
                    )
                    
                    # Perceptual loss
                    perceptual = self.lpips_loss(
                        (img_t * 2 - 1), (recon_img * 2 - 1)
                    ).mean()
                    
                    total_loss += loss + 0.1 * perceptual
                    hidden_state = temporal_out
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.multimodal_model.parameters(), 1.0)
                self.optimizer_mm.step()
                
                epoch_losses.append(total_loss.item() / seq_len)
                progress.set_postfix({'Loss': f'{epoch_losses[-1]:.4f}'})
            
            self.scheduler_mm.step()
            avg_loss = np.mean(epoch_losses)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {self.optimizer_mm.param_groups[0]['lr']:.2e}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_multimodal_model()
        
        print(f"✅ Multimodal training complete! Best loss: {best_loss:.4f}")
    
    def train_sensor_motor(self, dataloader, epochs=20):
        """Train sensor-motor compression model"""
        print(f"\n{'='*60}")
        print(f"🎯 Training Sensor-Motor Compressor (v{self.version})")
        print(f"{'='*60}")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.sensor_motor_model.train()
            epoch_losses = []
            
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress:
                image = batch['image'].to(self.device)
                motor = batch['motor'].to(self.device)
                
                batch_size, seq_len = image.shape[:2]
                
                # Flatten
                image = image.view(batch_size * seq_len, -1)
                motor = motor.view(batch_size * seq_len, -1)
                
                # Extract features (simplified - use mean pooling)
                sensor_feat = image.mean(dim=1, keepdim=True).expand(-1, 256)
                
                self.optimizer_sm.zero_grad()
                
                hidden_state = None
                total_loss = 0
                
                for t in range(seq_len):
                    idx = t
                    sensor_t = sensor_feat[idx::seq_len]
                    motor_t = motor[idx::seq_len]
                    
                    # Forward pass
                    compressed, sensor_decoded, motor_decoded, hidden_state, comp_level = \
                        self.sensor_motor_model(
                            sensor_t, motor_t, bandwidth_signal=1.0, hidden_state=hidden_state
                        )
                    
                    # Reconstruction loss
                    sensor_loss = self.mse_loss(sensor_decoded, sensor_t)
                    motor_loss = self.mse_loss(motor_decoded, motor_t)
                    
                    # Compression loss (encourage compression)
                    compression_loss = torch.mean(compressed.abs()) * 0.01
                    
                    loss = sensor_loss + motor_loss + compression_loss
                    total_loss += loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sensor_motor_model.parameters(), 1.0)
                self.optimizer_sm.step()
                
                epoch_losses.append(total_loss.item() / seq_len)
                progress.set_postfix({'Loss': f'{epoch_losses[-1]:.4f}'})
            
            self.scheduler_sm.step()
            avg_loss = np.mean(epoch_losses)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {self.optimizer_sm.param_groups[0]['lr']:.2e}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_sensor_motor_model()
        
        print(f"✅ Sensor-Motor training complete! Best loss: {best_loss:.4f}")
    
    def save_multimodal_model(self):
        """Save multimodal model with metadata"""
        model_path = self.model_dir / f"compressor_v{self.version}.pth"
        metadata_path = self.model_dir / f"metadata_v{self.version}.json"
        
        checkpoint = {
            'model_state_dict': self.multimodal_model.state_dict(),
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'multimodal_compressor'
        }
        
        torch.save(checkpoint, model_path)
        
        metadata = {
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'multimodal_compressor',
            'architecture': 'EnhancedMultimodalCompressor',
            'parameters': sum(p.numel() for p in self.multimodal_model.parameters()),
            'device': str(self.device)
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved model: {model_path}")
    
    def save_sensor_motor_model(self):
        """Save sensor-motor model"""
        model_path = self.model_dir / f"sensor_motor_v{self.version}.pth"
        
        checkpoint = {
            'model_state_dict': self.sensor_motor_model.state_dict(),
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'sensor_motor_compressor'
        }
        
        torch.save(checkpoint, model_path)
        print(f"💾 Saved sensor-motor model: {model_path}")
    
    def train_all(self, num_samples=1000, epochs=20, batch_size=4):
        """Train all models"""
        print(f"\n{'='*60}")
        print(f"🚀 Starting Complete Training Pipeline")
        print(f"{'='*60}")
        
        # Create dataset
        dataset = SyntheticMultimodalDataset(num_samples=num_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Train multimodal
        self.train_multimodal(dataloader, epochs=epochs)
        
        # Train sensor-motor
        self.train_sensor_motor(dataloader, epochs=epochs)
        
        # Save training history
        history_path = self.model_dir / f"training_history_v{self.version}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n✅ All training complete! Models saved to {self.model_dir}")
        print(f"   Version: {self.version}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train compression models on synthetic data')
    parser.add_argument('--version', type=str, default=None, help='Model version')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--samples', type=int, default=1000, help='Number of synthetic samples')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--model-dir', type=str, default='models', help='Model directory')
    
    args = parser.parse_args()
    
    trainer = AdvancedCompressionTrainer(model_dir=args.model_dir, version=args.version)
    trainer.train_all(
        num_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()

