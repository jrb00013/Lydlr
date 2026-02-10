# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Advanced Compression Models
- Neural quantization
- Learned entropy coding
- Attention-based compression
- Multi-scale compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NeuralQuantizer(nn.Module):
    """Learned quantization for compression"""
    
    def __init__(self, num_levels=256):
        super().__init__()
        self.num_levels = num_levels
        self.quantization_centers = nn.Parameter(
            torch.linspace(-1, 1, num_levels)
        )
    
    def forward(self, x):
        # Find nearest quantization center
        x_flat = x.view(-1, 1)
        centers = self.quantization_centers.unsqueeze(0)
        
        distances = torch.abs(x_flat - centers)
        indices = torch.argmin(distances, dim=1)
        
        # Quantize
        quantized = self.quantization_centers[indices].view(x.shape)
        
        # Straight-through estimator
        return x + (quantized - x).detach()


class LearnedEntropyCoder(nn.Module):
    """Learned entropy coding for better compression"""
    
    def __init__(self, feature_dim=256, num_symbols=256):
        super().__init__()
        self.num_symbols = num_symbols
        
        # Probability model
        self.probability_model = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_symbols),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features):
        # Predict symbol probabilities
        probs = self.probability_model(features)
        
        # Calculate entropy (bits)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
        
        return entropy, probs


class AttentionCompressor(nn.Module):
    """Attention-based compression - focus on important features"""
    
    def __init__(self, d_model=256, n_heads=8, compression_ratio=0.5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.compression_ratio = compression_ratio
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        
        # Compression projection
        self.compress_dim = int(d_model * compression_ratio)
        self.compress_proj = nn.Linear(d_model, self.compress_dim)
        self.decompress_proj = nn.Linear(self.compress_dim, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Self-attention
        attn_out, attn_weights = self.attention(x, x, x)
        
        # Compress using attention-weighted features
        compressed = self.compress_proj(attn_out)
        
        # Decompress
        decompressed = self.decompress_proj(compressed)
        
        return compressed, decompressed, attn_weights


class MultiScaleCompressor(nn.Module):
    """Multi-scale compression for different quality levels"""
    
    def __init__(self, input_dim=256, scales=[0.25, 0.5, 1.0]):
        super().__init__()
        self.scales = scales
        
        # Encoders for each scale
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, int(input_dim * scale)),
                nn.ReLU(),
                nn.Linear(int(input_dim * scale), int(input_dim * scale * 0.5))
            ) for scale in scales
        ])
        
        # Decoders
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(int(input_dim * scale * 0.5), int(input_dim * scale)),
                nn.ReLU(),
                nn.Linear(int(input_dim * scale), input_dim)
            ) for scale in scales
        ])
    
    def forward(self, x, target_scale_idx=2):
        """Compress at specified scale"""
        # Encode
        encoded = self.encoders[target_scale_idx](x)
        
        # Decode
        decoded = self.decoders[target_scale_idx](encoded)
        
        return encoded, decoded


class RevolutionaryCompressor(nn.Module):
    """Revolutionary compression model combining all techniques"""
    
    def __init__(self, input_dim=256, latent_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Components
        self.attention_compressor = AttentionCompressor(
            d_model=input_dim, compression_ratio=0.5
        )
        self.quantizer = NeuralQuantizer(num_levels=256)
        self.entropy_coder = LearnedEntropyCoder(feature_dim=latent_dim)
        self.multiscale = MultiScaleCompressor(input_dim=input_dim)
        
        # Final compression
        self.final_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.final_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x, target_quality=0.8):
        """
        Forward pass with adaptive quality
        
        Args:
            x: Input features (batch, seq_len, dim)
            target_quality: Target quality level (0-1)
        """
        batch_size, seq_len, dim = x.shape
        
        # Attention-based compression
        attn_compressed, attn_decompressed, attn_weights = \
            self.attention_compressor(x)
        
        # Multi-scale compression (select scale based on quality)
        if target_quality > 0.8:
            scale_idx = 2  # Full quality
        elif target_quality > 0.5:
            scale_idx = 1  # Medium quality
        else:
            scale_idx = 0  # Low quality
        
        ms_encoded, ms_decoded = self.multiscale(attn_decompressed, scale_idx)
        
        # Final compression
        compressed = self.final_encoder(ms_decoded)
        
        # Quantization
        compressed = self.quantizer(compressed)
        
        # Entropy estimation
        entropy, probs = self.entropy_coder(compressed.mean(dim=1))
        
        # Decompression
        decompressed = self.final_decoder(compressed)
        
        return {
            'compressed': compressed,
            'decompressed': decompressed,
            'entropy': entropy,
            'attention_weights': attn_weights,
            'compression_ratio': self.latent_dim / self.input_dim
        }


class SensorMotorRevolutionaryCompressor(nn.Module):
    """Revolutionary compressor specifically for sensor-motor data"""
    
    def __init__(self, sensor_dim=256, motor_dim=6, latent_dim=32):
        super().__init__()
        
        # Sensor processing
        self.sensor_compressor = RevolutionaryCompressor(
            input_dim=sensor_dim, latent_dim=latent_dim
        )
        
        # Motor processing (simpler - lower dimensional)
        self.motor_encoder = nn.Sequential(
            nn.Linear(motor_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim // 2)
        )
        
        self.motor_decoder = nn.Sequential(
            nn.Linear(latent_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, motor_dim)
        )
        
        # Fusion and temporal modeling
        self.temporal_model = nn.LSTM(
            latent_dim + latent_dim // 2,
            latent_dim,
            batch_first=True,
            num_layers=2
        )
        
        # Adaptive compression controller
        self.compression_controller = nn.Sequential(
            nn.Linear(latent_dim + 2, 64),  # +2 for bandwidth and quality
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sensor_data, motor_data=None, bandwidth=1.0, 
                target_quality=0.8, hidden_state=None):
        """
        Compress sensor and motor data
        
        Args:
            sensor_data: Sensor features (batch, seq_len, sensor_dim)
            motor_data: Motor commands (batch, seq_len, motor_dim)
            bandwidth: Available bandwidth (0-1)
            target_quality: Target quality (0-1)
            hidden_state: LSTM hidden state
        """
        batch_size, seq_len = sensor_data.shape[:2]
        
        # Compress sensor data
        sensor_compressed = []
        sensor_decompressed = []
        
        for t in range(seq_len):
            sensor_t = sensor_data[:, t:t+1, :]
            result = self.sensor_compressor(sensor_t, target_quality)
            sensor_compressed.append(result['compressed'])
            sensor_decompressed.append(result['decompressed'])
        
        sensor_compressed = torch.cat(sensor_compressed, dim=1)
        
        # Compress motor data
        if motor_data is not None:
            motor_compressed = self.motor_encoder(motor_data)
        else:
            motor_compressed = torch.zeros(
                batch_size, seq_len, self.motor_encoder[-1].out_features,
                device=sensor_data.device
            )
        
        # Combine
        combined = torch.cat([sensor_compressed, motor_compressed], dim=-1)
        
        # Temporal compression
        temporal_out, hidden_state = self.temporal_model(combined, hidden_state)
        
        # Adaptive compression
        bandwidth_tensor = torch.full(
            (batch_size, 1), bandwidth, device=sensor_data.device
        )
        quality_tensor = torch.full(
            (batch_size, 1), target_quality, device=sensor_data.device
        )
        
        controller_input = torch.cat([
            temporal_out[:, -1:, :].squeeze(1),
            bandwidth_tensor,
            quality_tensor
        ], dim=-1)
        
        compression_level = self.compression_controller(controller_input)
        final_compressed = temporal_out * compression_level
        
        # Decode motor
        if motor_data is not None:
            motor_decompressed = self.motor_decoder(motor_compressed)
        else:
            motor_decompressed = None
        
        return {
            'compressed': final_compressed,
            'sensor_decompressed': torch.cat(sensor_decompressed, dim=1),
            'motor_decompressed': motor_decompressed,
            'compression_level': compression_level,
            'hidden_state': hidden_state
        }

