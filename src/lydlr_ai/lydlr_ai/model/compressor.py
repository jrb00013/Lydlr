# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Enhanced Multimodal Compressor with all improvements
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import lpips 
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import math

from lydlr_ai.utils.voxel_utils import lidar_to_pointcloud

# ============================================================================
# IMPROVEMENT 1: Enhanced VAE with β-VAE and Progressive Decoding
# ============================================================================

class EnhancedVAE(nn.Module):
    """Enhanced VAE with ResNet18 backbone and progressive decoding"""
    
    def __init__(self, input_channels=3, latent_dim=256, input_height=480, input_width=640, beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder: ResNet18 backbone (fine-tunable)
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove final layers
        
        # Calculate feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, input_height, input_width)
            features = self.encoder(dummy_input)
            self.feature_dim = features.shape[1] * features.shape[2] * features.shape[3]
            self.encoder_channels = features.shape[1]  # This should be 512 for ResNet18
            self.encoder_height = features.shape[2]    # This should be 15 for 480x640 input
            self.encoder_width = features.shape[3]     # This should be 20 for 480x640 input
        
        # VAE bottleneck
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)
        
        # Progressive decoder with multiple scales - align with encoder output
        self.decoder_fc = nn.Linear(latent_dim, self.feature_dim)
        self.decoder_conv = nn.ModuleList([
            # Scale 1: 1/8 resolution - start with actual encoder output dimensions
            nn.Sequential(
                nn.ConvTranspose2d(self.encoder_channels, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.BatchNorm2d(128), nn.ReLU()
            ),
            # Scale 2: 1/4 resolution  
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.BatchNorm2d(32), nn.ReLU()
            ),
            # Scale 3: Full resolution - ensure output matches input size
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.BatchNorm2d(16), nn.ReLU(),
                nn.ConvTranspose2d(16, input_channels, 4, stride=2, padding=1),
                nn.Sigmoid()
            )
        ])
        
        # Multi-scale feature fusion - match the actual feature dimensions
        self.scale_fusion = nn.ModuleList([
            nn.Conv2d(256, 128, 1),  # Scale 1: 256 -> 128 (after first decoder)
            nn.Conv2d(128, 64, 1),   # Scale 2: 128 -> 64 (after second decoder)
            nn.Conv2d(32, 16, 1)     # Scale 3: 32 -> 16 (after third decoder)
        ])
        
        # Add final resize to ensure correct output dimensions
        self.final_resize = nn.AdaptiveAvgPool2d((input_height, input_width))
    
    def encode(self, x):
        """Encode input to latent space"""
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode_progressive(self, z, target_scale=2):
        """Progressive decoding with quality control"""
        x = self.decoder_fc(z)
        x = x.view(x.size(0), self.encoder_channels, self.encoder_height, self.encoder_width)  # Reshape to feature map
        
        outputs = []
        current = x
        
        for i, (decoder, fusion) in enumerate(zip(self.decoder_conv, self.scale_fusion)):
            current = decoder(current)
            # Only apply fusion if dimensions match
            if i < len(self.scale_fusion) and current.size(1) == self.scale_fusion[i].in_channels:
                current = fusion(current)
            outputs.append(current)
            
            if i == target_scale:  # Stop at target scale
                break
        
        final_output = outputs[-1] if outputs else current
        # Ensure final output matches expected dimensions
        if hasattr(self, 'final_resize'):
            final_output = self.final_resize(final_output)
        return final_output
    
    def forward(self, x, target_scale=2):
        """Forward pass with progressive decoding"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode_progressive(z, target_scale)
        return recon, mu, logvar
    
    def loss(self, recon_x, x, mu, logvar):
        """β-VAE loss with reconstruction and KL divergence"""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with β weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

# ============================================================================
# IMPROVEMENT 2: Attention-Based Multimodal Fusion
# ============================================================================

class CrossModalAttention(nn.Module):
    """Cross-modal attention for better fusion"""
    
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Output projection and residual connection
        output = self.out_proj(context)
        output = self.layer_norm(output + query)
        
        return output

class MultimodalFusion(nn.Module):
    """Enhanced multimodal fusion with attention"""
    
    def __init__(self, image_dim=128, lidar_dim=128, imu_dim=32, audio_dim=128, fusion_dim=256):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # Project all modalities to common dimension
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.lidar_proj = nn.Linear(lidar_dim, fusion_dim)
        self.imu_proj = nn.Linear(imu_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(fusion_dim, n_heads=8)
        
        # Final fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 4, fusion_dim * 2),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(), nn.Dropout(0.1)
        )
    
    def forward(self, image_feat, lidar_feat, imu_feat, audio_feat):
        # Project to common space
        img_proj = self.image_proj(image_feat)
        lidar_proj = self.lidar_proj(lidar_feat)
        imu_proj = self.imu_proj(imu_feat)
        audio_proj = self.audio_proj(audio_feat)
        
        # Stack features for attention
        features = torch.stack([img_proj, lidar_proj, imu_proj, audio_proj], dim=1)
        
        # Apply cross-modal attention
        attended_features = self.cross_attention(features, features, features)
        
        # Flatten and fuse
        fused = attended_features.view(attended_features.size(0), -1)
        fused = self.fusion_mlp(fused)
        
        return fused

# ============================================================================
# IMPROVEMENT 3: Neural Delta Compression
# ============================================================================

class DeltaCompressor(nn.Module):
    """Neural delta compression - encode only changes over time"""
    
    def __init__(self, feature_dim=256, delta_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.delta_dim = delta_dim
        
        # Delta encoder
        self.delta_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, delta_dim),  # Current + previous
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(delta_dim, delta_dim),
            nn.ReLU()
        )
        
        # Delta decoder
        self.delta_decoder = nn.Sequential(
            nn.Linear(delta_dim, feature_dim * 2),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU()
        )
        
        # Temporal predictor
        self.temporal_predictor = nn.LSTM(delta_dim, feature_dim, batch_first=True)
    
    def forward(self, current_feat, previous_feat=None):
        if previous_feat is None:
            # First frame - no delta
            return current_feat, torch.zeros_like(current_feat)
        
        # Compute delta
        combined = torch.cat([current_feat, previous_feat], dim=-1)
        delta = self.delta_encoder(combined)
        
        # Decode delta back to features
        reconstructed = self.delta_decoder(delta)
        
        # Temporal prediction
        delta_seq = delta.unsqueeze(1)  # Add time dimension
        predicted, _ = self.temporal_predictor(delta_seq)
        predicted = predicted.squeeze(1)
        
        return reconstructed, predicted

# ============================================================================
# IMPROVEMENT 4: Enhanced Temporal Modeling
# ============================================================================

class TemporalTransformer(nn.Module):
    """Enhanced temporal transformer with causal attention"""
    
    def __init__(self, d_model=256, n_heads=8, n_layers=6, max_seq_len=100):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch, features)
        output = self.transformer(x, mask=mask)
        output = output.transpose(0, 1)  # Back to (batch, seq_len, features)
        
        return self.output_proj(output)

# ============================================================================
# IMPROVEMENT 5: Progressive Quality Control
# ============================================================================

class QualityController(nn.Module):
    """Dynamic quality control based on predicted compression quality"""
    
    def __init__(self, feature_dim=256, quality_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.quality_dim = quality_dim
        
        # Quality predictor
        self.quality_predictor = nn.Sequential(
            nn.Linear(feature_dim, quality_dim),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(quality_dim, quality_dim // 2),
            nn.ReLU(),
            nn.Linear(quality_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Compression level controller
        self.compression_controller = nn.Sequential(
            nn.Linear(feature_dim + 1, quality_dim),  # features + quality_score
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(quality_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, target_quality=0.8):
        # Predict current quality
        predicted_quality = self.quality_predictor(features)
        
        # Determine compression level
        quality_input = torch.cat([features, predicted_quality], dim=-1)
        compression_level = self.compression_controller(quality_input)
        
        # Adjust based on target quality
        adjusted_level = torch.clamp(compression_level + (target_quality - predicted_quality), 0.1, 1.0)
        
        return adjusted_level, predicted_quality

# ============================================================================
# MAIN ENHANCED COMPRESSOR
# ============================================================================

class EnhancedMultimodalCompressor(nn.Module):
    """Enhanced multimodal compressor with all improvements"""
    
    def __init__(self, image_shape=(3, 480, 640), lidar_dim=1024, imu_dim=6, audio_dim=128*128):
        super().__init__()
        channels, height, width = image_shape
        
        # Enhanced VAE
        self.vae = EnhancedVAE(input_channels=channels, latent_dim=256, 
                              input_height=height, input_width=width, beta=0.1)
        
        # Modality encoders
        self.image_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_dim * 3, 256),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_dim, 64),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 256),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # Enhanced fusion
        self.fusion = MultimodalFusion(
            image_dim=1024, lidar_dim=128, imu_dim=32, audio_dim=128, fusion_dim=256
        )
        
        # Delta compression
        self.delta_compressor = DeltaCompressor(feature_dim=256, delta_dim=128)
        
        # Temporal modeling
        self.temporal_transformer = TemporalTransformer(d_model=256, n_heads=8, n_layers=4)
        
        # Quality control
        self.quality_controller = QualityController(feature_dim=256)
        
        # Final compression
        self.compression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
    
    def forward(self, image, lidar, imu, audio, hidden_state=None, compression_level=0.8, target_quality=0.8):
        batch_size = image.size(0)
        
        # Encode modalities
        img_feat = self.image_encoder(image).view(batch_size, -1)
        lidar_feat = self.lidar_encoder(lidar.view(batch_size, -1))
        imu_feat = self.imu_encoder(imu)
        audio_feat = self.audio_encoder(audio.view(batch_size, -1))
        
        # Multimodal fusion
        fused = self.fusion(img_feat, lidar_feat, imu_feat, audio_feat)
        
        # Delta compression (if we have previous state)
        if hidden_state is not None:
            fused, predicted = self.delta_compressor(fused, hidden_state)
        else:
            predicted = torch.zeros_like(fused)
        
        # Temporal modeling
        fused_seq = fused.unsqueeze(1)  # Add time dimension
        temporal_out = self.temporal_transformer(fused_seq)
        temporal_out = temporal_out.squeeze(1)
        
        # Quality control
        adjusted_compression, predicted_quality = self.quality_controller(temporal_out, target_quality)
        
        # Apply compression
        compressed = self.compression_head(temporal_out)
        compressed = compressed * adjusted_compression
        
        # VAE reconstruction for quality assessment
        recon_img, mu, logvar = self.vae(image, target_scale=2)
        
        return compressed, temporal_out, predicted, recon_img, mu, logvar, adjusted_compression, predicted_quality

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def compute_enhanced_loss(recon_img, image, mu, logvar, compressed, temporal_out, 
                         predicted_quality, target_quality=0.8, beta=0.1):
    """Enhanced loss function with all components"""
    
    # VAE loss - compute directly since we can't call class method on instance
    recon_loss = F.mse_loss(recon_img, image, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    vae_loss = recon_loss + beta * kl_loss
    
    # Compression loss - compare compressed output with temporal features
    # Project temporal_out to match compressed dimensions if needed
    if temporal_out.size(1) != compressed.size(1):
        projection = nn.Linear(temporal_out.size(1), compressed.size(1)).to(temporal_out.device)
        temporal_proj = projection(temporal_out)
    else:
        temporal_proj = temporal_out
    
    compression_loss = F.mse_loss(compressed, temporal_proj)
    
    # Quality loss
    quality_loss = F.mse_loss(predicted_quality, torch.full_like(predicted_quality, target_quality))
    
    # Rate-distortion loss
    rate_loss = torch.sum(compressed.abs()) * 0.01  # L1 regularization
    
    # Total loss
    total_loss = vae_loss + compression_loss + quality_loss + rate_loss
    
    return total_loss, {
        'vae_loss': vae_loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'compression_loss': compression_loss.item(),
        'quality_loss': quality_loss.item(),
        'rate_loss': rate_loss.item()
    }
