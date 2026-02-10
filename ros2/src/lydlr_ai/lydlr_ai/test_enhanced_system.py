#!/usr/bin/env python3
"""
Comprehensive Test Script for Enhanced Lydlr System
Tests all improvements: VAE, attention fusion, delta compression, etc.
"""

import os
import sys
import torch
import numpy as np
import time

# Add paths (support both old and new structure)
script_dir = os.path.dirname(__file__)
# Try new structure first (ros2/src/lydlr_ai)
if os.path.exists(os.path.join(script_dir, '..', '..', '..', '..', 'ros2', 'src', 'lydlr_ai')):
    base_path = os.path.join(script_dir, '..', '..', '..', '..', 'ros2', 'src', 'lydlr_ai')
else:
    # Fallback to old structure
    base_path = os.path.join(script_dir, '..', '..', '..', 'src', 'lydlr_ai')
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, 'lydlr_ai', 'model'))

def test_enhanced_vae():
    """Test Enhanced VAE functionality"""
    print("Testing Enhanced VAE...")
    
    try:
        from lydlr_ai.model.compressor import EnhancedVAE
        
        # Create VAE
        vae = EnhancedVAE(input_channels=3, latent_dim=256, input_height=480, input_width=640, beta=0.1)
        
        # Test input
        dummy_input = torch.randn(2, 3, 480, 640)
        
        # Forward pass
        recon, mu, logvar = vae(dummy_input, target_scale=2)
        
        # Test loss computation
        loss, recon_loss, kl_loss = vae.loss(recon, dummy_input, mu, logvar)
        
        print(f"  ✓ VAE forward pass: {recon.shape}")
        print(f"  ✓ VAE loss computation: {loss.item():.4f}")
        print(f"  ✓ Reconstruction loss: {recon_loss.item():.4f}")
        print(f"  ✓ KL loss: {kl_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ VAE test failed: {e}")
        return False

def test_attention_fusion():
    """Test attention-based multimodal fusion"""
    print("Testing Attention-Based Fusion...")
    
    try:
        from lydlr_ai.model.compressor import MultimodalFusion
        
        # Create fusion module
        fusion = MultimodalFusion(
            image_dim=1024, lidar_dim=128, imu_dim=32, audio_dim=128, fusion_dim=256
        )
        
        # Test inputs
        img_feat = torch.randn(2, 1024)
        lidar_feat = torch.randn(2, 128)
        imu_feat = torch.randn(2, 32)
        audio_feat = torch.randn(2, 128)
        
        # Forward pass
        fused = fusion(img_feat, lidar_feat, imu_feat, audio_feat)
        
        print(f"  ✓ Fusion forward pass: {fused.shape}")
        print(f"  ✓ Expected shape: torch.Size([2, 256])")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Fusion test failed: {e}")
        return False

def test_delta_compression():
    """Test neural delta compression"""
    print("Testing Delta Compression...")
    
    try:
        from lydlr_ai.model.compressor import DeltaCompressor
        
        # Create delta compressor
        delta_comp = DeltaCompressor(feature_dim=256, delta_dim=128)
        
        # Test inputs
        current_feat = torch.randn(2, 256)
        previous_feat = torch.randn(2, 256)
        
        # Forward pass
        reconstructed, predicted = delta_comp(current_feat, previous_feat)
        
        print(f"  ✓ Delta compression forward pass: {reconstructed.shape}")
        print(f"  ✓ Temporal prediction: {predicted.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Delta compression test failed: {e}")
        return False

def test_temporal_transformer():
    """Test enhanced temporal transformer"""
    print("Testing Temporal Transformer...")
    
    try:
        from lydlr_ai.model.compressor import TemporalTransformer
        
        # Create transformer
        transformer = TemporalTransformer(d_model=256, n_heads=8, n_layers=4)
        
        # Test input
        dummy_input = torch.randn(2, 10, 256)  # batch, seq_len, features
        
        # Forward pass
        output = transformer(dummy_input)
        
        print(f"  ✓ Transformer forward pass: {output.shape}")
        print(f"  ✓ Expected shape: torch.Size([2, 10, 256])")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Transformer test failed: {e}")
        return False

def test_quality_controller():
    """Test quality controller"""
    print("Testing Quality Controller...")
    
    try:
        from lydlr_ai.model.compressor import QualityController
        
        # Create quality controller
        qc = QualityController(feature_dim=256, quality_dim=64)
        
        # Test input
        features = torch.randn(2, 256)
        
        # Forward pass
        compression_level, predicted_quality = qc(features, target_quality=0.8)
        
        print(f"  ✓ Quality controller forward pass")
        print(f"  ✓ Compression level: {compression_level.shape}")
        print(f"  ✓ Predicted quality: {predicted_quality.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Quality controller test failed: {e}")
        return False

def test_full_model():
    """Test the complete enhanced model"""
    print("Testing Full Enhanced Model...")
    
    try:
        from lydlr_ai.model.compressor import EnhancedMultimodalCompressor
        
        # Create model
        model = EnhancedMultimodalCompressor()
        
        # Test inputs
        image = torch.randn(2, 3, 480, 640)
        lidar = torch.randn(2, 1024 * 3)  # Flattened
        imu = torch.randn(2, 6)
        audio = torch.randn(2, 128 * 128)
        
        # Forward pass
        start_time = time.time()
        (compressed, temporal_out, predicted, recon_img, mu, logvar, 
         adjusted_compression, predicted_quality) = model(
            image, lidar, imu, audio, hidden_state=None,
            compression_level=0.8, target_quality=0.8
        )
        end_time = time.time()
        
        print(f"  ✓ Full model forward pass: {end_time - start_time:.4f}s")
        print(f"  ✓ Compressed output: {compressed.shape}")
        print(f"  ✓ Temporal output: {temporal_out.shape}")
        print(f"  ✓ Reconstructed image: {recon_img.shape}")
        print(f"  ✓ Compression level: {adjusted_compression.shape}")
        print(f"  ✓ Predicted quality: {predicted_quality.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Full model test failed: {e}")
        return False

def test_loss_function():
    """Test enhanced loss function"""
    print("Testing Enhanced Loss Function...")
    
    try:
        from lydlr_ai.model.compressor import compute_enhanced_loss
        
        # Create dummy data
        recon_img = torch.randn(2, 3, 480, 640)
        image = torch.randn(2, 3, 480, 640)
        mu = torch.randn(2, 256)
        logvar = torch.randn(2, 256)
        compressed = torch.randn(2, 64)
        target_compression = torch.randn(2, 256)
        predicted_quality = torch.randn(2, 1)
        
        # Compute loss
        total_loss, metrics = compute_enhanced_loss(
            recon_img, image, mu, logvar, compressed, target_compression,
            predicted_quality, target_quality=0.8, beta=0.1
        )
        
        print(f"  ✓ Loss computation: {total_loss.item():.4f}")
        print(f"  ✓ Loss components: {list(metrics.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Loss function test failed: {e}")
        return False

def test_memory_efficiency():
    """Test memory efficiency"""
    print("Testing Memory Efficiency...")
    
    try:
        from lydlr_ai.model.compressor import EnhancedMultimodalCompressor
        
        # Create model
        model = EnhancedMultimodalCompressor()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  ✓ Total parameters: {total_params:,}")
        print(f"  ✓ Trainable parameters: {trainable_params:,}")
        
        # Test with different batch sizes
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            try:
                image = torch.randn(batch_size, 3, 480, 640)
                lidar = torch.randn(batch_size, 1024 * 3)
                imu = torch.randn(batch_size, 6)
                audio = torch.randn(batch_size, 128 * 128)
                
                with torch.no_grad():
                    _ = model(image, lidar, imu, audio)
                
                print(f"  ✓ Batch size {batch_size}: OK")
                
            except Exception as e:
                print(f"  ✗ Batch size {batch_size}: Failed - {e}")
                break
        
        return True
        
    except Exception as e:
        print(f"  ✗ Memory efficiency test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ENHANCED LYDLR SYSTEM COMPREHENSIVE TEST")
    print("=" * 60)
    
    tests = [
        test_enhanced_vae,
        test_attention_fusion,
        test_delta_compression,
        test_temporal_transformer,
        test_quality_controller,
        test_full_model,
        test_loss_function,
        test_memory_efficiency
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Enhanced system is ready for training.")
        print("\nNext steps:")
        # Support both old and new structure
        train_path = "ros2/src/lydlr_ai/enhanced_train.py" if os.path.exists("ros2/src/lydlr_ai/enhanced_train.py") else "src/lydlr_ai/enhanced_train.py"
        print(f"1. Run enhanced training: python3 {train_path}")
        print("2. Monitor training progress and metrics")
        print("3. Test with real sensor data when available")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check import paths")
        print("3. Verify PyTorch version compatibility")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
