#!/usr/bin/env python3
"""
Data Loader for Lydlr Training
Supports both synthetic data and real sensor data
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import glob
from typing import Tuple, List, Optional

class LydlrDataset(Dataset):
    """Dataset class for Lydlr multimodal sensor data"""
    
    def __init__(self, data_dir: str, sequence_length: int = 4, synthetic: bool = False):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.synthetic = synthetic
        
        if synthetic:
            self.data_list = None  # Will generate synthetic data on-the-fly
        else:
            # Load real data sequences
            self.data_list = self._load_real_data_sequences()
    
    def _load_real_data_sequences(self) -> List[str]:
        """Load paths to real data sequences"""
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist. Using synthetic data.")
            self.synthetic = True
            return []
        
        # Find all sequence directories
        sequence_pattern = os.path.join(self.data_dir, "sequence_*")
        sequences = glob.glob(sequence_pattern)
        
        if not sequences:
            print(f"No sequences found in {self.data_dir}. Using synthetic data.")
            self.synthetic = True
            return []
        
        print(f"Found {len(sequences)} real data sequences")
        return sorted(sequences)
    
    def _generate_synthetic_sequence(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic sensor data for training"""
        # Generate random data matching expected shapes - no batch dimension here
        images = torch.rand(self.sequence_length, 3, 480, 640)
        lidar = torch.rand(self.sequence_length, 1024 * 3)  # Flattened to 2D
        imu = torch.rand(self.sequence_length, 6)
        audio = torch.rand(self.sequence_length, 128 * 128)
        
        return images, lidar, imu, audio
    
    def _load_real_sequence(self, sequence_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load a real data sequence"""
        try:
            # Load numpy arrays
            images = np.load(os.path.join(sequence_path, 'images.npy'))
            lidar = np.load(os.path.join(sequence_path, 'lidar.npy'))
            imu = np.load(os.path.join(sequence_path, 'imu.npy'))
            audio = np.load(os.path.join(sequence_path, 'audio.npy'))
            
            # Convert to tensors and ensure correct shapes
            images = torch.from_numpy(images).float()  # [T, 3, 480, 640]
            lidar = torch.from_numpy(lidar).float()    # [T, 1024, 3]
            imu = torch.from_numpy(imu).float()        # [T, 6]
            audio = audio.unsqueeze(0)    # [T, 16384]
            
            # Flatten LiDAR to 2D - no batch dimension needed
            lidar = lidar.view(lidar.size(0), -1)  # [T, 1024*3]
            # images, imu, audio already have correct shapes
            
            return images, lidar, imu, audio
            
        except Exception as e:
            print(f"Error loading sequence {sequence_path}: {e}")
            # Fallback to synthetic data
            return self._generate_synthetic_sequence()
    
    def __len__(self) -> int:
        if self.synthetic:
            return 1000  # Generate 1000 synthetic sequences
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.synthetic:
            return self._generate_synthetic_sequence()
        else:
            sequence_path = self.data_list[idx % len(self.data_list)]
            return self._load_real_sequence(sequence_path)

def create_data_loader(
    data_dir: str = "~/lydlr_ws/data/training_data",
    batch_size: int = 2,
    sequence_length: int = 4,
    synthetic: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for Lydlr training"""
    
    # Expand user path
    data_dir = os.path.expanduser(data_dir)
    
    # Create dataset
    dataset = LydlrDataset(data_dir, sequence_length, synthetic)
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return loader

def test_data_loader():
    """Test the data loader with both synthetic and real data"""
    
    print("Testing Data Loader...")
    
    # Test synthetic data
    print("\n1. Testing synthetic data loader:")
    synthetic_loader = create_data_loader(synthetic=True, batch_size=2, sequence_length=4)
    
    for i, (images, lidar, imu, audio) in enumerate(synthetic_loader):
        print(f"  Batch {i}:")
        print(f"    Images: {images.shape}")
        print(f"    LiDAR: {lidar.shape}")
        print(f"    IMU: {imu.shape}")
        print(f"    Audio: {audio.shape}")
        if i >= 2:  # Just test first few batches
            break
    
    # Test real data loader (if data exists)
    print("\n2. Testing real data loader:")
    real_loader = create_data_loader(synthetic=False, batch_size=1, sequence_length=4)
    
    if real_loader.dataset.synthetic:
        print("  No real data found, using synthetic fallback")
    else:
        print(f"  Found {len(real_loader.dataset)} real sequences")
        for i, (images, lidar, imu, audio) in enumerate(real_loader):
            print(f"  Real Batch {i}:")
            print(f"    Images: {images.shape}")
            print(f"    LiDAR: {lidar.shape}")
            print(f"    IMU: {imu.shape}")
            print(f"    Audio: {audio.shape}")
            if i >= 1:  # Just test first batch
                break

if __name__ == "__main__":
    test_data_loader()
