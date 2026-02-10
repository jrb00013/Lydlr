"""
Model Manager - Handles model loading, inference, and lifecycle
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json
import sys
import os

# Add parent directory to path to import model classes
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from lydlr_ai.model.compressor import EnhancedMultimodalCompressor
    MODEL_CLASSES_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Model classes not available. Inference will be limited.")
    MODEL_CLASSES_AVAILABLE = False


class ModelManager:
    """Manages model lifecycle and inference"""
    
    def __init__(self, model_dir: Path, device: Optional[str] = None):
        self.model_dir = model_dir
        self.device = self._get_device(device)
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        
    def _get_device(self, device: Optional[str] = None) -> torch.device:
        """Determine the best available device"""
        if device:
            return torch.device(device)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        info = {
            "device": str(self.device),
            "gpu_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(0) / (1024 ** 2)
            info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(0) / (1024 ** 2)
        
        return info
    
    def load_model(self, version: str, architecture: str = "EnhancedMultimodalCompressor") -> bool:
        """Load a model with proper architecture"""
        try:
            # Find model file - support both old and new naming patterns
            model_file = None
            for pattern in [
                f"lydlr_compressor_v{version}.pth",  # New naming convention
                f"compressor_v{version}.pth",        # Old naming convention (backward compatibility)
                f"lydlr_sensor_motor_v{version}.pth",
                f"sensor_motor_v{version}.pth",
                f"*_v{version}.pth"  # Fallback pattern
            ]:
                matches = list(self.model_dir.glob(pattern))
                if matches:
                    model_file = matches[0]
                    break
            
            if not model_file or not model_file.exists():
                return False
            
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location=self.device)
            
            # Determine architecture from checkpoint or metadata
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                metadata = checkpoint.get('metadata', {})
            else:
                state_dict = checkpoint
                metadata = {}
            
            # Load metadata if available - try both naming patterns
            metadata_file = None
            for pattern in [
                f"metadata_lydlr_compressor_v{version}.json",
                f"metadata_v{version}.json"
            ]:
                potential_file = self.model_dir / pattern
                if potential_file.exists():
                    metadata_file = potential_file
                    break
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    file_metadata = json.load(f)
                    metadata.update(file_metadata)
            
            # Initialize model architecture
            if MODEL_CLASSES_AVAILABLE and architecture == "EnhancedMultimodalCompressor":
                # Get model parameters from metadata or use defaults
                image_shape = metadata.get('image_shape', (3, 480, 640))
                lidar_dim = metadata.get('lidar_dim', 1024)
                imu_dim = metadata.get('imu_dim', 6)
                audio_dim = metadata.get('audio_dim', 128 * 128)
                
                model = EnhancedMultimodalCompressor(
                    image_shape=image_shape,
                    lidar_dim=lidar_dim,
                    imu_dim=imu_dim,
                    audio_dim=audio_dim
                ).to(self.device)
                
                # Load state dict
                try:
                    model.load_state_dict(state_dict)
                except RuntimeError as e:
                    # Try with strict=False if there are missing keys
                    model.load_state_dict(state_dict, strict=False)
                    print(f"⚠️  Warning: Some keys were missing when loading model: {e}")
                
                model.eval()
            else:
                # Fallback: store state dict only
                model = None
                print(f"⚠️  Model architecture not available, storing state dict only")
            
            # Store model
            self.loaded_models[version] = {
                "model": model,
                "state_dict": state_dict,
                "filename": model_file.name,
                "loaded_at": datetime.utcnow(),
                "metadata": metadata,
                "architecture": architecture,
                "device": str(self.device)
            }
            
            # Initialize stats
            self.model_stats[version] = {
                "total_inferences": 0,
                "total_errors": 0,
                "total_inference_time_ms": 0.0,
                "total_compression_ratio": 0.0,
                "total_quality_score": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "last_inference": None
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model {version}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload_model(self, version: str) -> bool:
        """Unload a model from memory"""
        if version not in self.loaded_models:
            return False
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available() and self.loaded_models[version].get("model"):
            del self.loaded_models[version]["model"]
            torch.cuda.empty_cache()
        
        del self.loaded_models[version]
        if version in self.model_stats:
            del self.model_stats[version]
        
        return True
    
    def prepare_inputs(self, image: Optional[np.ndarray], lidar: Optional[np.ndarray],
                      imu: Optional[np.ndarray], audio: Optional[np.ndarray],
                      batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
        """Prepare input tensors for model inference"""
        device = self.device
        
        # Prepare image
        if image is not None:
            img_tensor = torch.tensor(image, dtype=torch.float32, device=device)
            if len(img_tensor.shape) == 3:  # [H, W, C]
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            elif len(img_tensor.shape) == 4:  # [B, H, W, C]
                img_tensor = img_tensor.permute(0, 3, 1, 2)  # [B, C, H, W]
            # Normalize to [0, 1] if needed
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
        else:
            img_tensor = torch.zeros(batch_size, 3, 480, 640, device=device)
        
        # Prepare LiDAR
        if lidar is not None:
            lidar_tensor = torch.tensor(lidar, dtype=torch.float32, device=device)
            if len(lidar_tensor.shape) == 1:
                lidar_tensor = lidar_tensor.unsqueeze(0)
            # Pad or truncate to expected size (1024 * 3)
            expected_size = 1024 * 3
            if lidar_tensor.shape[-1] < expected_size:
                padding = torch.zeros(lidar_tensor.shape[0], expected_size - lidar_tensor.shape[-1], device=device)
                lidar_tensor = torch.cat([lidar_tensor, padding], dim=-1)
            elif lidar_tensor.shape[-1] > expected_size:
                lidar_tensor = lidar_tensor[..., :expected_size]
        else:
            lidar_tensor = torch.zeros(batch_size, 1024 * 3, device=device)
        
        # Prepare IMU
        if imu is not None:
            imu_tensor = torch.tensor(imu, dtype=torch.float32, device=device)
            if len(imu_tensor.shape) == 1:
                imu_tensor = imu_tensor.unsqueeze(0)
            if imu_tensor.shape[-1] != 6:
                raise ValueError(f"IMU data must have 6 values, got {imu_tensor.shape[-1]}")
        else:
            imu_tensor = torch.zeros(batch_size, 6, device=device)
        
        # Prepare audio
        if audio is not None:
            audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device)
            if len(audio_tensor.shape) == 2:  # [H, W]
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif len(audio_tensor.shape) == 3:  # [B, H, W]
                audio_tensor = audio_tensor.unsqueeze(1)  # [B, 1, H, W]
            # Flatten to expected size
            audio_tensor = audio_tensor.view(audio_tensor.shape[0], -1)
            expected_audio_size = 128 * 128
            if audio_tensor.shape[-1] < expected_audio_size:
                padding = torch.zeros(audio_tensor.shape[0], expected_audio_size - audio_tensor.shape[-1], device=device)
                audio_tensor = torch.cat([audio_tensor, padding], dim=-1)
            elif audio_tensor.shape[-1] > expected_audio_size:
                audio_tensor = audio_tensor[..., :expected_audio_size]
        else:
            audio_tensor = torch.zeros(batch_size, 128 * 128, device=device)
        
        return img_tensor, lidar_tensor, imu_tensor, audio_tensor
    
    def run_inference(self, version: str, image: Optional[np.ndarray] = None,
                     lidar: Optional[np.ndarray] = None, imu: Optional[np.ndarray] = None,
                     audio: Optional[np.ndarray] = None, compression_level: float = 0.8,
                     target_quality: float = 0.8, hidden_state: Optional[torch.Tensor] = None,
                     return_reconstruction: bool = False, return_metrics: bool = True) -> Dict[str, Any]:
        """Run inference with a loaded model"""
        import time
        
        if version not in self.loaded_models:
            raise ValueError(f"Model {version} not loaded")
        
        model_info = self.loaded_models[version]
        model = model_info.get("model")
        
        if model is None:
            raise ValueError(f"Model {version} architecture not available")
        
        start_time = time.time()
        
        try:
            # Prepare inputs
            img_tensor, lidar_tensor, imu_tensor, audio_tensor = self.prepare_inputs(
                image, lidar, imu, audio
            )
            
            # Run inference
            with torch.no_grad():
                outputs = model(
                    img_tensor, lidar_tensor, imu_tensor, audio_tensor,
                    hidden_state=hidden_state,
                    compression_level=compression_level,
                    target_quality=target_quality
                )
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Parse outputs
            compressed, temporal_out, predicted, recon_img, mu, logvar, adjusted_compression, predicted_quality = outputs
            
            # Convert to numpy/cpu for serialization
            result = {
                "compressed": compressed.cpu().numpy().tolist(),
                "temporal_out": temporal_out.cpu().numpy().tolist() if temporal_out is not None else None,
                "predicted": predicted.cpu().numpy().tolist() if predicted is not None else None,
                "inference_time_ms": inference_time_ms,
                "adjusted_compression": adjusted_compression.cpu().item() if isinstance(adjusted_compression, torch.Tensor) else adjusted_compression,
                "predicted_quality": predicted_quality.cpu().item() if isinstance(predicted_quality, torch.Tensor) else predicted_quality
            }
            
            # Add reconstruction if requested
            if return_reconstruction and recon_img is not None:
                recon_np = recon_img.cpu().numpy()
                if len(recon_np.shape) == 4:  # [B, C, H, W]
                    recon_np = recon_np.transpose(0, 2, 3, 1)  # [B, H, W, C]
                result["reconstructed_image"] = recon_np.tolist()
            
            # Calculate metrics
            if return_metrics:
                # Calculate compression ratio (simplified)
                input_size = img_tensor.numel() * 4 + lidar_tensor.numel() * 4 + \
                           imu_tensor.numel() * 4 + audio_tensor.numel() * 4
                output_size = compressed.numel() * 4
                compression_ratio = input_size / max(output_size, 1)
                
                result["metrics"] = {
                    "compression_ratio": compression_ratio,
                    "predicted_quality": result["predicted_quality"],
                    "adjusted_compression": result["adjusted_compression"]
                }
                result["compression_ratio"] = compression_ratio
            
            # Update stats
            self._update_stats(version, inference_time_ms, result.get("compression_ratio", 0.0),
                             result.get("predicted_quality", 0.0))
            
            return result
            
        except Exception as e:
            self.model_stats[version]["total_errors"] += 1
            raise RuntimeError(f"Inference failed: {str(e)}")
    
    def _update_stats(self, version: str, inference_time_ms: float,
                     compression_ratio: float, quality_score: float):
        """Update model statistics"""
        if version not in self.model_stats:
            return
        
        stats = self.model_stats[version]
        stats["total_inferences"] += 1
        stats["total_inference_time_ms"] += inference_time_ms
        stats["total_compression_ratio"] += compression_ratio
        stats["total_quality_score"] += quality_score
        stats["last_inference"] = datetime.utcnow()
    
    def get_model_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model"""
        if version not in self.loaded_models:
            return None
        
        model_info = self.loaded_models[version]
        stats = self.model_stats.get(version, {})
        
        # Calculate model size
        model_size_mb = 0.0
        if model_info.get("model"):
            param_size = sum(p.numel() * p.element_size() for p in model_info["model"].parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model_info["model"].buffers())
            model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        elif model_info.get("state_dict"):
            model_size_mb = sum(t.numel() * t.element_size() for t in model_info["state_dict"].values()) / (1024 ** 2)
        
        return {
            "version": version,
            "filename": model_info["filename"],
            "loaded_at": model_info["loaded_at"],
            "metadata": model_info.get("metadata", {}),
            "architecture": model_info.get("architecture"),
            "device": model_info.get("device"),
            "is_loaded": True,
            "model_size_mb": model_size_mb,
            "stats": {
                "total_inferences": stats.get("total_inferences", 0),
                "total_errors": stats.get("total_errors", 0),
                "average_inference_time_ms": (
                    stats.get("total_inference_time_ms", 0) / max(stats.get("total_inferences", 1), 1)
                ),
                "average_compression_ratio": (
                    stats.get("total_compression_ratio", 0) / max(stats.get("total_inferences", 1), 1)
                ),
                "average_quality_score": (
                    stats.get("total_quality_score", 0) / max(stats.get("total_inferences", 1), 1)
                ),
                "last_inference": stats.get("last_inference"),
                "cache_hits": stats.get("cache_hits", 0),
                "cache_misses": stats.get("cache_misses", 0)
            }
        }

