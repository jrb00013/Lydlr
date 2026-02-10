"""
Pydantic schemas for Model Service API
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import numpy as np


class InferenceRequest(BaseModel):
    """Request schema for model inference"""
    version: str = Field(..., description="Model version to use")
    image: Optional[List[List[List[float]]]] = Field(None, description="Image data as nested list [H, W, C]")
    lidar: Optional[List[float]] = Field(None, description="LiDAR point cloud data")
    imu: Optional[List[float]] = Field(None, description="IMU sensor data (6 values)")
    audio: Optional[List[List[float]]] = Field(None, description="Audio spectrogram data")
    compression_level: float = Field(0.8, ge=0.1, le=1.0, description="Compression level (0.1-1.0)")
    target_quality: float = Field(0.8, ge=0.0, le=1.0, description="Target quality (0.0-1.0)")
    hidden_state: Optional[List[List[float]]] = Field(None, description="Previous hidden state for temporal models")
    return_reconstruction: bool = Field(False, description="Whether to return reconstructed image")
    return_metrics: bool = Field(True, description="Whether to return quality metrics")
    
    @validator('imu')
    def validate_imu(cls, v):
        if v is not None and len(v) != 6:
            raise ValueError("IMU data must have exactly 6 values")
        return v


class BatchInferenceRequest(BaseModel):
    """Request schema for batch inference"""
    version: str = Field(..., description="Model version to use")
    batch: List[InferenceRequest] = Field(..., min_items=1, max_items=100, description="List of inference requests")
    return_reconstruction: bool = Field(False, description="Whether to return reconstructed images")
    return_metrics: bool = Field(True, description="Whether to return quality metrics")


class InferenceResponse(BaseModel):
    """Response schema for model inference"""
    version: str
    compressed: List[List[float]] = Field(..., description="Compressed latent representation")
    temporal_out: Optional[List[List[float]]] = Field(None, description="Temporal output features")
    predicted: Optional[List[List[float]]] = Field(None, description="Predicted features")
    reconstructed_image: Optional[List[List[List[float]]]] = Field(None, description="Reconstructed image")
    metrics: Optional[Dict[str, float]] = Field(None, description="Quality metrics")
    compression_ratio: Optional[float] = Field(None, description="Compression ratio achieved")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchInferenceResponse(BaseModel):
    """Response schema for batch inference"""
    version: str
    results: List[InferenceResponse]
    batch_size: int
    total_time_ms: float
    average_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfo(BaseModel):
    """Model information schema"""
    version: str
    filename: str
    loaded_at: datetime
    metadata: Dict[str, Any] = {}
    architecture: Optional[str] = None
    input_shape: Optional[Dict[str, Any]] = None
    output_shape: Optional[Dict[str, Any]] = None
    model_size_mb: Optional[float] = None
    device: Optional[str] = None
    is_loaded: bool = True


class ModelUploadRequest(BaseModel):
    """Request schema for model upload"""
    version: str = Field(..., description="Model version identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model metadata")
    architecture: Optional[str] = Field(None, description="Model architecture name")
    input_shape: Optional[Dict[str, Any]] = Field(None, description="Expected input shapes")


class ModelStats(BaseModel):
    """Model statistics schema"""
    version: str
    total_inferences: int = 0
    total_errors: int = 0
    average_inference_time_ms: float = 0.0
    average_compression_ratio: float = 0.0
    average_quality_score: float = 0.0
    last_inference: Optional[datetime] = None
    cache_hits: int = 0
    cache_misses: int = 0


class ServiceHealth(BaseModel):
    """Service health check schema"""
    status: str
    models_loaded: int
    total_models: int
    device: str
    gpu_available: bool
    gpu_memory_mb: Optional[float] = None
    cpu_usage_percent: float
    memory_usage_mb: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelComparisonRequest(BaseModel):
    """Request schema for comparing model versions"""
    versions: List[str] = Field(..., min_items=2, max_items=5, description="Model versions to compare")
    test_data: InferenceRequest = Field(..., description="Test data for comparison")
    metrics: List[str] = Field(["compression_ratio", "quality", "latency"], description="Metrics to compare")


class ModelComparisonResponse(BaseModel):
    """Response schema for model comparison"""
    versions: List[str]
    results: Dict[str, Dict[str, Any]] = Field(..., description="Results per version")
    best_version: Optional[str] = Field(None, description="Best performing version")
    comparison_metrics: Dict[str, Any] = Field(..., description="Comparison summary")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

