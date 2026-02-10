"""
Model Hosting Service
Dedicated service for hosting and serving ML models
Fully featured production-ready service
"""
import asyncio
import logging
import psutil
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis
import torch
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiofiles
import hashlib
import time

from app.schemas import (
    InferenceRequest, InferenceResponse, BatchInferenceRequest, BatchInferenceResponse,
    ModelInfo, ModelUploadRequest, ModelStats, ServiceHealth,
    ModelComparisonRequest, ModelComparisonResponse
)
from app.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lydlr Model Service",
    description="Model Hosting and Inference Service - Production Ready",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Global services
mongodb_client: Optional[AsyncIOMotorClient] = None
redis_client: Optional[aioredis.Redis] = None
db = None
model_manager: Optional[ModelManager] = None

# Inference queue for async processing
inference_queue = asyncio.Queue()
inference_workers = []


@app.on_event("startup")
async def startup():
    global mongodb_client, redis_client, db, model_manager
    
    logger.info("🚀 Starting Model Service...")
    
    # MongoDB
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://lydlr:lydlr_password@localhost:27017/lydlr_db?authSource=admin")
    try:
        mongodb_client = AsyncIOMotorClient(mongodb_url, serverSelectionTimeoutMS=5000)
        db = mongodb_client.lydlr_db
        # Test connection
        await mongodb_client.admin.command('ping')
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.warning(f"⚠️  MongoDB connection failed: {e}")
        db = None
    
    # Redis
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    try:
        redis_client = aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=5)
        await redis_client.ping()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.warning(f"⚠️  Redis connection failed: {e}")
        redis_client = None
    
    # Initialize model manager
    device = os.getenv("DEVICE", None)
    model_manager = ModelManager(MODEL_DIR, device=device)
    logger.info(f"✅ Model Manager initialized on device: {model_manager.device}")
    
    # Load available models
    await load_available_models()
    
    logger.info("✅ Model Service Started Successfully")


@app.on_event("shutdown")
async def shutdown():
    global mongodb_client, redis_client, model_manager
    
    logger.info("🛑 Shutting down Model Service...")
    
    if model_manager:
        # Unload all models
        for version in list(model_manager.loaded_models.keys()):
            model_manager.unload_model(version)
    
    if mongodb_client:
        mongodb_client.close()
    
    if redis_client:
        await redis_client.close()
    
    logger.info("✅ Model Service Shutdown Complete")


async def load_available_models():
    """Load all available models into memory"""
    if not MODEL_DIR.exists():
        logger.warning(f"Model directory {MODEL_DIR} does not exist")
        return
    
    model_files = list(MODEL_DIR.glob("*.pth"))
    logger.info(f"Found {len(model_files)} model files")
    
    for model_file in model_files:
        try:
            # Extract version from filename
            version = None
            if "_v" in model_file.stem:
                version = model_file.stem.split("_v")[1]
            else:
                # Try to extract from other patterns
                parts = model_file.stem.split("_")
                for i, part in enumerate(parts):
                    if part == "v" and i + 1 < len(parts):
                        version = parts[i + 1]
                        break
            
            if not version:
                version = model_file.stem
            
            # Determine architecture
            architecture = "EnhancedMultimodalCompressor"
            if "sensor_motor" in model_file.stem.lower():
                architecture = "SensorMotorCompressor"
            
            # Load model
            success = model_manager.load_model(version, architecture=architecture)
            if success:
                logger.info(f"✅ Loaded model: {version} ({architecture})")
            else:
                logger.warning(f"⚠️  Failed to load model: {version}")
        except Exception as e:
            logger.error(f"❌ Error loading {model_file.name}: {e}")


async def log_inference_to_db(version: str, request_data: Dict, response_data: Dict, inference_time_ms: float):
    """Log inference to MongoDB"""
    if db is None:
        return
    
    try:
        inference_log = {
            "version": version,
            "request": request_data,
            "response": {
                "compression_ratio": response_data.get("compression_ratio"),
                "predicted_quality": response_data.get("predicted_quality"),
                "inference_time_ms": inference_time_ms
            },
            "timestamp": datetime.utcnow()
        }
        await db.inference_logs.insert_one(inference_log)
    except Exception as e:
        logger.warning(f"Failed to log inference to DB: {e}")


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Lydlr Model Hosting Service",
        "version": "2.0.0",
        "loaded_models": list(model_manager.loaded_models.keys()) if model_manager else [],
        "device": str(model_manager.device) if model_manager else "unknown",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=ServiceHealth)
async def health_check():
    """Comprehensive health check"""
    device_info = model_manager.get_device_info() if model_manager else {}
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    return ServiceHealth(
        status="healthy",
        models_loaded=len(model_manager.loaded_models) if model_manager else 0,
        total_models=len(list(MODEL_DIR.glob("*.pth"))) if MODEL_DIR.exists() else 0,
        device=device_info.get("device", "unknown"),
        gpu_available=device_info.get("gpu_available", False),
        gpu_memory_mb=device_info.get("gpu_memory_mb"),
        cpu_usage_percent=cpu_percent,
        memory_usage_mb=memory.used / (1024 ** 2)
    )


# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.get("/models", response_model=Dict[str, List[ModelInfo]])
async def list_models():
    """List all available and loaded models"""
    loaded = []
    if model_manager:
        for version in model_manager.loaded_models.keys():
            info = model_manager.get_model_info(version)
            if info:
                loaded.append(ModelInfo(**info))
    
    # Find all model files
    available = []
    if MODEL_DIR.exists():
        for model_file in MODEL_DIR.glob("*.pth"):
            version = model_file.stem.split("_v")[1] if "_v" in model_file.stem else model_file.stem
            if version not in [m.version for m in loaded]:
                metadata_file = MODEL_DIR / f"metadata_{version}.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                available.append(ModelInfo(
                    version=version,
                    filename=model_file.name,
                    loaded_at=datetime.fromtimestamp(model_file.stat().st_mtime),
                    metadata=metadata,
                    is_loaded=False,
                    model_size_mb=model_file.stat().st_size / (1024 ** 2)
                ))
    
    return {
        "loaded": loaded,
        "available": available
    }


@app.get("/models/{version}", response_model=ModelInfo)
async def get_model_info(version: str):
    """Get detailed information about a specific model"""
    if not model_manager or version not in model_manager.loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {version} not loaded")
    
    info = model_manager.get_model_info(version)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model {version} not found")
    
    return ModelInfo(**info)


@app.get("/models/{version}/stats", response_model=ModelStats)
async def get_model_stats(version: str):
    """Get statistics for a specific model"""
    if not model_manager or version not in model_manager.loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {version} not loaded")
    
    info = model_manager.get_model_info(version)
    if not info or "stats" not in info:
        raise HTTPException(status_code=404, detail=f"Stats not available for model {version}")
    
    stats = info["stats"]
    return ModelStats(
        version=version,
        **stats
    )


@app.post("/models/{version}/load")
async def load_model(version: str, architecture: Optional[str] = None):
    """Load a specific model version"""
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    if version in model_manager.loaded_models:
        return {"status": "already_loaded", "version": version}
    
    # Find model file
    model_file = None
    for pattern in [
        f"lydlr_compressor_v{version}.pth",  # New naming convention
        f"compressor_v{version}.pth",        # Old naming convention (backward compatibility)
        f"lydlr_sensor_motor_v{version}.pth",
        f"sensor_motor_v{version}.pth",
        f"*_v{version}.pth"  # Fallback pattern
    ]:
        matches = list(MODEL_DIR.glob(pattern))
        if matches:
            model_file = matches[0]
            break
    
    if not model_file or not model_file.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found for version {version}")
    
    # Determine architecture
    if not architecture:
        if "sensor_motor" in model_file.stem.lower():
            architecture = "SensorMotorCompressor"
        else:
            architecture = "EnhancedMultimodalCompressor"
    
    # Load model
    success = model_manager.load_model(version, architecture=architecture)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to load model {version}")
    
    # Cache in Redis
    if redis_client:
        await redis_client.setex(
            f"model:loaded:{version}",
            3600,
            json.dumps({"status": "loaded", "timestamp": datetime.utcnow().isoformat()})
        )
    
    logger.info(f"✅ Model {version} loaded successfully")
    return {"status": "success", "version": version, "architecture": architecture}


@app.post("/models/{version}/unload")
async def unload_model(version: str):
    """Unload a model from memory"""
    if not model_manager or version not in model_manager.loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {version} not loaded")
    
    success = model_manager.unload_model(version)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to unload model {version}")
    
    # Remove from Redis cache
    if redis_client:
        await redis_client.delete(f"model:loaded:{version}")
    
    logger.info(f"✅ Model {version} unloaded")
    return {"status": "success", "version": version}


# ============================================================================
# Inference Endpoints
# ============================================================================

@app.post("/models/{version}/inference", response_model=InferenceResponse)
async def run_inference(
    version: str,
    request: InferenceRequest,
    background_tasks: BackgroundTasks
):
    """Run inference with a loaded model"""
    if not model_manager or version not in model_manager.loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {version} not loaded")
    
    if request.version != version:
        raise HTTPException(status_code=400, detail="Version mismatch between URL and request")
    
    start_time = time.time()
    
    try:
        # Convert inputs to numpy arrays
        image = np.array(request.image) if request.image else None
        lidar = np.array(request.lidar) if request.lidar else None
        imu = np.array(request.imu) if request.imu else None
        audio = np.array(request.audio) if request.audio else None
        
        # Prepare hidden state if provided
        hidden_state = None
        if request.hidden_state:
            hidden_state = torch.tensor(request.hidden_state, device=model_manager.device)
        
        # Run inference
        result = model_manager.run_inference(
            version=version,
            image=image,
            lidar=lidar,
            imu=imu,
            audio=audio,
            compression_level=request.compression_level,
            target_quality=request.target_quality,
            hidden_state=hidden_state,
            return_reconstruction=request.return_reconstruction,
            return_metrics=request.return_metrics
        )
        
        inference_time_ms = result["inference_time_ms"]
        
        # Log to database in background
        if db:
            background_tasks.add_task(
                log_inference_to_db,
                version,
                request.dict(),
                result,
                inference_time_ms
            )
        
        return InferenceResponse(
            version=version,
            compressed=result["compressed"],
            temporal_out=result.get("temporal_out"),
            predicted=result.get("predicted"),
            reconstructed_image=result.get("reconstructed_image"),
            metrics=result.get("metrics"),
            compression_ratio=result.get("compression_ratio"),
            inference_time_ms=inference_time_ms
        )
        
    except Exception as e:
        logger.error(f"Inference error for model {version}: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/models/{version}/inference/batch", response_model=BatchInferenceResponse)
async def run_batch_inference(
    version: str,
    request: BatchInferenceRequest,
    background_tasks: BackgroundTasks
):
    """Run batch inference with a loaded model"""
    if not model_manager or version not in model_manager.loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {version} not loaded")
    
    if len(request.batch) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100")
    
    start_time = time.time()
    results = []
    
    try:
        for batch_item in request.batch:
            if batch_item.version != version:
                continue  # Skip mismatched versions
            
            # Convert inputs
            image = np.array(batch_item.image) if batch_item.image else None
            lidar = np.array(batch_item.lidar) if batch_item.lidar else None
            imu = np.array(batch_item.imu) if batch_item.imu else None
            audio = np.array(batch_item.audio) if batch_item.audio else None
            
            # Run inference
            result = model_manager.run_inference(
                version=version,
                image=image,
                lidar=lidar,
                imu=imu,
                audio=audio,
                compression_level=batch_item.compression_level,
                target_quality=batch_item.target_quality,
                return_reconstruction=request.return_reconstruction,
                return_metrics=request.return_metrics
            )
            
            results.append(InferenceResponse(
                version=version,
                compressed=result["compressed"],
                temporal_out=result.get("temporal_out"),
                predicted=result.get("predicted"),
                reconstructed_image=result.get("reconstructed_image"),
                metrics=result.get("metrics"),
                compression_ratio=result.get("compression_ratio"),
                inference_time_ms=result["inference_time_ms"]
            ))
        
        total_time_ms = (time.time() - start_time) * 1000
        average_time_ms = total_time_ms / len(results) if results else 0
        
        return BatchInferenceResponse(
            version=version,
            results=results,
            batch_size=len(results),
            total_time_ms=total_time_ms,
            average_time_ms=average_time_ms
        )
        
    except Exception as e:
        logger.error(f"Batch inference error for model {version}: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


# ============================================================================
# Model Upload/Download Endpoints
# ============================================================================

@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    version: Optional[str] = None,
    metadata: Optional[str] = None
):
    """Upload a new model file"""
    if not file.filename.endswith('.pth'):
        raise HTTPException(status_code=400, detail="Only .pth files are supported")
    
    # Determine version
    if not version:
        # Try to extract from filename
        if "_v" in file.filename:
            version = file.filename.split("_v")[1].replace(".pth", "")
        else:
            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Save model file
    model_path = MODEL_DIR / file.filename
    async with aiofiles.open(model_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Save metadata if provided
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
            metadata_path = MODEL_DIR / f"metadata_{version}.json"
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata_dict, indent=2))
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON metadata provided for {version}")
    
    logger.info(f"✅ Model uploaded: {file.filename} (version: {version})")
    return {
        "status": "success",
        "version": version,
        "filename": file.filename,
        "size_mb": model_path.stat().st_size / (1024 ** 2)
    }


@app.get("/models/{version}/download")
async def download_model(version: str):
    """Download a model file"""
    # Find model file
    model_file = None
    for pattern in [
        f"lydlr_compressor_v{version}.pth",  # New naming convention
        f"compressor_v{version}.pth",        # Old naming convention (backward compatibility)
        f"lydlr_sensor_motor_v{version}.pth",
        f"sensor_motor_v{version}.pth",
        f"*_v{version}.pth"  # Fallback pattern
    ]:
        matches = list(MODEL_DIR.glob(pattern))
        if matches:
            model_file = matches[0]
            break
    
    if not model_file or not model_file.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found for version {version}")
    
    return FileResponse(
        path=model_file,
        filename=model_file.name,
        media_type="application/octet-stream"
    )


# ============================================================================
# Model Comparison & A/B Testing
# ============================================================================

@app.post("/models/compare", response_model=ModelComparisonResponse)
async def compare_models(request: ModelComparisonRequest):
    """Compare multiple model versions on the same test data"""
    if len(request.versions) < 2:
        raise HTTPException(status_code=400, detail="At least 2 versions required for comparison")
    
    results = {}
    comparison_metrics = {}
    
    for version in request.versions:
        if not model_manager or version not in model_manager.loaded_models:
            results[version] = {"error": "Model not loaded"}
            continue
        
        try:
            # Convert test data
            image = np.array(request.test_data.image) if request.test_data.image else None
            lidar = np.array(request.test_data.lidar) if request.test_data.lidar else None
            imu = np.array(request.test_data.imu) if request.test_data.imu else None
            audio = np.array(request.test_data.audio) if request.test_data.audio else None
            
            # Run inference
            result = model_manager.run_inference(
                version=version,
                image=image,
                lidar=lidar,
                imu=imu,
                audio=audio,
                compression_level=request.test_data.compression_level,
                target_quality=request.test_data.target_quality,
                return_metrics=True
            )
            
            results[version] = {
                "compression_ratio": result.get("compression_ratio", 0),
                "predicted_quality": result.get("predicted_quality", 0),
                "inference_time_ms": result["inference_time_ms"],
                "adjusted_compression": result.get("adjusted_compression", 0)
            }
        except Exception as e:
            results[version] = {"error": str(e)}
    
    # Determine best version
    best_version = None
    if "compression_ratio" in request.metrics:
        best_version = max(
            [v for v in request.versions if v in results and "error" not in results[v]],
            key=lambda v: results[v].get("compression_ratio", 0)
        )
    elif "quality" in request.metrics:
        best_version = max(
            [v for v in request.versions if v in results and "error" not in results[v]],
            key=lambda v: results[v].get("predicted_quality", 0)
        )
    elif "latency" in request.metrics:
        best_version = min(
            [v for v in request.versions if v in results and "error" not in results[v]],
            key=lambda v: results[v].get("inference_time_ms", float('inf'))
        )
    
    # Calculate comparison metrics
    valid_results = {v: r for v, r in results.items() if "error" not in r}
    if valid_results:
        comparison_metrics = {
            "avg_compression_ratio": np.mean([r.get("compression_ratio", 0) for r in valid_results.values()]),
            "avg_quality": np.mean([r.get("predicted_quality", 0) for r in valid_results.values()]),
            "avg_latency_ms": np.mean([r.get("inference_time_ms", 0) for r in valid_results.values()]),
            "best_compression": max(valid_results.items(), key=lambda x: x[1].get("compression_ratio", 0))[0],
            "best_quality": max(valid_results.items(), key=lambda x: x[1].get("predicted_quality", 0))[0],
            "best_latency": min(valid_results.items(), key=lambda x: x[1].get("inference_time_ms", float('inf')))[0]
        }
    
    return ModelComparisonResponse(
        versions=request.versions,
        results=results,
        best_version=best_version,
        comparison_metrics=comparison_metrics
    )


# ============================================================================
# Statistics & Monitoring
# ============================================================================

@app.get("/stats")
async def get_stats():
    """Get comprehensive service statistics"""
    stats = {
        "models_loaded": len(model_manager.loaded_models) if model_manager else 0,
        "model_versions": list(model_manager.loaded_models.keys()) if model_manager else [],
        "total_size_mb": 0.0,
        "device_info": model_manager.get_device_info() if model_manager else {}
    }
    
    # Calculate total model size
    if MODEL_DIR.exists():
        total_size = sum(
            f.stat().st_size for f in MODEL_DIR.glob("*.pth")
        )
        stats["total_size_mb"] = total_size / (1024 ** 2)
    
    # Add per-model stats
    if model_manager:
        stats["model_stats"] = {}
        for version in model_manager.loaded_models.keys():
            info = model_manager.get_model_info(version)
            if info and "stats" in info:
                stats["model_stats"][version] = info["stats"]
    
    # System stats
    stats["system"] = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_mb": psutil.virtual_memory().used / (1024 ** 2),
        "memory_total_mb": psutil.virtual_memory().total / (1024 ** 2)
    }
    
    return stats


@app.get("/inference/history")
async def get_inference_history(version: Optional[str] = None, limit: int = 100):
    """Get inference history from database"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    query = {}
    if version:
        query["version"] = version
    
    cursor = db.inference_logs.find(query).sort("timestamp", -1).limit(limit)
    history = await cursor.to_list(length=limit)
    
    # Convert ObjectId to string
    for item in history:
        item["_id"] = str(item["_id"])
        if "timestamp" in item and isinstance(item["timestamp"], datetime):
            item["timestamp"] = item["timestamp"].isoformat()
    
    return {"history": history, "count": len(history)}


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
