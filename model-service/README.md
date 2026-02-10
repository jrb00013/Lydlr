# Lydlr Model Service

Production-ready model hosting and inference service for Lydlr's multimodal compression models.

## Features

### 🚀 Core Functionality
- **Model Management**: Load, unload, and manage multiple model versions
- **Real-time Inference**: Fast inference with proper input validation
- **Batch Processing**: Efficient batch inference support
- **GPU Support**: Automatic GPU detection and utilization
- **Model Versioning**: Support for multiple model versions with metadata

### 📊 Monitoring & Analytics
- **Comprehensive Health Checks**: System and model health monitoring
- **Performance Metrics**: Inference time, compression ratios, quality scores
- **Inference History**: Track all inference requests in MongoDB
- **Model Statistics**: Per-model performance tracking

### 🔧 Advanced Features
- **Model Upload/Download**: RESTful API for model file management
- **Model Comparison**: A/B testing and version comparison
- **Caching**: Redis-based caching for improved performance
- **Error Handling**: Comprehensive error handling and logging
- **Async Processing**: Background task support for logging

## API Endpoints

### Health & Status
- `GET /` - Service information
- `GET /health` - Comprehensive health check
- `GET /stats` - Service statistics

### Model Management
- `GET /models` - List all models (loaded and available)
- `GET /models/{version}` - Get model information
- `GET /models/{version}/stats` - Get model statistics
- `POST /models/{version}/load` - Load a model
- `POST /models/{version}/unload` - Unload a model
- `POST /models/upload` - Upload a new model
- `GET /models/{version}/download` - Download a model file

### Inference
- `POST /models/{version}/inference` - Run single inference
- `POST /models/{version}/inference/batch` - Run batch inference

### Model Comparison
- `POST /models/compare` - Compare multiple model versions

### History
- `GET /inference/history` - Get inference history

## Request/Response Schemas

### Inference Request
```json
{
  "version": "1.0",
  "image": [[[0.0, 0.0, 0.0], ...]],  // Optional: [H, W, C]
  "lidar": [0.0, 0.0, ...],           // Optional: point cloud
  "imu": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  // Optional: 6 values
  "audio": [[0.0, ...]],              // Optional: spectrogram
  "compression_level": 0.8,           // 0.1-1.0
  "target_quality": 0.8,             // 0.0-1.0
  "return_reconstruction": false,
  "return_metrics": true
}
```

### Inference Response
```json
{
  "version": "1.0",
  "compressed": [[0.0, ...]],         // Compressed latent
  "temporal_out": [[0.0, ...]],       // Temporal features
  "predicted": [[0.0, ...]],          // Predicted features
  "reconstructed_image": [[[0.0, ...]]], // Optional
  "metrics": {
    "compression_ratio": 10.5,
    "predicted_quality": 0.85,
    "adjusted_compression": 0.82
  },
  "compression_ratio": 10.5,
  "inference_time_ms": 45.2,
  "timestamp": "2025-01-01T00:00:00"
}
```

## Configuration

### Environment Variables
- `MODEL_DIR` - Directory for model files (default: `/models`)
- `MONGODB_URL` - MongoDB connection string
- `REDIS_URL` - Redis connection string
- `DEVICE` - Device to use (`cuda`, `cpu`, `mps`) - auto-detected if not set

### Docker
The service is configured to run in Docker with:
- Port: `8001`
- Model directory mounted at `/models`
- Automatic model loading on startup

## Usage Examples

### Load a Model
```bash
curl -X POST http://localhost:8001/models/1.0/load
```

### Run Inference
```bash
curl -X POST http://localhost:8001/models/1.0/inference \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0",
    "image": [[[0.5, 0.5, 0.5]]],
    "compression_level": 0.8,
    "target_quality": 0.8
  }'
```

### Compare Models
```bash
curl -X POST http://localhost:8001/models/compare \
  -H "Content-Type: application/json" \
  -d '{
    "versions": ["1.0", "1.1"],
    "test_data": {
      "version": "1.0",
      "compression_level": 0.8
    },
    "metrics": ["compression_ratio", "quality", "latency"]
  }'
```

## Model Architecture

The service supports:
- **EnhancedMultimodalCompressor**: Main compression model with:
  - Enhanced VAE with ResNet18 backbone
  - Multimodal fusion with attention
  - Delta compression
  - Temporal transformer
  - Quality control

## Performance

- **Inference Speed**: ~20-50ms per inference (GPU), ~100-200ms (CPU)
- **Batch Processing**: Supports up to 100 items per batch
- **Concurrent Requests**: Handled via FastAPI async support
- **Memory**: Models loaded on-demand, GPU memory managed automatically

## Development

### Running Locally
```bash
cd model-service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Testing
```bash
# Health check
curl http://localhost:8001/health

# List models
curl http://localhost:8001/models

# Get stats
curl http://localhost:8001/stats
```

## Architecture

```
model-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── schemas.py           # Pydantic schemas
│   └── model_manager.py     # Model lifecycle management
├── Dockerfile
├── requirements.txt
└── README.md
```

## Error Handling

The service includes comprehensive error handling:
- HTTP exceptions with detailed error messages
- Database connection failures (graceful degradation)
- Redis connection failures (graceful degradation)
- Model loading errors with detailed logs
- Inference errors with stack traces

## Logging

All operations are logged with appropriate levels:
- `INFO`: Normal operations, model loading
- `WARNING`: Non-critical issues (DB/Redis unavailable)
- `ERROR`: Critical errors (inference failures, model loading failures)

## Security

- CORS configured (adjust for production)
- Input validation via Pydantic
- File upload validation
- Error messages don't expose sensitive information

## Future Enhancements

- [ ] Model quantization support
- [ ] ONNX export/import
- [ ] Model serving with TensorRT
- [ ] Prometheus metrics export
- [ ] WebSocket support for streaming inference
- [ ] Model fine-tuning endpoints

