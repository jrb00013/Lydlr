# Lydlr Backend - Django REST Framework

This is the Django REST Framework backend for the Lydlr Revolutionary Compression System.

## Migration from FastAPI

The backend has been migrated from FastAPI to Django REST Framework.

## Features

- **Django REST Framework** - RESTful API endpoints
- **Channels** - WebSocket support for real-time updates
- **Redis Pub/Sub** - Real-time messaging between services
- **MongoDB** - Document database via Motor (async)
- **GStreamer Integration** - Video processing pipelines
- **NVIDIA Deepstream** - AI inference support (optional)

## Setup

### Using Poetry (Recommended)

```bash
cd backend
poetry install
poetry shell
python manage.py migrate
python manage.py runserver
```

### Using pip

```bash
cd backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## API Endpoints

- `GET /api/` - Root endpoint
- `GET /api/health/` - Health check
- `GET /api/nodes/` - List all nodes
- `GET /api/nodes/<node_id>/` - Get node details
- `POST /api/nodes/<node_id>/` - Update node config
- `GET /api/models/` - List available models
- `POST /api/deploy/` - Deploy model to nodes
- `GET /api/deployments/` - Get deployment history
- `GET /api/metrics/` - Get metrics
- `POST /api/metrics/` - Store metrics
- `GET /api/stats/` - System statistics
- `POST /api/nodes/<node_id>/<action>/` - Control node (start/stop/restart)

## WebSocket

- `ws://localhost:8000/ws/metrics/` - Real-time metrics updates

## Redis Pub/Sub Channels

- `node_config_update` - Node configuration updates
- `deployment` - Model deployment events
- `metrics_update` - Metrics updates
- `node_command` - Node control commands
- `system_event` - System events

## GStreamer/Deepstream

- `GET /gstreamer/status/` - GStreamer status
- `GET /gstreamer/deepstream/status/` - Deepstream status
- `POST /gstreamer/pipeline/compression/` - Create compression pipeline
- `POST /gstreamer/deepstream/inference/` - Run Deepstream inference

## Environment Variables

- `MONGODB_URL` - MongoDB connection string
- `REDIS_URL` - Redis connection string
- `REDIS_HOST` - Redis host (default: redis)
- `REDIS_PORT` - Redis port (default: 6379, internal container port)
- `REDIS_URL` - Redis connection URL (default: redis://redis:6379 for Docker)
- `MODEL_DIR` - Model directory path
- `DEBUG` - Django debug mode
- `SECRET_KEY` - Django secret key
- `DEEPSTREAM_ENABLED` - Enable Deepstream (default: False)
- `NVIDIA_DEEPSTREAM_PATH` - Deepstream installation path

## Docker

The backend is containerized and can be run with:

```bash
docker-compose up backend
```

Or build manually:

```bash
docker build -f backend/Dockerfile -t lydlr-backend .
docker run -p 8000:8000 lydlr-backend
```

