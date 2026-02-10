# Migration Notes: FastAPI to Django REST Framework

## Changes Made

1. **Framework Migration**: Migrated from FastAPI to Django REST Framework
2. **Dependency Management**: Added Poetry for dependency management
3. **Redis Pub/Sub**: Implemented Redis Pub/Sub for real-time messaging
4. **GStreamer/Deepstream**: Added GStreamer and NVIDIA Deepstream plugin support
5. **WebSocket**: Migrated from FastAPI WebSocket to Django Channels

## File Structure

```
backend/
├── backend/          # Django project
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py       # ASGI for Channels
│   └── routing.py   # WebSocket routing
├── api/              # API app
│   ├── views.py      # DRF views
│   ├── serializers.py
│   ├── urls.py
│   ├── connections.py  # MongoDB/Redis connections
│   ├── redis_pubsub.py  # Redis Pub/Sub
│   └── consumers.py     # WebSocket consumers
├── gstreamer/        # GStreamer/Deepstream app
│   ├── gst_pipeline.py
│   ├── deepstream_plugin.py
│   ├── views.py
│   └── urls.py
├── manage.py
├── pyproject.toml    # Poetry config
└── requirements.txt  # pip fallback
```

## API Endpoint Changes

| Old (FastAPI) | New (Django) |
|--------------|--------------|
| `/health` | `/api/health/` |
| `/api/nodes` | `/api/nodes/` |
| `/api/models` | `/api/models/` |
| `/api/deploy` | `/api/deploy/` |
| `/api/metrics` | `/api/metrics/` |
| `/api/stats` | `/api/stats/` |
| `/ws/metrics` | `/ws/metrics/` |

## Breaking Changes

1. **URLs**: All API endpoints now require trailing slashes
2. **WebSocket**: Uses Django Channels instead of FastAPI WebSocket
3. **Async**: Views use `asyncio.run()` wrapper for async operations
4. **Dependencies**: Now managed via Poetry instead of requirements.txt

## Migration Steps

1. Update frontend API URLs to include trailing slashes
2. Update WebSocket connections to use Django Channels format
3. Update environment variables (see README.md)
4. Run migrations: `python manage.py migrate`

## Old Files (Removed)

The following old FastAPI files have been removed after migration completion:

- `main.py` - Old FastAPI code (removed)
- `models.py` - Pydantic models (replaced by serializers.py, removed)
- `websocket_manager.py` - FastAPI WebSocket (replaced by consumers.py, removed)

