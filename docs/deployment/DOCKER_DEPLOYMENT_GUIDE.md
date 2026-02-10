#  Lydlr Full-Stack Deployment Guide

## Complete Dockerized System with Frontend Interface

This guide will help you deploy the complete Lydlr system with:
- **Frontend Web Interface** (React)
- **Backend API** (FastAPI)
- **Model Hosting Service**
- **MongoDB Database** (NoSQL)
- **Redis Cache**
- **ROS2 Runtime**
- **Nginx Reverse Proxy**

---

##  Prerequisites

- Docker (20.10+)
- Docker Compose (2.0+)
- 8GB+ RAM recommended
- 10GB+ free disk space

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Nginx (Port 80)                      │
│                  Reverse Proxy & Load Balancer          │
└────────────┬────────────────────────────────────────────┘
             │
     ┌───────┴────────┐
     │                │
┌────▼─────┐    ┌────▼─────┐
│ Frontend │    │ Backend  │
│  React   │    │  FastAPI │
│ Port 3000│    │ Port 8000│
└──────────┘    └────┬─────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼────┐  ┌───▼────┐  ┌───▼────────┐
   │ MongoDB │  │ Redis  │  │  Model     │
   │ NoSQL   │  │ Cache  │  │  Service   │
   │ DB      │  │        │  │  Port 8001 │
   └─────────┘  └────────┘  └────────────┘
                                  │
                            ┌─────▼──────┐
                            │ ROS2       │
                            │ Runtime    │
                            │ Edge Nodes │
                            └────────────┘
```

---

##  Quick Start

### Step 1: Environment Setup

Copy the environment template:

```bash
cd /mnt/c/Users/josep/Documents/Lydlr/Lydlr
cp .env.example .env
```

Edit `.env` if needed (default values should work).

### Step 2: Build and Launch

Build and start all services:

```bash
docker-compose up --build
```

Or run in detached mode:

```bash
docker-compose up --build -d
```

### Step 3: Access the System

- **Frontend Interface**: http://localhost (or http://localhost:3000)
- **Backend API Docs**: http://localhost:8000/docs
- **Model Service**: http://localhost:8001
- **MongoDB**: localhost:27017
- **Redis**: localhost:6380 (host port, container uses 6379)

---

##  Frontend Interface Features

### Dashboard
- **Real-time metrics visualization**
- Active nodes status
- Average compression ratio
- Latency monitoring
- Quality scores

### Nodes View
- List all edge nodes
- Node status (active/inactive/error)
- Performance metrics per node
- Node control (start/stop/restart)

### Models View
- Browse available models
- Upload new models (.pth files)
- Model metadata and versions
- Size and creation date

### Metrics View
- Historical performance data
- Interactive charts
- Filter by node
- Compression, latency, quality trends

### Deployment View
- Deploy models to nodes
- Select target nodes
- Deployment history
- Status tracking

---

##  Service Details

### Frontend (React + Recharts)
- Modern, responsive UI
- Real-time WebSocket updates
- Material-UI components
- Interactive charts

### Backend (FastAPI)
- RESTful API
- WebSocket support for real-time data
- MongoDB for persistent storage
- Redis for caching

### Model Service
- Dedicated ML model hosting
- Model loading/unloading
- Version management
- Inference endpoints

### MongoDB
- Store nodes, models, metrics, deployments
- Indexed for fast queries
- Automatic initialization

### Redis
- Real-time data caching
- WebSocket message broker
- Fast key-value store

### ROS2 Runtime
- Edge compression nodes
- Synthetic data publishers
- Model deployment manager
- Distributed coordinator

---

##  Docker Commands

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f mongodb
```

### Restart Service
```bash
docker-compose restart backend
```

### Rebuild Service
```bash
docker-compose up --build backend
```

### Check Status
```bash
docker-compose ps
```

### Execute Commands in Container
```bash
# Backend shell
docker-compose exec backend bash

# MongoDB shell
docker-compose exec mongodb mongosh -u lydlr -p lydlr_password lydlr_db
```

---

##  Development Workflow

### Hot Reloading

Both frontend and backend support hot reloading:

**Frontend:**
- Edit files in `frontend/src/`
- Changes auto-reload in browser

**Backend:**
- Edit files in `backend/`
- Uvicorn auto-reloads

### Installing Dependencies

**Frontend:**
```bash
docker-compose exec frontend npm install <package>
```

**Backend:**
```bash
docker-compose exec backend pip install <package>
# Then add to backend/requirements.txt
```

### Database Access

**MongoDB:**
```bash
docker-compose exec mongodb mongosh -u lydlr -p lydlr_password lydlr_db
```

**Redis:**
```bash
docker-compose exec redis redis-cli
```

---

##  API Endpoints

### Nodes
- `GET /api/nodes` - List all nodes
- `GET /api/nodes/{node_id}` - Get node details
- `POST /api/nodes/{node_id}/config` - Update node config
- `POST /api/nodes/{node_id}/start` - Start node
- `POST /api/nodes/{node_id}/stop` - Stop node
- `POST /api/nodes/{node_id}/restart` - Restart node

### Models
- `GET /api/models` - List models
- `POST /api/models/upload` - Upload model

### Deployments
- `POST /api/deploy` - Deploy model to nodes
- `GET /api/deployments` - Deployment history

### Metrics
- `GET /api/metrics` - Get metrics
- `POST /api/metrics` - Store metrics

### System
- `GET /api/stats` - System statistics
- `GET /health` - Health check

### WebSocket
- `WS /ws/metrics` - Real-time metrics stream

---

##  Security Notes

**For Production:**

1. Change default passwords in `.env`
2. Use HTTPS with SSL certificates
3. Set proper CORS origins
4. Enable authentication
5. Use secrets management
6. Configure firewall rules

---

##  Troubleshooting

### Frontend Can't Connect to Backend

Check backend is running:
```bash
docker-compose logs backend
curl http://localhost:8000/health
```

### MongoDB Connection Failed

Check MongoDB is running:
```bash
docker-compose logs mongodb
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"
```

### Port Already in Use

Stop conflicting services:
```bash
# Check what's using port 80
sudo lsof -i :80

# Or change ports in docker-compose.yml
```

### Models Not Loading

Check model directory:
```bash
docker-compose exec backend ls -la /app/models
```

### ROS2 Nodes Not Starting

Check ROS2 runtime logs:
```bash
docker-compose logs ros2-runtime
```

---

##  Performance Tuning

### For Production:

1. **Use production builds:**
   ```bash
   # Frontend
   docker-compose exec frontend npm run build
   ```

2. **Increase worker processes:**
   Edit `docker-compose.yml`:
   ```yaml
   backend:
     command: uvicorn backend.main:app --host 0.0.0.0 --workers 4
   ```

3. **Scale services:**
   ```bash
   docker-compose up --scale backend=3
   ```

4. **Add resource limits:**
   ```yaml
   services:
     backend:
       deploy:
         resources:
           limits:
             cpus: '2'
             memory: 4G
   ```

---

##  Next Steps

1. **Access the frontend** at http://localhost
2. **Upload trained models** in the Models view
3. **Start edge nodes** via ROS2 runtime
4. **Deploy models** to nodes in Deployment view
5. **Monitor performance** in real-time on Dashboard

---

##  Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [ROS2 Documentation](https://docs.ros.org/en/humble/)

---

**Enjoy your revolutionary real-time edge compression system! **

