#  Lydlr - Complete Revolutionary Compression System

## Full-Stack Real-Time Edge Compression with Web Interface

---

##  Table of Contents

1. [System Overview](#system-overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Features](#features)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Documentation](#documentation)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

---

##  System Overview

**Lydlr** is a revolutionary real-time edge compression system that combines:

- **Advanced Neural Compression** (VAE, Transformers, Attention Mechanisms)
- **Dynamic Model Deployment** (Hot-swapping, A/B testing, Rollback)
- **Real-Time Python Script Execution** on edge nodes
- **Multimodal Data Support** (Image, LiDAR, IMU, Audio, Motor)
- **Full-Stack Web Interface** (React + FastAPI)
- **NoSQL Database** (MongoDB)
- **Distributed Coordination** across nodes
- **Containerized Deployment** (Docker + Docker Compose)

**Revolutionary because:**
-  **Real-time model training** on synthetic data
-  **Hot-swappable models** on live nodes without downtime
-  **Dynamic Python scripts** executed at runtime
-  **Multimodal compression** for any sensor/motor data
-  **Web-based management** with live monitoring
-  **Fully dockerized** for easy deployment

---

##  Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 10GB+ disk space

### Launch in 3 Commands

```bash
cd /mnt/c/Users/josep/Documents/Lydlr/Lydlr

# Start the entire system
./start-lydlr.sh --build -d

# Check status
./start-lydlr.sh --status
```

### Access Points

- **Web Interface:** http://localhost
- **API Docs:** http://localhost:8000/docs
- **Model Service:** http://localhost:8001
- **MongoDB:** mongodb://localhost:27017
- **Redis:** localhost:6380 (host port, container uses 6379)

---

##  Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   WEB INTERFACE (React)                 │
│  Dashboard | Nodes | Models | Metrics | Deployment     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 NGINX REVERSE PROXY                     │
└────────┬───────────────────────────────────────┬────────┘
         │                                       │
┌────────▼──────────┐                  ┌────────▼─────────┐
│  BACKEND API      │                  │  MODEL SERVICE   │
│  (FastAPI)        │◄────────────────►│  (FastAPI)       │
│  - REST API       │                  │  - Model Hosting │
│  - WebSocket      │                  │  - Inference     │
│  - Node Control   │                  │  - Versioning    │
└────────┬──────────┘                  └────────┬─────────┘
         │                                       │
    ┌────┼───────────────────────────────────────┼────┐
    │    │                                       │    │
┌───▼────▼────┐  ┌──────────┐  ┌────────────┐  │    │
│  MongoDB    │  │  Redis   │  │ ROS2       │  │    │
│  (NoSQL DB) │  │  (Cache) │  │ Runtime    │◄─┘    │
│             │  │          │  │            │       │
│ - Nodes     │  │ - Metrics│  │ - Edge     │       │
│ - Models    │  │ - Cache  │  │   Nodes    │       │
│ - Metrics   │  │ - PubSub │  │ - Synthetic│       │
│ - Deploy    │  │          │  │   Data Pub │       │
└─────────────┘  └──────────┘  └────────────┘       │
                                                     │
┌────────────────────────────────────────────────────┘
│
│  EDGE COMPRESSION NODES
│  ┌─────────────────────┐  ┌─────────────────────┐
│  │ Node 0              │  │ Node 1              │
│  │ - Compressor        │  │ - Compressor        │
│  │ - Python Scripts    │  │ - Python Scripts    │
│  │ - Model Registry    │  │ - Model Registry    │
│  └─────────────────────┘  └─────────────────────┘
```

### Technology Stack

**Frontend:**
- React 18
- Recharts (visualizations)
- Material-UI
- WebSocket (real-time)

**Backend:**
- FastAPI (REST API)
- Motor (async MongoDB)
- Redis (caching)
- WebSockets (live updates)

**ML/AI:**
- PyTorch (models)
- Torchvision
- LPIPS (perceptual loss)
- Custom VAE, Transformers

**Infrastructure:**
- Docker & Docker Compose
- Nginx (reverse proxy)
- MongoDB (database)
- Redis (cache)
- ROS2 Humble (robot framework)

**Data:**
- MongoDB (persistent storage)
- Redis (real-time cache)
- File system (model storage)

---

##  Features

### 1. Advanced Neural Compression

- **Enhanced VAE** with β-VAE and progressive decoding
- **Cross-Modal Attention** for multimodal fusion
- **Temporal Transformers** for video/sequence data
- **Neural Delta Compression** for differential encoding
- **Quality Controller** for adaptive compression levels

### 2. Dynamic Model Management

- **Hot-swapping:** Replace models on live nodes
- **A/B Testing:** Compare model performance
- **Automatic Rollback:** Revert on performance degradation
- **Version Control:** Track model history
- **Model Registry:** Centralized model storage

### 3. Real-Time Python Execution

- **Dynamic Script Loading:** Load custom Python scripts at runtime
- **Hot Reloading:** Update scripts without restarting
- **Custom Processing:** Implement any processing logic
- **Isolated Execution:** Sandboxed script environments

### 4. Web Interface

- **Dashboard:** Real-time metrics overview
- **Nodes View:** Manage edge nodes
- **Models View:** Browse and upload models
- **Metrics View:** Historical performance data
- **Deployment View:** Deploy models to nodes

### 5. Multimodal Support

- **Image:** Camera, RGB-D
- **LiDAR:** Point clouds
- **IMU:** Accelerometer, gyroscope
- **Audio:** Microphone streams
- **Motor:** Actuator commands

### 6. Distributed System

- **Coordinator Node:** Global optimization
- **Bandwidth Allocation:** Dynamic resource management
- **Load Balancing:** Distribute workload
- **Fault Tolerance:** Handle node failures

### 7. Monitoring & Analytics

- **Real-time Metrics:** WebSocket streaming
- **Historical Data:** MongoDB storage
- **Performance Tracking:** Compression, latency, quality
- **System Stats:** CPU, memory, GPU, network

---

##  Installation

### Method 1: Docker (Recommended)

```bash
# Clone repository
cd /mnt/c/Users/josep/Documents/Lydlr/Lydlr

# Configure environment
cp .env.example .env

# Start system
./start-lydlr.sh --build
```

### Method 2: Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

**Database:**
```bash
# MongoDB
docker run -d -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=lydlr \
  -e MONGO_INITDB_ROOT_PASSWORD=lydlr_password \
  mongo:7.0

# Redis
docker run -d -p 6380:6379 redis:7-alpine
```

**ROS2:**
```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Build workspace
cd Lydlr
colcon build --symlink-install

# Source workspace
source install/setup.bash
```

---

##  Usage

### Starting the System

```bash
./start-lydlr.sh --build -d
```

### Training Models

```bash
# Activate virtual environment
source .venv/bin/activate

# Train models
python ros2/src/lydlr_ai/lydlr_ai/model/train_synthetic_models.py \
  --epochs 10 \
  --samples 1000 \
  --batch-size 8 \
  --version "v2.0"
```

### Deploying Models

**Via Web Interface:**
1. Go to http://localhost
2. Navigate to "Deploy" tab
3. Select model version
4. Choose target nodes
5. Click "Deploy Model"

**Via API:**
```bash
curl -X POST http://localhost:8000/api/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "model_version": "v2.0",
    "node_ids": ["node_0", "node_1"]
  }'
```

### Launching Edge Nodes

```bash
# Source ROS2
source install/setup.bash

# Launch node
ros2 run lydlr_ai edge_compressor_node --node-id node_0
```

### Monitoring Performance

**Web Interface:**
- Visit http://localhost
- Dashboard shows real-time metrics

**Command Line:**
```bash
# View logs
docker-compose logs -f backend

# Check metrics
curl http://localhost:8000/api/metrics

# System stats
curl http://localhost:8000/api/stats
```

---

##  Documentation

- **[DOCKER_DEPLOYMENT_GUIDE.md](../deployment/DOCKER_DEPLOYMENT_GUIDE.md)** - Complete Docker setup
- **[WEB_INTERFACE_GUIDE.md](../guides/WEB_INTERFACE_GUIDE.md)** - Web interface usage
- **[ARCHITECTURE.md](../architecture/ARCHITECTURE.md)** - System architecture details
- **[DESIGN_WALKTHROUGH.md](../architecture/DESIGN_WALKTHROUGH.md)** - Design decisions
- **[QUICK_START.md](../guides/QUICK_START.md)** - 5-minute quickstart
- **[README_LAUNCH.md](../guides/README_LAUNCH.md)** - Launch scripts guide

---

##  API Reference

### REST API

**Base URL:** `http://localhost:8000`

**Endpoints:**

```
GET    /health                    - Health check
GET    /api/stats                 - System statistics
GET    /api/nodes                 - List all nodes
GET    /api/nodes/{id}            - Get node details
POST   /api/nodes/{id}/start      - Start node
POST   /api/nodes/{id}/stop       - Stop node
POST   /api/nodes/{id}/restart    - Restart node
GET    /api/models                - List models
POST   /api/models/upload         - Upload model
POST   /api/deploy                - Deploy model
GET    /api/deployments           - Deployment history
GET    /api/metrics               - Get metrics
POST   /api/metrics               - Store metrics
```

**Full API docs:** http://localhost:8000/docs

### WebSocket

**Endpoint:** `ws://localhost:8000/ws/metrics`

**Message Types:**
```json
{
  "type": "metrics_update",
  "data": {
    "node_id": "node_0",
    "compression_ratio": 4.2,
    "latency_ms": 15.3,
    "quality_score": 0.95
  }
}
```

---

##  Troubleshooting

### Services won't start

```bash
# Check Docker
docker --version
docker-compose --version

# Check ports
sudo lsof -i :80
sudo lsof -i :8000
sudo lsof -i :27017

# View logs
docker-compose logs
```

### Frontend can't connect to backend

```bash
# Check backend
curl http://localhost:8000/health

# Check CORS settings in backend/main.py
```

### Models not training

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check dependencies
pip list | grep torch
```

### WebSocket disconnecting

```bash
# Check Redis
docker-compose logs redis

# Check backend WebSocket handler
docker-compose logs backend | grep WebSocket
```

---

##  Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

**Areas for contribution:**
- New compression models
- Additional sensor types
- UI/UX improvements
- Performance optimizations
- Documentation
- Tests

---

##  License

[Add your license here]

---

##  Acknowledgments

- ROS2 community
- PyTorch team
- FastAPI framework
- React ecosystem
- MongoDB and Redis teams

---

##  Support

- **Documentation:** See docs/ folder
- **Issues:** Create GitHub issue
- **Email:** [your-email]

---

##  What's Next?

**Roadmap:**
- [ ] GPU acceleration for inference
- [ ] Multi-GPU support
- [ ] Kubernetes deployment
- [ ] Cloud integration (AWS, Azure, GCP)
- [ ] Mobile app
- [ ] Advanced analytics
- [ ] ML model marketplace
- [ ] Federated learning

---

**Built with  for revolutionary compression**

**Start compressing smarter, not harder! **

