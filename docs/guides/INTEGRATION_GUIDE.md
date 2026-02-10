# Lydlr Script Integration Guide

## How `start-lydlr.sh` Uses `launch_lydlr_system.sh`

The `start-lydlr.sh` script now integrates with `launch_lydlr_system.sh` to provide a unified launch experience for both Docker services and ROS2 runtime.

---

## Integration Overview

**Two Launch Modes:**

1. **Docker Services Only** (Web Interface + Backend)
   - Frontend, Backend, MongoDB, Redis, Model Service, Nginx
   - No ROS2 nodes running

2. **Full System** (Docker + ROS2 Runtime)
   - All Docker services
   - Plus ROS2 edge nodes, synthetic publisher, deployment manager, coordinator

---

## Usage Examples

### Start Docker Services Only

```bash
./start-lydlr.sh --build -d
```

This starts:
- Frontend (React)
- Backend (FastAPI)
- MongoDB
- Redis
- Model Service
- Nginx

**Access:** http://localhost

---

### Start Full System (Docker + ROS2)

```bash
./start-lydlr.sh --build -d --ros2
```

This starts:
- All Docker services (above)
- ROS2 runtime via `launch_lydlr_system.sh`:
  - Synthetic multimodal publisher
  - Edge compression nodes (node_0, node_1, etc.)
  - Model deployment manager
  - Distributed coordinator

**Access:** http://localhost + ROS2 nodes running

---

### Launch ROS2 After Docker is Running

If Docker is already running, you can launch ROS2 separately:

```bash
./start-lydlr.sh --ros2
```

This will:
1. Check if Docker services are running
2. Launch `launch_lydlr_system.sh` in the background
3. Start all ROS2 nodes

---

## Command Reference

| Command | Description |
|---------|-------------|
| `./start-lydlr.sh --build -d` | Start Docker services only |
| `./start-lydlr.sh --build -d --ros2` | Start Docker + ROS2 runtime |
| `./start-lydlr.sh --ros2` | Launch ROS2 runtime (Docker must be running) |
| `./start-lydlr.sh --stop-ros2` | Stop ROS2 runtime only |
| `./start-lydlr.sh --stop` | Stop everything (Docker + ROS2) |
| `./start-lydlr.sh --status` | Show status of Docker + ROS2 |
| `./start-lydlr.sh --restart` | Restart Docker services |
| `./start-lydlr.sh --clean` | Clean up everything |

---

## How It Works

### Integration Flow

```
start-lydlr.sh
    │
    ├─► Check Docker/Compose
    │
    ├─► Start Docker Services
    │   ├─► Frontend
    │   ├─► Backend
    │   ├─► MongoDB
    │   ├─► Redis
    │   ├─► Model Service
    │   └─► Nginx
    │
    └─► If --ros2 flag:
        │
        ├─► Check launch_lydlr_system.sh exists
        ├─► Check ROS2 not already running
        ├─► Execute launch_lydlr_system.sh
        │   │
        │   ├─► Setup ROS2 environment
        │   ├─► Build ROS2 packages
        │   ├─► Deploy models
        │   ├─► Launch synthetic publisher
        │   ├─► Launch edge nodes
        │   ├─► Launch deployment manager
        │   └─► Launch coordinator
        │
        └─► Monitor ROS2 process
```

---

## Technical Details

### ROS2 Launch Process

When `--ros2` flag is used:

1. **Check Prerequisites:**
   ```bash
   - launch_lydlr_system.sh exists
   - ROS2 runtime not already running
   - Script is executable
   ```

2. **Launch in Background:**
   ```bash
   ./launch_lydlr_system.sh > /tmp/lydlr_ros2_launch.log 2>&1 &
   ```

3. **Process Management:**
   - PID stored for monitoring
   - Logs redirected to `/tmp/lydlr_ros2_launch.log`
   - Can be stopped with `--stop-ros2`

### Status Checking

The `--status` command shows:
- Docker container status
- ROS2 process status
- Running ROS2 nodes

### Cleanup

The `--stop` and `--clean` commands:
- Stop Docker containers
- Kill ROS2 processes (if running)
- Clean up volumes (with `--clean`)

---

## Monitoring

### Docker Services

```bash
# View Docker logs
docker-compose logs -f

# Check Docker status
docker-compose ps
```

### ROS2 Runtime

```bash
# View ROS2 launch logs
tail -f /tmp/lydlr_ros2_launch.log

# Check ROS2 processes
ps aux | grep "ros2 run lydlr_ai"

# Check ROS2 topics
ros2 topic list

# View node metrics
ros2 topic echo /node_0/metrics
```

### Combined Status

```bash
./start-lydlr.sh --status
```

Shows both Docker and ROS2 status.

---

## Use Cases

### Development (Web Interface Only)

```bash
./start-lydlr.sh --build -d
```

Use when:
- Testing frontend/backend
- Uploading models
- Viewing metrics
- No need for actual edge nodes

---

### Full System Testing

```bash
./start-lydlr.sh --build -d --ros2
```

Use when:
- Testing complete system
- Need edge nodes running
- Testing model deployment
- Monitoring real compression

---

### Production Deployment

**Option 1: Docker Only (Recommended for Cloud)**
```bash
./start-lydlr.sh --build -d
```

**Option 2: Full System (Edge Deployment)**
```bash
./start-lydlr.sh --build -d --ros2
```

---

## Troubleshooting

### ROS2 Won't Start

**Check:**
```bash
# Is launch_lydlr_system.sh executable?
chmod +x launch_lydlr_system.sh

# Check logs
cat /tmp/lydlr_ros2_launch.log

# Is ROS2 installed?
source /opt/ros/humble/setup.bash
ros2 --version
```

### ROS2 Already Running

**Solution:**
```bash
# Stop first
./start-lydlr.sh --stop-ros2

# Then start again
./start-lydlr.sh --ros2
```

### Docker Services Not Ready

**Wait for health check:**
```bash
# Check backend
curl http://localhost:8000/health

# Wait and retry
sleep 5
./start-lydlr.sh --ros2
```

---

## Best Practices

1. **Start Docker First:**
   ```bash
   ./start-lydlr.sh --build -d
   # Wait for services to be ready
   ./start-lydlr.sh --ros2
   ```

2. **Use Status Command:**
   ```bash
   ./start-lydlr.sh --status
   ```

3. **Monitor Logs:**
   ```bash
   # Docker
   docker-compose logs -f backend
   
   # ROS2
   tail -f /tmp/lydlr_ros2_launch.log
   ```

4. **Clean Shutdown:**
   ```bash
   ./start-lydlr.sh --stop
   ```

---

## Benefits of Integration

- **Single Command** to start everything  
- **Flexible** - Docker only or full system  
- **Unified Status** - Check everything at once  
- **Clean Shutdown** - Stop all components  
- **Process Management** - Automatic PID tracking  
- **Logging** - Centralized log locations  

---

**Now you can launch the entire Lydlr system with one command!**

