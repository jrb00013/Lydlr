FROM ros:humble

# Install Python and dependencies
# Note: rclpy is already included in the ros:humble base image
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    vim \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy workspace first (so we can access requirements.txt from it)
COPY ros2/src /app/src
COPY scripts /app/scripts
COPY launch /app/launch
COPY launch_lydlr_system.sh /app/
RUN chmod +x /app/launch_lydlr_system.sh

# Create virtual environment during build and install dependencies
# Note: rclpy is already included in the ros:humble base image and should not be installed via pip
RUN python3 -m venv /app/.venv && \
    /app/.venv/bin/pip install --upgrade pip setuptools wheel && \
    if [ -f /app/src/lydlr_ai/requirements.txt ]; then \
        grep -v "^rclpy" /app/src/lydlr_ai/requirements.txt > /tmp/requirements_no_rclpy.txt || true; \
        if [ -s /tmp/requirements_no_rclpy.txt ]; then \
            /app/.venv/bin/pip install --no-cache-dir -r /tmp/requirements_no_rclpy.txt; \
        else \
            /app/.venv/bin/pip install --no-cache-dir torch torchvision torchaudio scikit-image matplotlib psutil numpy lpips open3d; \
        fi; \
    else \
        /app/.venv/bin/pip install --no-cache-dir torch torchvision torchaudio scikit-image matplotlib psutil numpy lpips open3d; \
    fi && \
    /app/.venv/bin/pip install --no-cache-dir motor pymongo || true

# Create models directory structure
RUN mkdir -p /app/src/lydlr_ai/models && \
    mkdir -p /app/models && \
    mkdir -p /app/install && \
    mkdir -p /app/log

# Build ROS2 workspace (with error handling)
RUN . /opt/ros/humble/setup.sh && \
    cd /app && \
    colcon build --symlink-install --packages-select lydlr_ai || \
    (echo "Build failed, but continuing..." && \
     colcon build --symlink-install --packages-select lydlr_ai --cmake-args -DCMAKE_BUILD_TYPE=Release || true)

# Setup environment script
RUN echo '#!/bin/bash\n\
set -e\n\
export PYTHONPATH=/app/src:/app/src/lydlr_ai:${PYTHONPATH}\n\
export MODEL_DIR=/app/models\n\
export NODE_ID=${NODE_ID:-node_0}\n\
source /opt/ros/humble/setup.bash\n\
if [ -f /app/install/setup.bash ]; then\n\
    source /app/install/setup.bash\n\
fi\n\
exec "$@"' > /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

# Setup display for headless operation
RUN echo 'export XDG_RUNTIME_DIR=/tmp/runtime-root\n\
mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR\n\
if ! pgrep -x Xvfb > /dev/null; then\n\
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
    sleep 1\n\
fi\n\
export DISPLAY=:99' >> /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

