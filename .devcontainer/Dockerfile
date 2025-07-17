# Use ROS2 Humble base image
FROM osrf/ros:humble-desktop

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-pip \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /root/lydlr/lydlr_ws

# Copy your project files
COPY . .

# Create virtual environment inside container
RUN python3 -m venv /root/lydlr/lydlr_ws/.venv

# Upgrade pip inside venv and install dependencies
RUN /root/lydlr/lydlr_ws/.venv/bin/pip install --upgrade pip setuptools wheel

# If you have a requirements.txt file, install it
RUN if [ -f requirements.txt ]; then /root/lydlr/lydlr_ws/.venv/bin/pip install -r requirements.txt; fi

# Source ROS2 setup.bash and your venv in a script for convenience
RUN echo "#!/bin/bash\n\
source /opt/ros/humble/setup.bash\n\
source /root/lydlr/lydlr_ws/install/setup.bash\n\
source /root/lydlr/lydlr_ws/.venv/bin/activate\n\
exec \"\$@\"" > /usr/local/bin/ros2_entrypoint.sh && chmod +x /usr/local/bin/ros2_entrypoint.sh

# Entrypoint runs bash with ros2 and venv sourced
ENTRYPOINT ["/usr/local/bin/ros2_entrypoint.sh"]

# Default shell
CMD ["bash"]
