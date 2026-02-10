FROM python:3.11-slim

WORKDIR /app

# Install comprehensive GStreamer ecosystem and Docker CLI
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    pkg-config \
    docker.io \
    docker-compose \
    # Python development headers (needed for PyGObject)
    python3-dev \
    # Cairo development libraries (required for PyGObject/pycairo)
    libcairo2-dev \
    # GObject Introspection (needed for gi module and PyGObject)
    libgirepository1.0-dev \
    libglib2.0-dev \
    libffi-dev \
    gobject-introspection \
    # Core GStreamer
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    # GStreamer plugins - base set
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    # Additional codecs and formats
    gstreamer1.0-plugins-base-apps \
    # RTSP and streaming
    gstreamer1.0-rtsp \
    gstreamer1.0-nice \
    # WebRTC support
    libnice-dev \
    libsrtp2-dev \
    # Python bindings (system packages as fallback)
    python3-gi \
    python3-gi-cairo \
    python3-gst-1.0 \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    # NVIDIA support (if available)
    libgstreamer-plugins-bad1.0-dev \
    # Additional utilities
    v4l-utils \
    libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy Poetry files
COPY backend/pyproject.toml backend/poetry.lock* ./

# Configure Poetry to not create virtual environment (we're in a container)
RUN poetry config virtualenvs.create false

# Install dependencies - use requirements.txt for reliability
COPY backend/requirements.txt ./

# Use system python3-gi package (already installed above)
# The system package should work with the Python installation
# Note: We're using the system python3-gi instead of building PyGObject from pip
# because the pip build requires girepository-2.0 which has pkg-config issues in some Debian versions

# Verify gi module is importable (using system python3)
RUN python3 -c "import gi; print('✓ gi module available')" || \
    echo "WARNING: gi module not available"

# Try to verify GStreamer bindings (may not work if GStreamer not fully configured)
RUN python3 -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; Gst.init(None); print('✓ GStreamer Python bindings OK')" 2>/dev/null || \
    echo "INFO: GStreamer bindings verification skipped (may need runtime configuration)"

RUN pip install --no-cache-dir -r requirements.txt

# Verify daphne is installed and in PATH
RUN python -c "import daphne; print('daphne module found')" && \
    which daphne && daphne --version || \
    (echo "daphne not in PATH, installing..." && pip install --no-cache-dir daphne>=4.0.0 && which daphne)

# Copy application
COPY backend /app/backend

# Set environment variables
ENV PYTHONPATH=/app
ENV DJANGO_SETTINGS_MODULE=backend.backend.settings

# Collect static files
RUN cd /app && python manage.py collectstatic --noinput || true

EXPOSE 8000

# Use daphne for ASGI (supports WebSockets)
# Use python -m daphne as fallback if daphne command not in PATH
CMD python -m daphne -b 0.0.0.0 -p 8000 backend.backend.asgi:application
