# GStreamer Integration for Lydlr

Comprehensive GStreamer integration with support for video processing, streaming, and AI inference.

## Features

- **Multiple Pipeline Types**: File compression, RTSP streaming, HLS, WebRTC
- **Codec Support**: H.264, H.265, VP8, VP9
- **NVIDIA Deepstream**: AI inference pipeline support
- **Health Checks**: Comprehensive diagnostics and monitoring
- **Plugin Management**: Dynamic plugin discovery and management
- **Error Handling**: Robust error handling and logging

## Installation

GStreamer packages are installed in the Docker container. For local development:

```bash
sudo apt-get update
sudo apt-get install -y \
    libgstreamer1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    python3-gi \
    python3-gst-1.0
```

## API Endpoints

### Status and Diagnostics

- `GET /api/gstreamer/status/` - GStreamer installation status
- `GET /api/gstreamer/health/` - Health check
- `GET /api/gstreamer/diagnostics/` - Detailed diagnostics
- `GET /api/gstreamer/plugins/` - List available plugins
- `GET /api/gstreamer/plugins/<name>/` - Plugin details
- `POST /api/gstreamer/test-pipeline/` - Test pipeline string

### Pipeline Creation

- `POST /api/gstreamer/pipeline/compression/` - Create compression pipeline
- `POST /api/gstreamer/pipeline/rtsp/` - Create RTSP streaming pipeline
- `POST /api/gstreamer/pipeline/hls/` - Create HLS streaming pipeline
- `POST /api/gstreamer/pipeline/webrtc/` - Create WebRTC pipeline

### Deepstream

- `GET /api/gstreamer/deepstream/status/` - Deepstream status
- `POST /api/gstreamer/deepstream/inference/` - Run Deepstream inference

## Usage Examples

### Video Compression

```python
from backend.gstreamer.gst_pipeline import VideoCompressionPipeline

pipeline = VideoCompressionPipeline(
    input_source="/path/to/input.mp4",
    output_sink="/path/to/output.mp4",
    compression_level=0.8,
    codec="x264"
)

if pipeline.create_pipeline():
    pipeline.start()
```

### RTSP Streaming

```python
from backend.gstreamer.gst_pipeline import RTSPStreamPipeline

pipeline = RTSPStreamPipeline(
    input_source="/dev/video0",
    rtsp_port=8554,
    mount_point="/test",
    codec="h264"
)

if pipeline.create_pipeline():
    pipeline.start()
    # Stream available at rtsp://localhost:8554/test
```

### HLS Streaming

```python
from backend.gstreamer.gst_pipeline import HLSPipeline

pipeline = HLSPipeline(
    input_source="/path/to/video.mp4",
    output_dir="/tmp/hls",
    playlist_name="playlist.m3u8",
    segment_duration=2
)

if pipeline.create_pipeline():
    pipeline.start()
```

### Using Pipeline Builder

```python
from backend.gstreamer.utils import PipelineBuilder

builder = PipelineBuilder()
pipeline_string = (builder
    .add_source("file", "/path/to/input.mp4")
    .add_decoder()
    .add_video_convert()
    .add_video_scale(1280, 720)
    .add_encoder("h264", bitrate=2000, speed_preset="ultrafast")
    .add_sink("file", "/path/to/output.mp4")
    .build())

print(pipeline_string)
```

## Configuration

Environment variables:

- `GSTREAMER_PLUGINS_PATH` - Path to GStreamer plugins (default: `/usr/lib/x86_64-linux-gnu/gstreamer-1.0`)
- `GSTREAMER_DEBUG` - Debug level (0-5)
- `DEEPSTREAM_ENABLED` - Enable Deepstream support (default: `False`)
- `NVIDIA_DEEPSTREAM_PATH` - Path to Deepstream installation

## Health Checks

The health check system provides:

- GStreamer installation verification
- Plugin availability
- Codec support
- Deepstream availability
- System information

Access via `/api/gstreamer/health/` endpoint.

## Pipeline Types

### VideoCompressionPipeline
Compresses video files with configurable codec and bitrate.

### RTSPStreamPipeline
Streams video over RTSP protocol.

### HLSPipeline
Creates HLS streaming with configurable segments.

### WebRTCPipeline
WebRTC streaming with STUN server support.

### DeepstreamPipeline
NVIDIA Deepstream AI inference pipeline.

## Error Handling

All pipelines include comprehensive error handling:

- Message bus monitoring
- State change tracking
- Error callbacks
- Statistics collection

## Logging

GStreamer operations are logged with appropriate levels:

- `ERROR`: Pipeline failures, errors
- `WARNING`: Warnings from GStreamer
- `INFO`: Pipeline state changes, starts/stops
- `DEBUG`: Detailed state information

## Performance

Pipelines include statistics tracking:

- Frames processed
- Bytes processed
- Errors and warnings
- Uptime

Access via `pipeline.get_stats()`.

## Troubleshooting

1. **GStreamer not found**: Check installation and PATH
2. **Plugin missing**: Install required plugin packages
3. **Pipeline fails**: Check pipeline string with `/api/gstreamer/test-pipeline/`
4. **Deepstream unavailable**: Verify NVIDIA Deepstream installation

## Development

To add new pipeline types:

1. Extend `GStreamerPipeline` class
2. Implement `create_pipeline()` method
3. Add API endpoint in `views.py`
4. Register URL in `urls.py`

